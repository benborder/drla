// This is an implementation of the Dreamer v3 training algorithm. The following papers were used as reference:
// Dreamerv3 - https://arxiv.org/abs/2301.04104
// Dreamerv2 - https://arxiv.org/abs/2010.02193
// Dreamerv1 - https://arxiv.org/abs/1912.01603
// Many implementation details were not included in the paper, but code was provided by the author:
// https://github.com/danijar/dreamerv3

#include "dreamer.h"

#include "bernoulli.h"
#include "discrete.h"
#include "dreamer_model.h"
#include "normal.h"

#include <spdlog/spdlog.h>
#include <torch/torch.h>

#include <chrono>
#include <filesystem>
#include <memory>

using namespace drla;

Dreamer::Dreamer(
	const Config::AgentTrainAlgorithm& config,
	const ActionSpace& action_space,
	std::shared_ptr<HybridModelInterface> model,
	HybridReplayBuffer& buffer)
		: config_(std::get<Config::Dreamer::TrainConfig>(config))
		, action_space_(action_space)
		, model_(model)
		, buffer_(buffer)
		, world_optimiser_(config_.world_optimiser, model_->world_model_parameters(), config_.total_timesteps)
		, actor_optimiser_(config_.actor_optimiser, model_->actor_parameters(), config_.total_timesteps)
		, critic_optimiser_(config_.critic_optimiser, model_->critic_parameters(), config_.total_timesteps)
		, moments_(std::make_shared<Moments>())
{
	moments_->to(model_->critic_parameters().back().device());
	moments_->train();
	model_->train();
}

std::string Dreamer::name() const
{
	return "Dreamer";
}

Metrics Dreamer::update(int timestep)
{
	if (timestep == 0)
	{
		// make the target initially equal the critic
		model_->update(1.0);
	}

	world_optimiser_.update(timestep);
	actor_optimiser_.update(timestep);
	critic_optimiser_.update(timestep);

	Metrics metrics;
	metrics.add({"wm_learning_rate", TrainResultType::kLearningRate, world_optimiser_.get_lr()});
	metrics.add({"actor_learning_rate", TrainResultType::kLearningRate, actor_optimiser_.get_lr()});
	metrics.add({"critic_learning_rate", TrainResultType::kLearningRate, critic_optimiser_.get_lr()});

	ImaginedTrajectory imagined_trajectory;
	{
		auto device = model_->parameters().front().device();
		// batch has shape [batch, rollout, ...]
		auto batch = buffer_.sample(config_.batch_size, device);
		metrics.add("real_obs", batch.observation);

		// WorldModel loss
		auto wm_output = model_->evaluate_world_model(batch.observation, batch.action, batch.states, batch.is_first);
		world_model_loss(wm_output, batch, metrics);
		wm_output.observation.clear(); // clear to save some ram

		// Use all samples from batch and imagine trajectories for all, so reshape dims to [batch*rollout, ..]
		imagined_trajectory = model_->imagine_trajectory(config_.horizon, wm_output);
		{
			torch::NoGradGuard no_grad;
			imagined_trajectory.non_terminal = Bernoulli({}, imagined_trajectory.non_terminal).mode().squeeze(-1);
			// repalce first step from imagined non_terminal (a prediction) with first step from batch (measured)
			imagined_trajectory.non_terminal[0] = batch.non_terminal.view({-1});

			// used to weight the loss terms of the actor and critic by the cumulative predicted discount factors to softly
			// account for the possibility of episode ends
			imagined_trajectory.weight =
				torch::cumprod(config_.discount * imagined_trajectory.non_terminal, 0) / config_.discount;

			if (config_.use_per)
			{
				// Update priorities for PER buffer
				auto priorities =
					torch::pow((wm_output.values.view_as(batch.values) - batch.values).abs(), config_.per_alpha)
						// Clamping to 1e-8 to avoid 0 as this can lead to divide by zero issues in some circumstances
						.clamp_min(1e-8);
				buffer_.update_priorities(priorities.to(torch::kCPU), batch.indicies);
				// TODO: maybe rather than using values, use the latents to determine the priorities
			}
		}
	}

	behavioural_model_loss(imagined_trajectory, metrics);

	model_->update(config_.tau);

	return metrics;
}

void Dreamer::world_model_loss(const WorldModelOutput& wm_output, const HybridBatch& batch, Metrics& metrics)
{
	auto device = model_->parameters().front().device();
	auto obs_loss = torch::zeros(wm_output.non_terminal.size(0), device);
	Observations dec_observations;
	for (size_t i = 0; i < batch.observation.size(); ++i)
	{
		auto& dec_obs = wm_output.observation[i];
		auto obs = batch.observation[i].view_as(dec_obs);
		if (dec_obs.dim() <= 3)
		{
			obs = symlog(obs);
			dec_observations.push_back(symexp(dec_obs).view_as(batch.observation[i]));
		}
		else
		{
			dec_observations.push_back(dec_obs.view_as(batch.observation[i]));
		}
		auto loss = mse_logprob_loss(dec_obs, obs, obs.dim() - 1);
		obs_loss -= loss;
	}
	auto reward_loss = -Discrete(wm_output.reward).log_prob(symlog(batch.reward.view({-1, batch.reward.size(-1)})));
	auto nterm_loss = -Bernoulli({}, wm_output.non_terminal).log_prob(batch.non_terminal.view({-1, 1})).sum(-1);

	// The prediction loss trains the decoder and reward predictor via the symlog loss and the continue (non terminal)
	// predictor via binary classification loss.
	auto pred_loss = obs_loss + reward_loss + nterm_loss;

	const torch::Tensor& post_logits = wm_output.latents[2];
	const torch::Tensor& prior_logits = wm_output.latents[3];

	// The dynamics loss trains the sequence model to predict the next representation by minimizing the KL divergence
	// between the predictor p_φ(z_t | h_t) and the next stochastic representation q_φ(z_t | h_t, x_t).
	auto kl_dyn = kl_divergence_independent(Categorical({}, post_logits.detach()), Categorical({}, prior_logits));
	auto dyn_loss = torch::maximum(kl_dyn, torch::ones(1, device));
	// The representation loss trains the representations to become more predictable if the dynamics cannot predict
	// their distribution, allowing us to use a factorized dynamics predictor for fast sampling when training the actor
	// critic.
	auto kl_rep = kl_divergence_independent(Categorical({}, post_logits), Categorical({}, prior_logits.detach()));
	auto rep_loss = torch::maximum(kl_rep, torch::ones(1, device));
	auto kl_loss = config_.dyn_beta * dyn_loss + config_.rep_beta * rep_loss;

	auto wm_loss = config_.pred_beta * pred_loss + kl_loss;
	if (config_.use_per)
	{
		wm_loss = wm_loss.view({batch.weight.size(0), -1});
		wm_loss *= batch.weight.unsqueeze(-1).broadcast_to(wm_loss.sizes());
	}
	wm_loss = wm_loss.mean();

	// WorldModel backprop and step optimiser
	world_optimiser_.step(wm_loss);

	torch::NoGradGuard no_grad;
	auto dims = batch.action.sizes().slice(0, 2).vec();
	auto h = wm_output.latents[0].view(dims + auto_resize(wm_output.latents[0].sizes().slice(1).vec(), 3));
	auto z = wm_output.latents[1].view(dims + auto_resize(wm_output.latents[1].sizes().slice(1).vec()));

	metrics.add("dec_obs", dec_observations);
	metrics.add("h", h);
	metrics.add("z", z);
	metrics.add({"loss_observations", TrainResultType::kLoss, obs_loss.mean().item<float>()});
	metrics.add({"loss_reward", TrainResultType::kLoss, reward_loss.mean().item<float>()});
	metrics.add({"loss_continue", TrainResultType::kLoss, nterm_loss.mean().item<float>()});
	metrics.add({"loss_prediction", TrainResultType::kLoss, pred_loss.mean().item<float>()});
	metrics.add({"loss_kl", TrainResultType::kLoss, kl_loss.mean().item<float>()});
	metrics.add({"loss_world_model", TrainResultType::kLoss, wm_loss.item<float>()});
	metrics.add({"kl_divergence", TrainResultType::kRegularisation, kl_dyn.mean().item<float>()});
}

void Dreamer::behavioural_model_loss(const ImaginedTrajectory& imagined_trajectory, Metrics& metrics)
{
	using namespace torch::indexing;

	auto bm_output = model_->evaluate_behavioural_model(imagined_trajectory.latents, imagined_trajectory.action);

	torch::Tensor returns;
	torch::Tensor values;
	{
		torch::NoGradGuard no_grad;
		Discrete value_dist(bm_output.values);
		values = symexp(value_dist.mean());
		returns = calculate_returns(imagined_trajectory.reward, values.index({Slice(1)}), imagined_trajectory.non_terminal);
		values = values.index({Slice(0, -1)});
	}

	// Actor Loss
	auto [offset, invscale] = moments_->forward(returns);
	auto normalised_return = (returns - offset) / invscale;
	auto normalised_values = (values - offset) / invscale;
	auto advantage = (normalised_return - normalised_values).mean(-1);
	torch::Tensor actor_loss;
	if (is_action_discrete(action_space_))
	{
		// Use reinforce for discrete action space
		actor_loss = -bm_output.log_probs.index({Slice(0, -1)}) * advantage.detach();
	}
	else
	{
		// Use backprop of advantages for continuous action space
		actor_loss = -advantage;
	}
	actor_loss -= config_.actor_entropy_scale * bm_output.entropy.index({Slice(0, -1)});
	actor_loss *= imagined_trajectory.weight.index({Slice(0, -1)});
	actor_loss = torch::mean(actor_loss * config_.actor_loss_scale);

	// Actor backprop and step optimiser
	actor_optimiser_.step(actor_loss);

	// Critic Loss
	int reward_shape = values.size(-1);
	int bins = bm_output.values.size(-1);
	Discrete value_dist(bm_output.values.index({Slice(0, -1)}).view({-1, reward_shape, bins}));
	Discrete target_value_dist(bm_output.target_values.index({Slice(0, -1)}).view({-1, reward_shape, bins}));
	auto value_loss = -value_dist.log_prob(symlog(returns.view({-1, reward_shape}).detach()));
	auto reg = -value_dist.log_prob(target_value_dist.mean().view({-1, reward_shape}).detach());
	value_loss += config_.target_regularisation_scale * reg;
	value_loss *= imagined_trajectory.weight.index({Slice(0, -1)}).view({-1});
	value_loss = value_loss.mean() * config_.critic_loss_scale;

	// Critic backprop and step optimiser
	critic_optimiser_.step(value_loss);

	metrics.add({"loss_policy", TrainResultType::kLoss, actor_loss.item<float>()});
	metrics.add({"loss_value", TrainResultType::kLoss, value_loss.item<float>()});
	metrics.add({"advantage", TrainResultType::kPerformanceEvaluation, advantage.mean().item<float>()});
	metrics.add({"entropy", TrainResultType::kPolicyEvaluation, bm_output.entropy.mean().item<float>()});
}

torch::Tensor Dreamer::calculate_returns(
	const torch::Tensor& reward, const torch::Tensor& value, const torch::Tensor& non_terminal) const
{
	using namespace torch::indexing;
	std::vector<torch::Tensor> vals{value[-1]};
	auto disc = config_.discount * non_terminal.index({Slice(1)}).unsqueeze(-1);
	auto interm = reward + disc * value * (1.0F - config_.return_lambda);
	for (int t = config_.horizon - 1; t >= 0; --t)
	{
		vals.push_back(interm[t] + disc[t] * config_.return_lambda * vals.back());
	}
	std::reverse(vals.begin(), vals.end());
	vals.pop_back();
	return torch::stack(vals);
}

void Dreamer::save(const std::filesystem::path& path) const
{
	torch::save(world_optimiser_.get_optimiser(), path / "wm_optimiser.pt");
	torch::save(actor_optimiser_.get_optimiser(), path / "actor_optimiser.pt");
	torch::save(critic_optimiser_.get_optimiser(), path / "critic_optimiser.pt");
	torch::save(std::dynamic_pointer_cast<torch::nn::Module>(moments_), path / "moments.pt");
	model_->save(path);
}

void Dreamer::load(const std::filesystem::path& path)
{
	auto load_path = path / "wm_optimiser.pt";
	if (std::filesystem::exists(load_path))
	{
		torch::load(world_optimiser_.get_optimiser(), load_path);
		spdlog::info("World optimiser loaded");
	}
	load_path = path / "actor_optimiser.pt";
	if (std::filesystem::exists(load_path))
	{
		torch::load(actor_optimiser_.get_optimiser(), load_path);
		spdlog::info("Actor optimiser loaded");
	}
	load_path = path / "critic_optimiser.pt";
	if (std::filesystem::exists(load_path))
	{
		torch::load(critic_optimiser_.get_optimiser(), load_path);
		spdlog::info("Critic optimiser loaded");
	}
	load_path = path / "moments.pt";
	if (std::filesystem::exists(load_path))
	{
		auto moments = std::dynamic_pointer_cast<torch::nn::Module>(moments_);
		torch::load(moments, load_path);
		spdlog::info("Moments loaded");
	}
	model_->load(path);
	model_->train();
}
