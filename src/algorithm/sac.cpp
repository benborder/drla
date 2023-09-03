#include "sac.h"

#include "soft_actor_critic_model.h"
#include "utils.h"

#include <spdlog/spdlog.h>
#include <torch/torch.h>

#include <chrono>
#include <filesystem>
#include <memory>

using namespace drla;

SAC::SAC(
	const Config::AgentTrainAlgorithm& config,
	const ActionSpace& action_space,
	ReplayBuffer& buffer,
	std::shared_ptr<Model> model)
		: config_(std::get<Config::SAC>(config))
		, action_space_(action_space)
		, buffer_(buffer)
		, model_(std::dynamic_pointer_cast<SoftActorCriticModel>(model))
		, log_ent_coef_(torch::ones(1, buffer_.get_device()).log().requires_grad_())
		, actor_optimiser_(config_.actor_optimiser, model_->actor_parameters(), config_.total_timesteps)
		, critic_optimiser_(config_.critic_optimiser, model_->critic_parameters(), config_.total_timesteps)
		, ent_coef_optimiser_(config_.ent_coef_optimiser, {log_ent_coef_}, config_.total_timesteps)
		, target_entropy_(-static_cast<double>(action_space_.shape.size()))
{
	if (is_action_discrete(action_space_))
	{
		target_entropy_ = -config_.target_entropy_scale *
											std::log(1.0 / std::accumulate(action_space_.shape.begin(), action_space_.shape.end(), 0));
	}
	model_->train();
}

std::string SAC::name() const
{
	return "SAC";
}

Metrics SAC::update(int timestep)
{
	float ent_coefs = 0.0F;
	float ent_coef_losses = 0.0F;
	float critic_losses = 0.0F;
	float actor_losses = 0.0F;

	auto [ratio_ent, lr_ent] = ent_coef_optimiser_.update(timestep);
	auto [ratio_critic, lr_critic] = critic_optimiser_.update(timestep);
	auto [ratio_actor, lr_actor] = actor_optimiser_.update(timestep);

	for (int gradient_step = 0; gradient_step < config_.gradient_steps; gradient_step++)
	{
		auto replay_data = buffer_.sample(config_.batch_size);

		// Action by the current actor for the sampled state
		auto action_output = model_->action_output(replay_data.observations, replay_data.state);

		// The entropy coefficient or entropy can be learned automatically see Automating Entropy Adjustment for Maximum
		// Entropy RL section of https://arxiv.org/abs/1812.05905
		auto ent_coef = log_ent_coef_.detach().exp();
		ent_coefs += ent_coef.item<float>();
		auto ent_coef_loss = -(log_ent_coef_ * (action_output.log_prob + target_entropy_).detach());
		if (is_action_discrete(action_space_))
		{
			auto probs = torch::softmax(action_output.actions_pi, -1);
			ent_coef_loss *= probs.detach();
		}
		ent_coef_loss = ent_coef_loss.mean();
		ent_coef_losses += ent_coef_loss.item<float>();

		// Optimize entropy coefficient, also called entropy temperature or alpha in the paper
		ent_coef_optimiser_.step(ent_coef_loss);

		torch::Tensor target_q_values;
		{
			torch::NoGradGuard no_grad;
			// Select action according to policy
			auto next_action_output = model_->action_output(replay_data.next_observations, replay_data.next_state);
			// Compute the next Q values: min over all critics targets
			auto next_q_values = torch::stack(
				model_->critic_target(replay_data.next_observations, next_action_output.action, replay_data.next_state));
			next_q_values = std::get<0>(torch::min(next_q_values, 0));
			if (is_action_discrete(action_space_))
			{
				// get expected value and add entropy term
				auto next_probs = torch::softmax(next_action_output.actions_pi, -1);
				next_q_values = (next_probs * (next_q_values - ent_coef * next_action_output.log_prob)).sum(-1, true);
			}
			else
			{
				// add entropy term
				next_q_values -= ent_coef * next_action_output.log_prob;
			}
			// td error
			target_q_values = replay_data.rewards + replay_data.episode_non_terminal * buffer_.get_gamma() * next_q_values;
		}

		// Get current Q-values estimates for each critic network using action from the replay buffer
		auto current_q_values_list = model_->critic(replay_data.observations, replay_data.actions, replay_data.state);
		torch::Tensor critic_loss = torch::zeros({1}, target_q_values.device());
		for (auto& current_q_values : current_q_values_list)
		{
			if (is_action_discrete(action_space_))
			{
				current_q_values = current_q_values.gather(1, replay_data.actions.to(torch::kLong));
			}
			// Compute critic loss
			critic_loss += config_.value_loss_coef * torch::nn::functional::mse_loss(current_q_values, target_q_values);
		}

		auto values = std::get<0>(torch::min(torch::stack(current_q_values_list), 0));
		auto priorities =
			torch::pow((values.detach() - replay_data.values).abs(), config_.per_alpha).sum(-1).clamp_min(1e-8);
		buffer_.update_priorities(priorities, replay_data.indicies);

		critic_losses += critic_loss.item<float>();

		// Optimize the critic
		critic_optimiser_.step(critic_loss);

		// Compute actor loss
		torch::Tensor min_qf_pi;
		{
			torch::NoGradGuard no_grad;
			auto q_values_pi = model_->critic(replay_data.observations, action_output.actions_pi, replay_data.state);
			min_qf_pi = std::get<0>(torch::min(torch::stack(q_values_pi), 0));
		}
		auto actor_loss = config_.actor_loss_coef * (ent_coef * action_output.log_prob - min_qf_pi);
		if (is_action_discrete(action_space_))
		{
			auto probs = torch::softmax(action_output.actions_pi, -1);
			actor_loss *= probs;
		}
		// TODO: replay_data.weight;
		actor_loss = actor_loss.mean();
		actor_losses += actor_loss.item<float>();

		// Optimize the actor
		actor_optimiser_.step(actor_loss);
	}

	// Update target networks
	if (timestep % config_.target_update_interval == 0)
	{
		model_->update(config_.tau);
	}

	ent_coefs /= static_cast<float>(config_.gradient_steps);
	ent_coef_losses /= static_cast<float>(config_.gradient_steps);
	critic_losses /= static_cast<float>(config_.gradient_steps);
	actor_losses /= static_cast<float>(config_.gradient_steps);

	Metrics metrics;
	metrics.add({"loss", TrainResultType::kLoss, critic_losses});
	metrics.add({"loss_policy", TrainResultType::kLoss, actor_losses});
	metrics.add({"loss_entropy", TrainResultType::kLoss, ent_coef_losses});
	metrics.add({"learning_rate_ent", TrainResultType::kLearningRate, static_cast<float>(lr_ent)});
	metrics.add({"learning_rate_critic", TrainResultType::kLearningRate, static_cast<float>(lr_critic)});
	metrics.add({"learning_rate_actor", TrainResultType::kLearningRate, static_cast<float>(lr_actor)});
	metrics.add({"entropy_coeficients", TrainResultType::kRegularisation, ent_coefs});
	return metrics;
}

void SAC::save(const std::filesystem::path& path) const
{
	torch::save(actor_optimiser_.get_optimiser(), path / "actor_optimiser.pt");
	torch::save(critic_optimiser_.get_optimiser(), path / "critic_optimiser.pt");
	torch::save(ent_coef_optimiser_.get_optimiser(), path / "ent_coef_optimiser.pt");
	model_->save(path);
}

void SAC::load(const std::filesystem::path& path)
{
	auto opt_actor_path = path / "actor_optimiser.pt";
	auto opt_critic_path = path / "critic_optimiser.pt";
	auto opt_ent_coef_path = path / "ent_coef_optimiser.pt";
	if (std::filesystem::exists(opt_actor_path) && std::filesystem::exists(opt_critic_path))
	{
		torch::load(actor_optimiser_.get_optimiser(), opt_actor_path);
		torch::load(critic_optimiser_.get_optimiser(), opt_critic_path);
		torch::load(ent_coef_optimiser_.get_optimiser(), opt_ent_coef_path);
		spdlog::info("Optimisers loaded");
	}
	model_->load(path);
	model_->train();
}
