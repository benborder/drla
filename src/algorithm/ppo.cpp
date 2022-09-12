#include "ppo.h"

#include "actor_critic_model.h"
#include "minibatch_buffer.h"
#include "utils.h"

#include <spdlog/spdlog.h>
#include <torch/torch.h>

#include <chrono>
#include <filesystem>
#include <memory>

// See https://costa.sh/blog-the-32-implementation-details-of-ppo.html for more details on PPO implementation

using namespace drla;

PPO::PPO(
		const Config::AgentTrainAlgorithm& config,
		const ObservationShapes& observation_shape,
		RolloutBuffer& buffer,
		std::shared_ptr<Model> model)
		: config_(std::get<Config::PPO>(config))
		, buffer_(buffer)
		, model_(std::dynamic_pointer_cast<ActorCriticModelInterface>(model))
		, optimiser_(model_->parameters(), torch::optim::AdamOptions(config_.learning_rate).eps(config_.epsilon))
{
	model_->train();
}

std::vector<UpdateResult> PPO::update(int timestep)
{
	update_learning_rate(timestep);

	auto values = buffer_.get_values().narrow(0, 0, buffer_.get_values().size(0) - 1);
	auto returns = buffer_.get_returns().narrow(0, 0, buffer_.get_returns().size(0) - 1);
	auto explained_var = explained_variance(values, returns);

	torch::Tensor loss;
	double total_value_loss = 0;
	double total_policy_loss = 0;
	double total_entropy_loss = 0;
	double kl_divergence = 0;
	double clip_fraction = 0;
	int update_count = 0;

	for (int epoch = 0; epoch < config_.num_epoch; ++epoch)
	{
		kl_divergence = 0;

		// Loop through randomised minibatches from the rollout buffer
		auto mini_batch_buffer = buffer_.get(config_.num_mini_batch);
		for (const auto& mini_batch : mini_batch_buffer)
		{
			auto eval_result = model_->evaluate_actions(mini_batch.observations, mini_batch.actions);
			total_entropy_loss += eval_result.dist_entropy.item<float>();

			auto advantages = mini_batch.advantages;
			if (config_.normalise_advantage)
			{
				advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8);
			}

			// ratio between old and new policy, should be one at the first iteration
			auto ratio = torch::exp(eval_result.action_log_probs - mini_batch.old_log_probs);

			// clipped surrogate loss
			auto policy_loss_1 = advantages * ratio;
			auto policy_loss_2 = advantages * torch::clamp(ratio, 1.0 - clip_policy_, 1.0 + clip_policy_);
			auto policy_loss = -torch::min(policy_loss_1, policy_loss_2).mean();
			total_policy_loss += policy_loss.item<float>();

			clip_fraction += (ratio - 1.0).abs().gt(clip_policy_).to(torch::kFloat).mean().item<float>();

			torch::Tensor values_pred;
			if (config_.clip_vf)
			{
				values_pred =
						mini_batch.old_values + torch::clamp(eval_result.values - mini_batch.old_values, -clip_vf_, clip_vf_);
			}
			else
			{
				values_pred = eval_result.values;
			}

			// Value loss using the TD(gae_lambda) target
			auto value_loss = torch::nn::functional::mse_loss(mini_batch.returns, values_pred);
			total_value_loss += value_loss.item<float>();

			// Total loss
			loss =
					(value_loss * config_.value_loss_coef + policy_loss * config_.policy_loss_coef -
					 eval_result.dist_entropy * config_.entropy_coef);

			// Calculate approximate form of reverse KL Divergence for early stopping
			// see Schulman blog: http://joschu.net/blog/kl-approx.html
			{
				torch::NoGradGuard no_grad;
				auto log_ratio = eval_result.action_log_probs - mini_batch.old_log_probs;
				kl_divergence += ((torch::exp(log_ratio) - 1) - log_ratio).mean().item<float>();
			}

			// Backprop and step optimiser_
			optimiser_.zero_grad();
			loss.backward();
			torch::nn::utils::clip_grad_norm_(model_->parameters(), config_.max_grad_norm);
			optimiser_.step();

			update_count++;
		}

		kl_divergence /= static_cast<double>(mini_batch_buffer.size());
		if (kl_divergence > (config_.kl_target * 1.5))
		{
			break;
		}
	}

	total_value_loss /= update_count;
	total_policy_loss /= update_count;
	total_entropy_loss /= update_count;
	clip_fraction /= update_count;

	return {
			{TrainResultType::kLoss, loss.mean().item<float>()},
			{TrainResultType::kValueLoss, total_value_loss},
			{TrainResultType::kPolicyLoss, total_policy_loss},
			{TrainResultType::kEntropyLoss, total_entropy_loss},
			{TrainResultType::kClipFraction, clip_fraction},
			{TrainResultType::kKLDivergence, kl_divergence},
			{TrainResultType::kLearningRate, lr_},
			{TrainResultType::kExplainedVariance, explained_var}};
}

void PPO::save(const std::filesystem::path& path) const
{
	torch::save(optimiser_, path / "optimiser.pt");
	model_->save(path);
}

void PPO::load(const std::filesystem::path& path)
{
	auto opt_path = path / "optimiser.pt";
	if (std::filesystem::exists(opt_path))
	{
		torch::load(optimiser_, opt_path);
		spdlog::info("Optimiser loaded");
	}
	model_->load(path);
	model_->train();
}

void PPO::update_learning_rate(int timestep)
{
	double alpha = learning_rate_decay(&config_, timestep, config_.total_timesteps);
	lr_ = config_.learning_rate * alpha;
	clip_policy_ = config_.clip_range_policy * alpha;
	clip_vf_ = config_.clip_range_vf * alpha;
	for (auto& group : optimiser_.param_groups())
	{
		if (group.has_options())
		{
			dynamic_cast<torch::optim::AdamOptions&>(group.options()).lr(lr_);
		}
	}
}
