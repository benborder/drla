#include "a2c.h"

#include "actor_critic_model.h"
#include "utils.h"

#include <spdlog/spdlog.h>
#include <torch/torch.h>

#include <chrono>
#include <filesystem>
#include <memory>

using namespace drla;

A2C::A2C(
		const Config::AgentTrainAlgorithm& config,
		const ObservationShapes& observation_shape,
		RolloutBuffer& buffer,
		std::shared_ptr<Model> model)
		: config_(std::get<Config::A2C>(config))
		, buffer_(buffer)
		, model_(std::dynamic_pointer_cast<ActorCriticModelInterface>(model))
		, optimiser_(
					model_->parameters(),
					torch::optim::RMSpropOptions(config_.learning_rate).eps(config_.epsilon).alpha(config_.alpha))
{
	model_->train();
}

std::vector<UpdateResult> A2C::update(int timestep)
{
	update_learning_rate(timestep);

	auto action_shape = buffer_.get_actions().size(-1);
	auto rewards_shape = buffer_.get_rewards().sizes();
	int n_steps = rewards_shape[0];
	int n_envs = rewards_shape[1];

	auto observations = buffer_.get_observations();
	for (size_t i = 0; i < observations.size(); i++)
	{
		auto observations_shape = observations[i].sizes().vec();
		observations_shape.erase(observations_shape.begin());
		observations_shape[0] = -1;
		observations[i] = observations[i].narrow(0, 0, n_steps).view(observations_shape).to(buffer_.get_device());
	}
	auto evaluate_result = model_->evaluate_actions(observations, buffer_.get_actions().view({-1, action_shape}));
	observations.clear();
	auto values = evaluate_result.values.view({n_steps, n_envs, evaluate_result.values.size(1)});
	auto action_log_probs = evaluate_result.action_log_probs.view({n_steps, n_envs, 1});

	auto advantages = buffer_.get_advantages();
	if (config_.normalise_advantage)
	{
		advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8);
	}

	auto policy_loss = -(advantages.detach().narrow(0, 0, n_steps) * action_log_probs).mean();

	auto value_loss = torch::nn::functional::mse_loss(
			buffer_.get_returns().narrow(0, 0, n_steps).view({n_steps, n_envs, evaluate_result.values.size(1)}), values);

	// Total loss
	auto loss = value_loss * config_.value_loss_coef + policy_loss * config_.policy_loss_coef -
							evaluate_result.dist_entropy * config_.entropy_coef;

	// Backprop and step optimiser_
	optimiser_.zero_grad();
	loss.backward();
	torch::nn::utils::clip_grad_norm_(model_->parameters(), config_.max_grad_norm);
	optimiser_.step();

	auto explained_var = explained_variance(buffer_.get_values(), buffer_.get_returns());

	return {
			{TrainResultType::kLoss, loss.mean().item<float>()},
			{TrainResultType::kValueLoss, value_loss.item<float>()},
			{TrainResultType::kPolicyLoss, policy_loss.item<float>()},
			{TrainResultType::kEntropyLoss, evaluate_result.dist_entropy.item<float>()},
			{TrainResultType::kLearningRate, lr_param_},
			{TrainResultType::kExplainedVariance, explained_var}};
}

void A2C::save(const std::filesystem::path& path) const
{
	torch::save(optimiser_, path / "optimiser.pt");
	model_->save(path);
}

void A2C::load(const std::filesystem::path& path)
{
	if (std::filesystem::exists(path / "optimiser.pt"))
	{
		torch::load(optimiser_, path / "optimiser.pt");
		spdlog::info("Optimiser loaded");
	}
	model_->load(path);
	model_->train();
}

void A2C::update_learning_rate(int timestep)
{
	double alpha = learning_rate_decay(&config_, timestep, config_.total_timesteps);
	lr_param_ = config_.learning_rate * alpha;
	for (auto& group : optimiser_.param_groups())
	{
		if (group.has_options())
		{
			dynamic_cast<torch::optim::RMSpropOptions&>(group.options()).lr(lr_param_);
		}
	}
}
