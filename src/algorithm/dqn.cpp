#include "dqn.h"

#include "qnet_model.h"
#include "utils.h"

#include <spdlog/spdlog.h>
#include <torch/torch.h>

#include <chrono>
#include <filesystem>
#include <memory>

using namespace drla;

DQN::DQN(const Config::AgentTrainAlgorithm& config, ReplayBuffer& buffer, std::shared_ptr<Model> model)
		: config_(std::get<Config::DQN>(config))
		, buffer_(buffer)
		, model_(std::dynamic_pointer_cast<QNetModelInterface>(model))
		, optimiser_(config_.optimiser, model_->parameters(), config_.total_timesteps)
{
	model_->train();
	update_exploration(0);
}

std::string DQN::name() const
{
	return "DQN";
}

Metrics DQN::update(int timestep)
{
	update_exploration(timestep);

	optimiser_.update(timestep);

	float total_loss = 0;

	for (int steps = 0; steps < config_.gradient_steps; steps++, n_updates_++)
	{
		auto replay_data = buffer_.sample(config_.batch_size);

		torch::Tensor target_q_values;
		{
			torch::NoGradGuard no_grad;
			auto next_q_values = model_->forward_target(replay_data.next_observations, replay_data.next_state);
			std::tie(next_q_values, std::ignore) = next_q_values.max(2);
			target_q_values = replay_data.rewards + replay_data.episode_non_terminal * buffer_.get_gamma() * next_q_values;
		}
		auto current_q_values = model_->forward(replay_data.observations, replay_data.state);
		current_q_values =
			torch::gather(current_q_values, 2, replay_data.actions.expand_as(target_q_values).unsqueeze(1).to(torch::kLong))
				.squeeze(1);

		// TODO: add weight for importance sampling
		auto loss = torch::nn::functional::smooth_l1_loss(current_q_values, target_q_values);
		total_loss += loss.item<float>();

		auto priorities =
			torch::pow((current_q_values.detach() - replay_data.values).abs(), config_.per_alpha).sum(-1).clamp_min(1e-8);
		buffer_.update_priorities(priorities, replay_data.indicies);

		optimiser_.step(loss);

		if (n_updates_ % config_.target_update_interval == 0)
		{
			model_->update(config_.tau);
		}
	}

	Metrics metrics;
	metrics.add({"loss", TrainResultType::kLoss, total_loss / config_.gradient_steps});
	metrics.add({"learning_rate", TrainResultType::kLearningRate, static_cast<float>(optimiser_.get_lr())});
	metrics.add({"exploration", TrainResultType::kPolicyEvaluation, static_cast<float>(exploration_param_)});
	return metrics;
}

void DQN::save(const std::filesystem::path& path) const
{
	torch::save(optimiser_.get_optimiser(), path / "optimiser.pt");
	model_->save(path);
}

void DQN::load(const std::filesystem::path& path)
{
	if (std::filesystem::exists(path / "optimiser.pt"))
	{
		torch::load(optimiser_.get_optimiser(), path / "optimiser.pt");
		spdlog::info("Optimiser loaded");
	}
	model_->load(path);
	model_->train();
}

void DQN::update_exploration(int timestep)
{
	double progress = static_cast<double>(timestep) / static_cast<double>(config_.total_timesteps);
	if (progress > config_.exploration_fraction)
	{
		exploration_param_ = config_.exploration_final;
	}
	else
	{
		exploration_param_ = config_.exploration_init + progress * (config_.exploration_final - config_.exploration_init) /
																											config_.exploration_fraction;
	}
	model_->set_exploration(exploration_param_);
}
