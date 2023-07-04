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
		, optimiser_(model_->parameters(), torch::optim::AdamOptions(config_.learning_rate).eps(config_.epsilon))
{
	model_->train();
	update_exploration(0);
}

std::string DQN::name() const
{
	return "DQN";
}

std::vector<UpdateResult> DQN::update(int timestep)
{
	update_learning_rate(timestep);
	update_exploration(timestep);

	float total_loss = 0;

	for (int steps = 0; steps < config_.gradient_steps; steps++, n_updates_++)
	{
		auto replay_data = buffer_.sample(config_.batch_size);

		torch::Tensor target_q_values;
		{
			torch::NoGradGuard no_grad;
			auto next_q_values = model_->forward_target(replay_data.next_observations, replay_data.next_state);
			std::tie(next_q_values, std::ignore) = next_q_values.max(1);
			next_q_values = next_q_values.reshape({-1, 1});
			target_q_values = replay_data.rewards + replay_data.episode_non_terminal * buffer_.get_gamma() * next_q_values;
		}
		auto current_q_values = model_->forward(replay_data.observations, replay_data.state);
		current_q_values = torch::gather(current_q_values, 1, replay_data.actions.to(torch::kLong));

		auto loss = torch::nn::functional::smooth_l1_loss(current_q_values, target_q_values);
		total_loss += loss.item<float>();

		auto priorities =
			torch::pow((current_q_values.detach() - replay_data.values).abs(), config_.per_alpha).sum(-1).clamp_min(1e-8);
		buffer_.update_priorities(priorities, replay_data.indicies);

		optimiser_.zero_grad();
		loss.backward();
		torch::nn::utils::clip_grad_norm_(model_->parameters(), config_.max_grad_norm);
		optimiser_.step();

		if (n_updates_ % config_.target_update_interval == 0)
		{
			model_->update(config_.tau);
		}
	}

	update_exploration(timestep);

	return {
		{TrainResultType::kLoss, total_loss / config_.gradient_steps},
		{TrainResultType::kLearningRate, static_cast<float>(lr_param_)},
		{TrainResultType::kExploration, static_cast<float>(exploration_param_)}};
}

void DQN::save(const std::filesystem::path& path) const
{
	torch::save(optimiser_, path / "optimiser.pt");
	model_->save(path);
}

void DQN::load(const std::filesystem::path& path)
{
	if (std::filesystem::exists(path / "optimiser.pt"))
	{
		torch::load(optimiser_, path / "optimiser.pt");
		spdlog::info("Optimiser loaded");
	}
	model_->load(path);
	model_->train();
}

void DQN::update_learning_rate(int timestep)
{
	double alpha = learning_rate_decay(&config_, timestep, config_.total_timesteps);
	lr_param_ = std::max(config_.learning_rate * alpha, config_.learning_rate_min);
	for (auto& group : optimiser_.param_groups())
	{
		if (group.has_options())
		{
			dynamic_cast<torch::optim::AdamOptions&>(group.options()).lr(lr_param_);
		}
	}
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
