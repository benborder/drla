#include "rollout_buffer.h"

#include "minibatch_buffer.h"

#include <c10/util/ArrayRef.h>
#include <torch/torch.h>

#include <memory>
#include <vector>

using namespace drla;

RolloutBuffer::RolloutBuffer(
	int buffer_size,
	int n_envs,
	const EnvironmentConfiguration& env_config,
	int reward_shape,
	torch::Tensor gamma,
	torch::Tensor gae_lambda,
	torch::Device device)
		: device_(device), gamma_(gamma), gae_lambda_(gae_lambda), buffer_size_(buffer_size)
{
	for (size_t i = 0; i < env_config.observation_shapes.size(); i++)
	{
		std::vector<int64_t> observations_shape{buffer_size_ + 1, n_envs};
		observations_shape.insert(
			observations_shape.end(), env_config.observation_shapes[i].begin(), env_config.observation_shapes[i].end());
		observations_.push_back(
			torch::zeros(observations_shape, torch::TensorOptions(torch::kCPU).dtype(env_config.observation_dtypes[i])));
	}
	rewards_ = torch::zeros({buffer_size_, n_envs, reward_shape}, torch::TensorOptions(device));
	values_ = torch::zeros({buffer_size_ + 1, n_envs, reward_shape}, torch::TensorOptions(device));
	returns_ = torch::zeros({buffer_size_ + 1, n_envs, reward_shape}, torch::TensorOptions(device));
	advantages_ = torch::zeros({buffer_size_ + 1, n_envs, reward_shape}, torch::TensorOptions(device));
	action_log_probs_ = torch::zeros({buffer_size_, n_envs, 1}, torch::TensorOptions(device));
	std::vector<int64_t> action_shape{buffer_size_, n_envs};
	c10::ScalarType action_type;
	if (is_action_discrete(env_config.action_space))
	{
		action_type = torch::kLong;
		for (size_t i = 0; i < env_config.action_space.shape.size(); i++) { action_shape.push_back(1); }
	}
	else
	{
		action_type = torch::kFloat;
		action_shape.insert(action_shape.end(), env_config.action_space.shape.begin(), env_config.action_space.shape.end());
	}

	actions_ = torch::zeros(action_shape, torch::TensorOptions(device).dtype(action_type));
	episode_non_terminal_ = torch::ones({buffer_size_ + 1, n_envs, reward_shape}, torch::TensorOptions(device));
	pos_.resize(n_envs, 0);
}

void RolloutBuffer::initialise(const StepData& step_data)
{
	std::fill(pos_.begin(), pos_.end(), 0);
	for (size_t i = 0; i < observations_.size(); i++)
	{
		observations_[i][0][step_data.env].copy_(step_data.env_data.observation[i]);
	}
}

void RolloutBuffer::reset()
{
	for (auto& obs : observations_) { obs.zero_(); }
	rewards_.zero_();
	values_.zero_();
	returns_.zero_();
	advantages_.zero_();
	action_log_probs_.zero_();
	actions_.zero_();
	episode_non_terminal_.fill_(1.0F);
	std::fill(pos_.begin(), pos_.end(), 0);
}

void RolloutBuffer::add(const StepData& step_data)
{
	int pos = pos_[step_data.env];
	for (size_t i = 0; i < observations_.size(); i++)
	{
		observations_[i][pos + 1][step_data.env].copy_(step_data.env_data.observation[i]);
	}
	actions_[pos][step_data.env].copy_(step_data.predict_result.action[0]);
	action_log_probs_[pos][step_data.env].copy_(step_data.predict_result.action_log_probs[0]);
	values_[pos][step_data.env].copy_(step_data.predict_result.values[0]);
	if (rewards_.size(2) == 1 && rewards_.size(2) != step_data.reward.size(0))
	{
		rewards_[pos][step_data.env].copy_(step_data.reward.sum());
	}
	else
	{
		rewards_[pos][step_data.env].copy_(step_data.reward);
	}
	pos_[step_data.env] = (pos + 1) % buffer_size_;

	// When the episode ends zero all masks
	episode_non_terminal_[pos + 1][step_data.env] = step_data.env_data.state.episode_end ? 0.0F : 1.0F;
}

void RolloutBuffer::add(const TimeStepData& timestep_data)
{
	int pos = pos_[0];
	for (size_t i = 0; i < observations_.size(); i++) { observations_[i][pos + 1].copy_(timestep_data.observations[i]); }
	actions_[pos].copy_(timestep_data.predict_results.action);
	action_log_probs_[pos].copy_(timestep_data.predict_results.action_log_probs);
	values_[pos].copy_(timestep_data.predict_results.values);
	if (rewards_.size(2) == 1 && rewards_.size(2) != timestep_data.rewards.size(1))
	{
		rewards_[pos].copy_(timestep_data.rewards.sum({1}));
	}
	else
	{
		rewards_[pos].copy_(timestep_data.rewards);
	}

	for (size_t i = 0; i < pos_.size(); i++)
	{
		pos_[i] = (pos + 1) % buffer_size_;
		// When the episode ends zero all masks
		episode_non_terminal_[pos + 1][i] = timestep_data.states[i].episode_end ? 0.0F : 1.0F;
	}
}

MiniBatchBuffer RolloutBuffer::get(int num_mini_batch)
{
	auto env_size = actions_.size(1);
	auto batch_size = env_size * buffer_size_;
	if (batch_size < num_mini_batch)
	{
		throw std::runtime_error(
			"The number of samples '" + std::to_string(env_size * buffer_size_) +
			"' must be >= to the number of minibatches '" + std::to_string(num_mini_batch) + "'");
	}
	auto mini_batch_size = batch_size / num_mini_batch;
	return MiniBatchBuffer(*this, mini_batch_size);
}

const Observations& RolloutBuffer::get_observations() const
{
	return observations_;
}

Observations RolloutBuffer::get_observations(int step) const
{
	Observations obs;
	for (const auto& observation_group : observations_) { obs.push_back(observation_group[step].to(device_)); }
	return obs;
}

Observations RolloutBuffer::get_observations(int step, int env) const
{
	Observations obs;
	for (const auto& observation_group : observations_) { obs.push_back(observation_group[step][env].to(device_)); }
	return obs;
}

torch::Tensor RolloutBuffer::get_rewards() const
{
	return rewards_;
}

torch::Tensor RolloutBuffer::get_values() const
{
	return values_;
}

torch::Tensor RolloutBuffer::get_returns() const
{
	return returns_;
}

torch::Tensor RolloutBuffer::get_advantages() const
{
	return advantages_;
}

torch::Tensor RolloutBuffer::get_action_log_probs() const
{
	return action_log_probs_;
}

torch::Tensor RolloutBuffer::get_actions() const
{
	return actions_;
}

void RolloutBuffer::compute_returns_and_advantage(const torch::Tensor& last_values)
{
	torch::NoGradGuard no_grad;
	values_[-1] = last_values;
	torch::Tensor last_gae_lam = torch::zeros({rewards_.size(1), 1}, torch::TensorOptions(device_));
	for (int pos = buffer_size_ - 1; pos >= 0; --pos)
	{
		auto delta = rewards_[pos] + gamma_ * values_[pos + 1] * episode_non_terminal_[pos + 1] - values_[pos];
		last_gae_lam = delta + gamma_ * gae_lambda_ * episode_non_terminal_[pos + 1] * last_gae_lam;
		advantages_[pos] = last_gae_lam;
	}
	returns_ = advantages_ + values_;
}

void RolloutBuffer::prepare_next_batch()
{
	for (auto& observation_group : observations_) { observation_group[0].copy_(observation_group[-1]); }
	episode_non_terminal_[0].copy_(episode_non_terminal_[-1]);
}

int RolloutBuffer::get_buffer_sample_size() const
{
	return buffer_size_ * static_cast<int>(pos_.size());
}

void RolloutBuffer::to(torch::Device device)
{
	device_ = device;
	rewards_ = rewards_.to(device);
	values_ = values_.to(device);
	returns_ = returns_.to(device);
	action_log_probs_ = action_log_probs_.to(device);
	actions_ = actions_.to(device);
	episode_non_terminal_ = episode_non_terminal_.to(device);
	gamma_ = gamma_.to(device);
	gae_lambda_ = gae_lambda_.to(device);
}

torch::Device RolloutBuffer::get_device() const
{
	return device_;
}
