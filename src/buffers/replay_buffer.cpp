#include "replay_buffer.h"

using namespace drla;

ReplayBuffer::ReplayBuffer(
		int buffer_size, int n_envs, const EnvironmentConfiguration& env_config, int reward_shape, torch::Device device)
		: device_(device), buffer_size_(buffer_size)
{
	for (size_t i = 0; i < env_config.observation_shapes.size(); i++)
	{
		std::vector<int64_t> observations_shape{buffer_size_, n_envs};
		observations_shape.insert(
				observations_shape.end(), env_config.observation_shapes[i].begin(), env_config.observation_shapes[i].end());
		observations_.push_back(
				torch::zeros(observations_shape, torch::TensorOptions(torch::kCPU).dtype(env_config.observation_dtypes[i])));
	}
	rewards_ = torch::zeros({buffer_size_, n_envs, reward_shape}, torch::TensorOptions(device));
	std::vector<int64_t> action_shape{buffer_size_, n_envs};
	c10::ScalarType action_type;
	if (is_action_discrete(env_config.action_space))
	{
		action_type = torch::kLong;
		for (size_t i = 0; i < env_config.action_space.shape.size(); i++)
		{
			action_shape.push_back(1);
		}
	}
	else
	{
		action_type = torch::kFloat;
		action_shape.insert(action_shape.end(), env_config.action_space.shape.begin(), env_config.action_space.shape.end());
	}
	actions_ = torch::zeros(action_shape, torch::TensorOptions(device).dtype(action_type));
	episode_non_terminal_ = torch::ones({buffer_size_, n_envs, reward_shape}, torch::TensorOptions(device));
	pos_.resize(n_envs, 0);
}

void ReplayBuffer::reset()
{
	for (auto& obs : observations_)
	{
		obs.zero_();
	}
	rewards_.zero_();
	actions_.zero_();
	episode_non_terminal_.fill_(1.0F);
	std::fill(pos_.begin(), pos_.end(), 0);
	full_ = false;
}

void ReplayBuffer::add(const StepData& step_data)
{
	int pos = pos_[step_data.env];
	int next_pos = (pos + 1) % buffer_size_;
	full_ |= next_pos < pos;
	for (size_t i = 0; i < observations_.size(); i++)
	{
		observations_[i][next_pos][step_data.env].copy_(step_data.step_result.observation[i]);
	}
	actions_[pos][step_data.env].copy_(step_data.predict_result.action[0]);
	if (rewards_.size(2) == 1 && rewards_.size(2) != step_data.reward.size(0))
	{
		rewards_[pos][step_data.env].copy_(step_data.reward.sum());
	}
	else
	{
		rewards_[pos][step_data.env].copy_(step_data.reward);
	}

	// When the episode ends zero all masks
	episode_non_terminal_[pos][step_data.env] = step_data.step_result.state.episode_end ? 0.0F : 1.0F;

	pos_[step_data.env] = next_pos;
}

void ReplayBuffer::add(const TimeStepData& timestep_data)
{
	int pos = pos_[0];
	int next_pos = (pos + 1) % buffer_size_;
	full_ |= next_pos < pos;
	for (size_t i = 0; i < observations_.size(); i++)
	{
		observations_[i][next_pos].copy_(timestep_data.observations[i]);
	}
	actions_[pos].copy_(timestep_data.predict_results.action);
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
		// When the episode ends zero all masks
		episode_non_terminal_[pos][i] = timestep_data.states[i].episode_end ? 0.0F : 1.0F;
		pos_[i] = next_pos;
	}
}

const Observations& ReplayBuffer::get_observations() const
{
	return observations_;
}

Observations ReplayBuffer::get_observations_head() const
{
	Observations obs;
	for (const auto& observation_group : observations_)
	{
		auto obs_shape = observation_group.sizes().vec();
		obs_shape.erase(obs_shape.begin());
		torch::Tensor obs_grp = torch::empty(obs_shape, device_);
		for (size_t i = 0; i < pos_.size(); i++)
		{
			obs_grp[i] = observation_group[pos_[i]][i].to(device_);
		}
		obs.push_back(obs_grp);
	}
	return obs;
}

Observations ReplayBuffer::get_observations_head(int env) const
{
	Observations obs;
	for (const auto& observation_group : observations_)
	{
		obs.push_back(observation_group[pos_[env]][env].unsqueeze(0).to(device_));
	}
	return obs;
}

ReplayBufferSamples ReplayBuffer::sample(int sample_size)
{
	torch::Tensor indices;
	int total_samples = buffer_size_ * static_cast<int>(pos_.size());
	// The most recent sample is not valid (there is no next observation)
	if (full_)
	{
		indices = (torch::randperm(total_samples, torch::TensorOptions(torch::kLong)).narrow(0, 0, sample_size) + pos_[0]) %
							buffer_size_;
	}
	else
	{
		indices = torch::randperm((pos_[0] - 2) * static_cast<int>(pos_.size()), torch::TensorOptions(torch::kLong))
									.narrow(0, 0, sample_size);
	}
	auto next_indicies = (indices + 1) % total_samples;

	ReplayBufferSamples data;
	for (size_t i = 0; i < observations_.size(); i++)
	{
		auto observations_shape = observations_[i].sizes().vec();
		observations_shape.erase(observations_shape.begin());
		observations_shape[0] = -1;
		auto obs = observations_[i].view(observations_shape);
		data.observations.push_back(obs.index({indices}).to(device_));
		data.next_observations.push_back(obs.index({next_indicies}).to(device_));
	}

	auto action_shape = actions_.sizes().vec();
	action_shape.erase(action_shape.begin());
	action_shape[0] = -1;
	data.actions = actions_.view(action_shape).index({indices}).to(device_);
	data.rewards = rewards_.view({-1, rewards_.size(2)}).index({indices}).to(device_);
	data.episode_non_terminal = episode_non_terminal_.view({-1, 1}).index({indices}).to(device_);
	return data;
}
