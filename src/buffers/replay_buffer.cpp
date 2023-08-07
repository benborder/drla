#include "replay_buffer.h"

#include "off_policy_episode.h"
#include "utils.h"

#include <spdlog/spdlog.h>

using namespace drla;

ReplayBuffer::ReplayBuffer(
	int buffer_size,
	int n_envs,
	const EnvironmentConfiguration& env_config,
	int reward_shape,
	StateShapes state_shape,
	const std::vector<float>& gamma,
	float per_alpha,
	torch::Device device)
		: EpisodicPERBuffer(gamma, {buffer_size, reward_shape, 1, per_alpha, env_config.action_space})
		, device_(device)
		, n_envs_(n_envs)
		, episode_queue_(1)
		, action_shape_({n_envs})
		, state_shapes_(state_shape)
{
	current_episodes_.resize(n_envs);
	for (size_t i = 0; i < env_config.observation_shapes.size(); i++)
	{
		std::vector<int64_t> observations_shape{n_envs};
		observations_shape.insert(
			observations_shape.end(), env_config.observation_shapes[i].begin(), env_config.observation_shapes[i].end());
		observation_shape_.push_back(std::move(observations_shape));
	}
	if (is_action_discrete(env_config.action_space))
	{
		for (size_t i = 0; i < env_config.action_space.shape.size(); i++) { action_shape_.push_back(1); }
	}
	else
	{
		action_shape_.insert(
			action_shape_.end(), env_config.action_space.shape.begin(), env_config.action_space.shape.end());
	}
	gamma_ = gamma_.to(device_);
}

void ReplayBuffer::add_episode(std::shared_ptr<Episode> episode)
{
	if (episode->length() == 0)
	{
		spdlog::error("Episode length must be non zero. The episode has not been added to the buffer.");
		return;
	}

	std::lock_guard lock(m_episodes_);

	episode->set_id(total_episodes_++);
	episode->init_priorities(gamma_, options_.per_alpha);

	total_steps_ += episode->length();
	if (get_num_samples() < options_.buffer_size)
	{
		episodes_.push_back(std::move(episode));
	}
	else
	{
		total_steps_ -= episodes_.back()->length();
		while (get_num_samples() >= options_.buffer_size)
		{
			episodes_.pop_back();
			total_steps_ -= episodes_.back()->length();
		}
		episodes_.back() = std::move(episode);
	}
	std::sort(episodes_.begin(), episodes_.end(), [](const auto& e1, const auto& e2) {
		return e1->get_priority() > e2->get_priority();
	});
}

void ReplayBuffer::add(StepData step_data)
{
	auto& episode = current_episodes_.at(step_data.env);
	bool episode_end = step_data.env_data.state.episode_end;
	episode.push_back(std::move(step_data));

	if (episode_end)
	{
		episode_queue_.queue_task([this, episode = std::move(episode)]() {
			add_episode(std::make_shared<OffPolicyEpisode>(
				std::move(episode), OffPolicyEpisodeOptions{static_cast<int>(flatten(options_.action_space.shape))}));
		});
	}
}

Observations ReplayBuffer::get_observations_head() const
{
	Observations obs_head;
	for (auto& shape : observation_shape_) { obs_head.push_back(torch::empty(shape, device_)); }
	for (int env = 0; env < n_envs_; ++env)
	{
		auto& episode = current_episodes_[env];
		auto& step_data = episode.back();
		for (size_t i = 0; i < obs_head.size(); ++i) { obs_head[i][env] = step_data.env_data.observation[i]; }
	}
	return obs_head;
}

Observations ReplayBuffer::get_observations_head(int env) const
{
	return current_episodes_.at(env).back().env_data.observation;
}

torch::Tensor ReplayBuffer::get_actions_head() const
{
	torch::Tensor actions_head = torch::empty(action_shape_, device_);
	for (int env = 0; env < n_envs_; ++env) { actions_head[env] = current_episodes_[env].back().predict_result.action; }
	return actions_head;
}

torch::Tensor ReplayBuffer::get_actions_head(int env) const
{
	return current_episodes_.at(env).back().predict_result.action;
}

std::vector<torch::Tensor> ReplayBuffer::get_state_head() const
{
	std::vector<torch::Tensor> states_head;
	for (auto& shape : state_shapes_) { states_head.push_back(torch::empty({n_envs_, shape}, device_)); }
	for (int env = 0; env < n_envs_; ++env)
	{
		auto& step_data = current_episodes_[env].back();
		for (size_t i = 0; i < states_head.size(); ++i) { states_head[i][env] = step_data.predict_result.state[i]; }
	}
	return states_head;
}

std::vector<torch::Tensor> ReplayBuffer::get_state_head(int env) const
{
	return current_episodes_.at(env).back().predict_result.state;
}

ReplayBufferSamples ReplayBuffer::sample(int sample_size)
{
	std::vector<int64_t> action_shape(action_shape_);
	action_shape[0] = sample_size;
	c10::ScalarType action_type = is_action_discrete(options_.action_space) ? torch::kLong : torch::kFloat;

	ReplayBufferSamples data;
	for (auto obs_shape : observation_shape_)
	{
		obs_shape[0] = sample_size;
		data.observations.push_back(torch::empty(obs_shape, device_));
		data.next_observations.push_back(torch::empty(obs_shape, device_));
	}
	for (auto& state : state_shapes_)
	{
		data.state.push_back(torch::empty({sample_size, state}, device_));
		data.next_state.push_back(torch::empty({sample_size, state}, device_));
	}
	data.actions = torch::empty(action_shape, torch::TensorOptions(device_).dtype(action_type));
	data.rewards = torch::empty({sample_size, options_.reward_shape}, device_);
	data.values = torch::empty({sample_size, options_.reward_shape}, device_);
	data.episode_non_terminal = torch::empty({sample_size, 1}, device_);

	int sample_index = 0;
	auto episodes = sample_episodes(sample_size);
	for (const auto& [episode, episode_prob] : episodes)
	{
		auto [index, probs] = episode->sample_position(gen_);
		auto target = episode->make_target(index, gamma_);
		auto next_target = episode->make_target(index + 1, gamma_);
		auto sample_obs = episode->get_observations(index, device_);
		auto sample_next_obs = episode->get_observations(index + 1, device_);

		data.indicies.emplace_back(episode->get_id(), index);

		for (size_t i = 0; i < data.observations.size(); ++i)
		{
			data.observations[i][sample_index] = sample_obs[i].detach().to(device_);
			data.next_observations[i][sample_index] = sample_next_obs[i].detach().to(device_);
		}
		data.actions[sample_index] = target.actions.detach().to(device_);
		data.rewards[sample_index] = target.rewards.detach().to(device_);
		data.values[sample_index] = target.values.detach().to(device_);
		for (size_t i = 0; i < data.state.size(); ++i)
		{
			data.state[sample_index] = target.states[i];
			data.next_state[sample_index] = next_target.states[i];
		}
		data.episode_non_terminal[sample_index] = target.non_terminal.detach().to(device_);
		++sample_index;
	}
	return data;
}

torch::Device ReplayBuffer::get_device() const
{
	return device_;
}
