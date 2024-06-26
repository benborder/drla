#include "off_policy_episode.h"

#include "functions.h"
#include "tensor_storage.h"
#include "utils.h"

#include <spdlog/spdlog.h>

#include <algorithm>

using namespace drla;

OffPolicyEpisode::OffPolicyEpisode(std::vector<StepData> episode_data, OffPolicyEpisodeOptions options)
		: options_(options), episode_length_(static_cast<int>(episode_data.size() - 1))
{
	assert(episode_length_ > 0);

	// Skip the first step as it doesn't contain policy and value data
	const auto& initial_step_data = episode_data.at(1);
	for (auto& obs : initial_step_data.env_data.observation)
	{
		auto dims = obs.sizes().vec();
		dims.insert(dims.begin(), episode_length_ + 1);
		observations_.push_back(torch::empty(dims, obs.scalar_type()));
	}
	for (auto& state : initial_step_data.predict_result.state)
	{
		auto dims = state.sizes().vec();
		dims.insert(dims.begin(), episode_length_ + 1);
		state_.push_back(torch::empty(dims));
	}
	{
		auto dims = initial_step_data.predict_result.action.sizes().vec();
		dims.insert(dims.begin(), episode_length_);
		actions_ = torch::empty(dims);
	}
	{
		auto dims = initial_step_data.reward.sizes().vec();
		dims.insert(dims.begin(), episode_length_);
		rewards_ = torch::empty(dims);
	}
	{
		auto dims = initial_step_data.predict_result.values.sizes().vec();
		dims.insert(dims.begin(), episode_length_);
		values_ = torch::empty(dims);
	}

	size_t obs_dims = initial_step_data.env_data.observation.size();
	size_t state_dims = initial_step_data.predict_result.state.size();
	int step = 0;
	for (auto& step_data : episode_data)
	{
		// Inputs to models
		for (size_t i = 0; i < obs_dims; ++i) { observations_[i][step] = step_data.env_data.observation[i]; }
		for (size_t i = 0; i < state_dims; ++i) { state_[i][step] = step_data.predict_result.state[i]; }
		// Outputs from models/env
		if (step > 0)
		{
			const int s = step - 1;
			actions_[s] = step_data.predict_result.action;
			rewards_[s] = step_data.reward;
			values_[s] = step_data.predict_result.values;
		}
		++step;
	}
}

OffPolicyEpisode::OffPolicyEpisode(
	const std::filesystem::path& path, const StateShapes& state_shapes, OffPolicyEpisodeOptions options)
		: options_(options), episode_length_(0), saved_path_(path)
{
	actions_ = load_tensor(path / "actions.bin");
	rewards_ = load_tensor(path / "rewards.bin");
	observations_ = load_tensor_vector(path, "observations");
	episode_length_ = static_cast<int>(actions_.size(0) - 1);
	values_ = torch::zeros(std::vector<int64_t>{episode_length_, rewards_.size(1)});
	for (auto& shape : state_shapes) { state_.push_back(torch::zeros(std::vector<int64_t>{actions_.size(0)} + shape)); }
}

void OffPolicyEpisode::set_id(int id)
{
	id_ = id;
}

int OffPolicyEpisode::get_id() const
{
	return id_;
}

Observations OffPolicyEpisode::get_observations(int step, torch::Device device) const
{
	Observations stacked_obs;
	for (auto& obs : observations_) { stacked_obs.push_back(obs.narrow(0, step, 1).squeeze(0).to(device)); }
	return stacked_obs;
}

ObservationShapes OffPolicyEpisode::get_observation_shapes() const
{
	ObservationShapes shapes;
	for (const auto& obs : observations_)
	{
		// assume shape is [step, channels, ...]
		auto dims = obs.sizes().vec();
		dims.erase(dims.begin());
		shapes.push_back(dims);
	}
	return shapes;
}

StateShapes OffPolicyEpisode::get_state_shapes() const
{
	StateShapes shapes;
	for (auto& state : state_) { shapes.push_back(state.sizes().slice(1).vec()); }
	return shapes;
}

void OffPolicyEpisode::init_priorities(torch::Tensor gamma, float per_alpha)
{
	gamma = gamma.to(torch::kCPU);
	priorities_.resize(episode_length_);
	for (int i = 0; i < episode_length_; ++i)
	{
		int next_index = std::max(i - 1, 0);
		auto next_value = gamma * values_[i] + rewards_[next_index];
		auto priority = (values_[i] - next_value).abs().pow(per_alpha).sum(-1);
		auto p = priority.item<float>();
		if (p > episode_priority_)
		{
			episode_priority_ = p;
		}
		priorities_[i] = p;
	}
}

void OffPolicyEpisode::update_priorities(int index, torch::Tensor priorities)
{
	std::lock_guard lock(m_updates_);
	auto p = priorities.item<float>();
	priorities_[index] = p;
	episode_priority_ = *std::max_element(priorities_.begin(), priorities_.end());
}

float OffPolicyEpisode::get_priority() const
{
	return episode_priority_;
}

std::pair<int, float> OffPolicyEpisode::sample_position(std::mt19937& gen, bool force_uniform) const
{
	std::lock_guard lock(m_updates_);
	int step_index = 0;
	if (force_uniform)
	{
		std::uniform_int_distribution<int> step_dist(0, priorities_.size() - 1);
		step_index = step_dist(gen);
	}
	else
	{
		std::discrete_distribution<int> step_dist(priorities_.begin(), priorities_.end());
		step_index = step_dist(gen);
	}
	return {step_index, priorities_[step_index]};
}

EpisodeSampleTargets OffPolicyEpisode::make_target(int index, [[maybe_unused]] torch::Tensor gamma) const
{
	EpisodeSampleTargets target;
	if (index < episode_length_)
	{
		target.actions = actions_[index];
		target.rewards = rewards_[index];
		target.values = values_[index];
		for (auto& state : state_) { target.states.push_back(state[index]); }
		if ((index + 1) < episode_length_)
		{
			target.non_terminal = torch::ones(1);
		}
		else
		{
			target.non_terminal = torch::zeros(1);
		}
	}
	else if (index == episode_length_)
	{
		target.actions = torch::randint_like(actions_[0], options_.num_actions);
		target.rewards = torch::zeros_like(rewards_[0]);
		target.values = torch::zeros_like(values_[0]);
		for (auto& state : state_) { target.states.push_back(state[index]); }
		target.non_terminal = torch::zeros(1);
	}
	else
	{
		target.actions = torch::randint_like(actions_[0], options_.num_actions);
		target.rewards = torch::zeros_like(rewards_[0]);
		target.values = torch::zeros_like(values_[0]);
		for (auto& state : state_) { target.states.push_back(torch::zeros_like(state[0])); }
		target.non_terminal = torch::zeros(1);
	}
	return target;
}

void OffPolicyEpisode::update_values(const torch::Tensor& values)
{
	std::lock_guard lock(m_updates_);
	reanalysed_values_ = values;
}

void OffPolicyEpisode::update_states(const HiddenStates& states)
{
	std::lock_guard lock(m_updates_);
	reanalysed_state_ = states;
}

int OffPolicyEpisode::length() const
{
	return episode_length_;
}

void OffPolicyEpisode::set_sequence_length([[maybe_unused]] int length)
{
}

void OffPolicyEpisode::save(const std::filesystem::path& path)
{
	if (!std::filesystem::exists(path))
	{
		spdlog::error("Unable to save episode data: The path '{}' does not exist.", path.string());
		return;
	}
	auto ep_path = path / ("episode_" + options_.name);
	std::filesystem::create_directory(ep_path);

	save_tensor(actions_, ep_path / "actions.bin");
	save_tensor(rewards_, ep_path / "rewards.bin");

	for (size_t i = 0; i < observations_.size(); ++i)
	{
		save_tensor(observations_[i], ep_path / (std::string{"observations"} + std::to_string(i) + ".bin"));
	}
	saved_path_ = ep_path;
}

const std::filesystem::path& OffPolicyEpisode::get_path() const
{
	return saved_path_;
}
