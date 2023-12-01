#include "hybrid_episode.h"

#include "functions.h"
#include "utils.h"

#include <algorithm>

using namespace drla;

HybridEpisode::HybridEpisode(std::vector<StepData> episode_data, HybridEpisodeOptions options)
		: options_(options)
		, episode_length_(static_cast<int>(episode_data.size() - 1))
		, sequence_length_(options_.unroll_steps)
{
	assert(episode_length_ > 0);
	torch::NoGradGuard no_grad;

	const auto& initial_step_data = episode_data.at(0);
	std::vector<int64_t> epsz{episode_length_ + 1};
	actions_ = torch::empty(epsz + initial_step_data.predict_result.action[0].sizes().vec());
	rewards_ = torch::empty(epsz + initial_step_data.reward.sizes().vec());
	values_ = torch::empty(std::vector<int64_t>{episode_length_} + initial_step_data.predict_result.values.sizes().vec());
	for (auto& obs : initial_step_data.env_data.observation)
	{
		observations_.push_back(torch::empty(epsz + obs.sizes().vec(), obs.scalar_type()));
	}
	for (auto& state : initial_step_data.predict_result.state)
	{
		states_.push_back(torch::empty(epsz + state.sizes().slice(1).vec()));
	}

	size_t obs_dims = initial_step_data.env_data.observation.size();
	size_t state_dims = initial_step_data.predict_result.state.size();
	for (auto& step_data : episode_data)
	{
		for (size_t i = 0; i < obs_dims; ++i) { observations_[i][step_data.step] = step_data.env_data.observation[i]; }
		actions_[step_data.step] = step_data.predict_result.action[0];
		rewards_[step_data.step] = step_data.reward;
		for (size_t i = 0; i < state_dims; ++i)
		{
			states_[i][step_data.step] = step_data.predict_result.state[i][0].to(torch::kCPU);
		}
		if (step_data.step > 0)
		{
			values_[step_data.step - 1] = step_data.predict_result.values[0];
		}
		if (step_data.env_data.state.episode_end)
		{
			is_terminal_ = true;
		}
	}
}

void HybridEpisode::allocate_reserve(torch::Tensor& x)
{
	constexpr int kAllocateAhead = 100;
	int len = (((episode_length_ + 1) / kAllocateAhead) + 1) * kAllocateAhead;
	if (len > x.size(0))
	{
		auto dims = x.sizes().vec();
		dims[0] = len;
		x.resize_(dims);
	}
}

void HybridEpisode::add_step(StepData&& data)
{
	std::lock_guard lock(m_updates_);

	for (auto& obs : observations_) { allocate_reserve(obs); }
	for (auto& state : states_) { allocate_reserve(state); }
	allocate_reserve(actions_);
	allocate_reserve(rewards_);
	allocate_reserve(values_);

	for (size_t i = 0; i < observations_.size(); ++i) { observations_[i][data.step] = data.env_data.observation[i]; }
	for (size_t i = 0; i < states_.size(); ++i)
	{
		states_[i][data.step] = data.predict_result.state[i][0].to(torch::kCPU);
	}
	actions_[data.step] = data.predict_result.action[0];
	rewards_[data.step] = data.reward;
	values_[data.step - 1] = data.predict_result.values[0];
	++episode_length_;
	if (data.env_data.state.episode_end)
	{
		is_terminal_ = true;
		for (auto& obs : observations_) { obs = obs.narrow(0, 0, episode_length_ + 1).clone(); }
		for (auto& state : states_) { state = state.narrow(0, 0, episode_length_ + 1).clone(); }
		actions_ = actions_.narrow(0, 0, episode_length_ + 1).clone();
		rewards_ = rewards_.narrow(0, 0, episode_length_ + 1).clone();
		values_ = values_.narrow(0, 0, episode_length_).clone();
	}
}

void HybridEpisode::set_id(int id)
{
	id_ = id;
}

int HybridEpisode::get_id() const
{
	return id_;
}

Observations HybridEpisode::get_observations(int step, torch::Device device) const
{
	int sequence_len = std::min(step + sequence_length_, episode_length_ + 1) - step;
	if (episode_length_ < sequence_length_)
	{
		sequence_len = episode_length_;
	}

	Observations stacked_obs;
	if (sequence_len < sequence_length_)
	{
		for (auto& obs : observations_)
		{
			auto real_obs = obs.narrow(0, step, sequence_len).to(device);
			auto zeros_shape = real_obs.sizes().vec();
			zeros_shape[0] = sequence_length_ - sequence_len;
			stacked_obs.push_back(torch::cat(
				{real_obs, torch::zeros(zeros_shape, torch::TensorOptions(real_obs.scalar_type()).device(device))}, 0));
		}
	}
	else
	{
		for (auto& obs : observations_) { stacked_obs.push_back(obs.narrow(0, step, sequence_length_).to(device)); }
	}
	return stacked_obs;
}

ObservationShapes HybridEpisode::get_observation_shapes() const
{
	ObservationShapes shapes;
	for (const auto& obs : observations_)
	{
		// assume shape is [step, channels, ...]
		auto dims = obs.sizes().vec();
		dims[0] = sequence_length_;
		shapes.push_back(dims);
	}
	return shapes;
}

StateShapes HybridEpisode::get_state_shapes() const
{
	StateShapes shapes;
	for (auto& state : states_)
	{
		// assume shape is [step, ..., ...]
		auto dims = state.sizes().vec();
		dims[0] = sequence_length_;
		shapes.push_back(dims);
	}
	return shapes;
}

Observations HybridEpisode::get_observation(int step, torch::Device device) const
{
	Observations observation;
	for (auto& obs : observations_) { observation.push_back(obs[step].unsqueeze(0).to(device)); }
	return observation;
}

torch::Tensor HybridEpisode::get_action(int step) const
{
	return actions_[step].unsqueeze(0);
}

torch::Tensor HybridEpisode::compute_target_value(int index, torch::Tensor gamma) const
{
	torch::Tensor value;
	auto bootstrap_index = index + sequence_length_;
	if (bootstrap_index < episode_length_)
	{
		auto prev_value = values_[bootstrap_index];
		value = prev_value * gamma.pow(sequence_length_);
	}
	else
	{
		value = torch::zeros_like(values_[0]);
	}

	for (int i = index + 1, n = std::min(bootstrap_index + 1, episode_length_); i < n; ++i)
	{
		value += rewards_[i] * gamma.pow(i - index - 1);
	}
	return value;
}

void HybridEpisode::init_priorities(torch::Tensor gamma, float per_alpha)
{
	gamma = gamma.to(torch::kCPU);
	std::lock_guard lock(m_updates_);
	priorities_.resize(episode_length_);
	for (int i = 0; i < episode_length_; ++i)
	{
		auto priority = (values_[i] - compute_target_value(i, gamma)).abs().pow(per_alpha).sum(-1);
		auto p = priority.item<float>();
		if (p > episode_priority_)
		{
			episode_priority_ = p;
		}
		priorities_[i] = p;
	}
}

void HybridEpisode::update_priorities(int index, torch::Tensor priorities)
{
	index = (index + episode_length_) % episode_length_;
	std::lock_guard lock(m_updates_);
	for (int i = index, end = std::min<int>(i + priorities.size(0), episode_length_); i < end; ++i)
	{
		priorities_[i] = priorities[i - index].sum(-1).item<float>();
	}
	episode_priority_ = *std::max_element(priorities_.begin(), priorities_.end());
}

float HybridEpisode::get_priority() const
{
	return episode_priority_;
}

std::pair<int, float> HybridEpisode::sample_position(std::mt19937& gen, bool force_uniform) const
{
	std::lock_guard lock(m_updates_);
	if (episode_length_ < sequence_length_)
	{
		return {0, priorities_[0]};
	}
	int step_index = 0;
	// Dont use the last step if the episode is in progress
	int ep_len = episode_length_ + (is_terminal_ ? 1 : 0);
	if (force_uniform)
	{
		std::uniform_int_distribution<int> step_dist(0, ep_len - sequence_length_);
		step_index = step_dist(gen);
	}
	else
	{
		std::discrete_distribution<int> step_dist(priorities_.begin(), priorities_.end());
		step_index = std::min(step_dist(gen) + sequence_length_, ep_len) - sequence_length_;
	}
	return {step_index, priorities_[step_index]};
}

EpisodeSampleTargets HybridEpisode::make_target(int index, [[maybe_unused]] torch::Tensor gamma) const
{
	EpisodeSampleTargets target;
	{
		auto dims = actions_.sizes().vec();
		dims[0] = sequence_length_;
		target.actions = torch::empty(dims);
	}
	{
		auto dims = rewards_.sizes().vec();
		dims[0] = sequence_length_;
		target.rewards = torch::empty(dims);
		target.values = torch::empty(dims);
	}
	for (auto& state : states_)
	{
		auto dims = state.sizes().vec();
		dims[0] = sequence_length_;
		target.states.push_back(torch::empty(dims));
	}
	target.non_terminal = torch::ones({sequence_length_});

	// TODO: potential optimisation: if j < episode_length_ then use a direct narrow for rewards/states/actions. Not sure
	// if will actually make a difference though

	std::lock_guard lock(m_updates_);

	for (int i = 0; i < sequence_length_; ++i)
	{
		int j = index + i;
		if (j < episode_length_)
		{
			target.actions[i] = actions_[j];
			target.rewards[i] = rewards_[j];
			target.values[i] = values_[j];
			for (size_t k = 0; k < states_.size(); ++k) { target.states[k][i] = states_[k][j]; }
		}
		else if (j == episode_length_)
		{
			target.actions[i] = actions_[j];
			target.rewards[i] = rewards_[j];
			target.values[i] = torch::zeros_like(values_[0]);
			for (size_t k = 0; k < states_.size(); ++k) { target.states[k][i] = states_[k][j]; }
			target.non_terminal[i] = 0;
		}
		else
		{
			target.actions[i] = torch::randint_like(actions_[0], options_.num_actions);
			target.rewards[i] = torch::zeros_like(rewards_[0]);
			target.values[i] = torch::zeros_like(values_[0]);
			for (size_t k = 0; k < states_.size(); ++k) { target.states[k][i] = torch::zeros_like(states_[k][0]); }
			target.non_terminal[i] = 0;
		}
	}

	return target;
}

void HybridEpisode::update_values(torch::Tensor values)
{
	std::lock_guard lock(m_updates_);
	if (values_.size(0) <= values.size(0))
	{
		values_ = values;
	}
	else
	{
		values_.narrow(0, 0, values.size(0)) = values;
	}
}

void HybridEpisode::update_states(HiddenStates& states)
{
	std::lock_guard lock(m_updates_);
	if (states_.front().size(0) == states.front().size(0))
	{
		states_ = states;
	}
	else
	{
		for (size_t i = 0; i < states_.size(); ++i) { states_[i].narrow(0, 0, states[i].size(0)) = states[i]; }
	}
}

int HybridEpisode::length() const
{
	return episode_length_;
}

void HybridEpisode::set_sequence_length(int length)
{
	sequence_length_ = length;
}
