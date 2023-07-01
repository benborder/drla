#pragma once

#include "agent_types.h"
#include "configuration.h"
#include "types.h"

#include <c10/util/ArrayRef.h>
#include <torch/torch.h>

#include <deque>

namespace drla
{

struct ReplayBufferSamples
{
	Observations observations;
	Observations next_observations;
	torch::Tensor rewards;
	torch::Tensor actions;
	torch::Tensor episode_non_terminal;
	HiddenStates state;
	HiddenStates next_state;
};

class ReplayBuffer
{
public:
	ReplayBuffer(
		int buffer_size,
		int n_envs,
		const EnvironmentConfiguration& env_config,
		int reward_shape,
		StateShapes state_shape,
		torch::Device device);

	void reset();

	void add(const StepData& step_data);
	void add(const TimeStepData& timestep_data);

	const Observations& get_observations() const;
	Observations get_observations_head() const;
	Observations get_observations_head(int env) const;
	torch::Tensor get_actions_head() const;
	torch::Tensor get_actions_head(int env) const;
	std::vector<torch::Tensor> get_state_head() const;
	std::vector<torch::Tensor> get_state_head(int env) const;

	ReplayBufferSamples sample(int sample_size);

	torch::Device get_device() const;

private:
	torch::Device device_;

	Observations observations_;
	torch::Tensor rewards_;
	torch::Tensor actions_;
	torch::Tensor episode_non_terminal_;
	std::vector<torch::Tensor> state_;

	const int buffer_size_;
	std::vector<int> pos_;
	bool full_ = false;
};

} // namespace drla
