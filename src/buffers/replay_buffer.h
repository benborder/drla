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
};

class ReplayBuffer
{
public:
	ReplayBuffer(
		int buffer_size, int n_envs, const EnvironmentConfiguration& env_config, int reward_shape, torch::Device device);

	void reset();

	void add(const StepData& step_data);
	void add(const TimeStepData& timestep_data);

	const Observations& get_observations() const;
	Observations get_observations_head() const;
	Observations get_observations_head(int env) const;

	ReplayBufferSamples sample(int sample_size);

private:
	torch::Device device_;

	Observations observations_;
	torch::Tensor rewards_;
	torch::Tensor actions_;
	torch::Tensor episode_non_terminal_;

	const int buffer_size_;
	std::vector<int> pos_;
	bool full_ = false;
};

} // namespace drla
