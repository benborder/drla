#pragma once

#include "agent_types.h"
#include "configuration.h"
#include "types.h"

#include <ATen/core/Tensor.h>
#include <c10/util/ArrayRef.h>

#include <vector>

namespace drla
{

class MiniBatchBuffer;

class RolloutBuffer
{
public:
	RolloutBuffer(
		int buffer_size,
		int n_envs,
		const EnvironmentConfiguration& env_config,
		int reward_shape,
		StateShapes state_shape,
		torch::Tensor gamma,
		torch::Tensor gae_lambda,
		torch::Device device);

	void initialise(const StepData& step_data);
	void reset();

	void add(const StepData& step_data);
	void add(const TimeStepData& timestep_data);

	MiniBatchBuffer get(int num_mini_batch);

	const Observations& get_observations() const;
	Observations get_observations(int step) const;
	Observations get_observations(int step, int env) const;

	torch::Tensor get_rewards() const;
	torch::Tensor get_values() const;
	torch::Tensor get_returns() const;
	torch::Tensor get_advantages() const;
	torch::Tensor get_action_log_probs() const;
	torch::Tensor get_actions() const;
	HiddenStates get_states() const;
	HiddenStates get_states(int step) const;
	HiddenStates get_states(int step, int env) const;

	void compute_returns_and_advantage(const torch::Tensor& last_values);
	void prepare_next_batch();
	int get_buffer_sample_size() const;

	void to(torch::Device device);
	torch::Device get_device() const;

private:
	torch::Device device_;

	Observations observations_;
	torch::Tensor rewards_;
	torch::Tensor values_;
	torch::Tensor returns_;
	torch::Tensor advantages_;
	torch::Tensor action_log_probs_;
	torch::Tensor actions_;
	torch::Tensor episode_non_terminal_;
	HiddenStates states_;

	torch::Tensor gamma_;
	torch::Tensor gae_lambda_;

	const int buffer_size_;
	std::vector<int> pos_;
};

} // namespace drla
