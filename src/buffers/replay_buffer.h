#pragma once

#include "agent_types.h"
#include "configuration.h"
#include "episodic_per_buffer.h"
#include "threadpool.h"
#include "types.h"

#include <torch/torch.h>

#include <vector>

namespace drla
{

struct ReplayBufferSamples
{
	Observations observations;
	Observations next_observations;
	torch::Tensor rewards;
	torch::Tensor values;
	torch::Tensor actions;
	torch::Tensor episode_non_terminal;
	HiddenStates state;
	HiddenStates next_state;
	std::vector<std::pair<int, int>> indicies;
};

class ReplayBuffer final : public EpisodicPERBuffer
{
public:
	ReplayBuffer(
		int buffer_size,
		int n_envs,
		const EnvironmentConfiguration& env_config,
		int reward_shape,
		StateShapes state_shape,
		const std::vector<float>& gamma,
		float per_alpha,
		torch::Device device);

	void add(StepData step_data);

	Observations get_observations_head() const;
	Observations get_observations_head(int env) const;
	torch::Tensor get_actions_head() const;
	torch::Tensor get_actions_head(int env) const;
	std::vector<torch::Tensor> get_state_head() const;
	std::vector<torch::Tensor> get_state_head(int env) const;

	ReplayBufferSamples sample(int sample_size);

	torch::Device get_device() const;

protected:
	void add_episode(std::shared_ptr<Episode> episode) override;

private:
	torch::Device device_;
	const int n_envs_;

	ThreadPool episode_queue_;
	std::vector<std::vector<StepData>> current_episodes_;

	ObservationShapes observation_shape_;
	std::vector<int64_t> action_shape_;
	StateShapes state_shapes_;
};

} // namespace drla
