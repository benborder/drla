#pragma once

#include "episodic_per_buffer.h"
#include "types.h"

#include <torch/torch.h>

#include <deque>
#include <mutex>
#include <random>
#include <vector>

namespace drla
{

/// @brief A batch sampled from a buffer
struct HybridBatch
{
	// episode id and episode step index
	std::vector<std::pair<int, int>> indicies;
	Observations observation;
	torch::Tensor reward;
	torch::Tensor values;
	torch::Tensor action;
	torch::Tensor non_terminal;
	torch::Tensor weight;
	torch::Tensor is_first;
	HiddenStates states;
};

class HybridEpisode;

/// @brief Replay Buffer for hybrid based agents
class HybridReplayBuffer final : public EpisodicPERBuffer
{
public:
	HybridReplayBuffer(std::vector<float> gamma, int n_envs, EpisodicPERBufferOptions options);

	/// @brief Flushes the buffer's cached step data, moving any cached data in new_episodes_ to inprogress_episodes_
	/// where possible
	void flush_cache();

	/// @brief Adds an in-progress episode
	/// @param episode The in-progress hybrid episode to add
	/// @param env The environment id
	void add_in_progress_episode(std::shared_ptr<HybridEpisode> episode, int env);

	/// @brief Adds a step to the buffer for its associated env. While the episode is incomplete the all steps reside in a
	/// temporary location. Episodes with length greater than the unroll/sequence size are also used for training.
	/// @param step_data The data to add from a policy and env step
	/// @param force_cache Forces caching the step data until the episode ends
	void add(StepData step_data, bool force_cache = false);

	/// @brief Randomly sample a series of episodes from the buffer using the distribution formed from the saved
	/// priorities.
	/// @param num_samples The number of episodes to sample from the buffer.
	/// @param force_uniform Forces the sample distribution to be uniform rather than prioritised
	/// @return The sampled episodes and their policy probability
	std::vector<std::pair<std::shared_ptr<Episode>, float>>
	sample_episodes(int num_samples, bool force_uniform = false) const override;

	/// @brief Samples from the buffer, using the priorities to form a distribution to sample from
	/// @param batch_size The number of step index samples to retrieve
	/// @param device The device the samples should be on
	HybridBatch sample(int batch_size, torch::Device device = torch::kCPU) const;

	/// @brief Returns the total number of samples including the current in progress samples (episode has not terminated)
	/// @return The total number of samples in the buffer
	int get_num_samples() const override;

	/// @brief Reanalyse a random epiosde in the buffer
	/// @param model The model to use when reanalysing
	void reanalyse(std::shared_ptr<HybridModelInterface> model);

private:
	std::vector<std::vector<StepData>> new_episodes_;
	std::vector<std::shared_ptr<HybridEpisode>> inprogress_episodes_;
};
} // namespace drla
