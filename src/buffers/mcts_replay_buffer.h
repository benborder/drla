#pragma once

#include "episodic_per_buffer.h"

#include <torch/torch.h>

#include <deque>
#include <mutex>
#include <random>
#include <vector>

namespace drla
{

/// @brief A batch sampled from a MCTS buffer
struct MCTSBatch
{
	// episode id and episode step index
	std::vector<std::pair<int, int>> indicies;
	Observations observation;
	torch::Tensor reward;
	torch::Tensor values;
	torch::Tensor policy;
	torch::Tensor action;
	torch::Tensor non_terminal;
	torch::Tensor weight;
	torch::Tensor gradient_scale;
};

/// @brief Replay Buffer for MCTS based agents
class MCTSReplayBuffer final : public EpisodicPERBuffer
{
public:
	MCTSReplayBuffer(std::vector<float> gamma, EpisodicPERBufferOptions options);

	/// @brief Samples from the buffer, using the priorities to form a distribution to sample from
	/// @param batch_size The number of step index samples to retrieve
	/// @param device The device the samples should be on
	MCTSBatch sample(int batch_size, torch::Device device = torch::kCPU) const;

	/// @brief Reanalyse a random epiosde in the buffer
	/// @param model The model to use when reanalysing
	void reanalyse(std::shared_ptr<MCTSModelInterface> model);
};
} // namespace drla
