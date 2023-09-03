#pragma once

#include "algorithm.h"
#include "configuration/algorithm.h"
#include "mcts_replay_buffer.h"
#include "model.h"
#include "optimiser.h"

#include <torch/torch.h>

#include <string>
#include <vector>

namespace drla
{

class MuZero final : public Algorithm
{
public:
	MuZero(
		const Config::AgentTrainAlgorithm& config, std::shared_ptr<MCTSModelInterface> model, MCTSReplayBuffer& buffer);

	std::string name() const override;
	Metrics update(int timestep) override;

	void save(const std::filesystem::path& path) const override;
	void load(const std::filesystem::path& path) override;

private:
	const Config::MuZero::TrainConfig config_;
	std::shared_ptr<MCTSModelInterface> model_;
	MCTSReplayBuffer& buffer_;

	Optimiser optimiser_;
};

} // namespace drla
