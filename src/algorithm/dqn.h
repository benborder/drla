#pragma once

#include "algorithm.h"
#include "configuration/algorithm.h"
#include "model.h"
#include "optimiser.h"
#include "replay_buffer.h"

#include <torch/torch.h>

#include <string>
#include <vector>

namespace drla
{

class QNetModel;

class DQN final : public Algorithm
{
public:
	DQN(const Config::AgentTrainAlgorithm& config, ReplayBuffer& buffer, std::shared_ptr<Model> model);

	std::string name() const override;
	Metrics update(int timestep) override;

	void save(const std::filesystem::path& path) const override;
	void load(const std::filesystem::path& path) override;

private:
	void update_exploration(int timestep);

private:
	const Config::DQN config_;
	ReplayBuffer& buffer_;
	std::shared_ptr<QNetModelInterface> model_;
	Optimiser optimiser_;
	double exploration_param_;
	int n_updates_ = 0;
};

} // namespace drla
