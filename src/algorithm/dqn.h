#pragma once

#include "algorithm.h"
#include "configuration/algorithm.h"
#include "model.h"
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
	DQN(const Config::AgentTrainAlgorithm& config,
			const ObservationShapes& observation_shape,
			ReplayBuffer& buffer,
			std::shared_ptr<Model> model,
			torch::Tensor gamma);

	std::vector<UpdateResult> update(int timestep) override;

	void save(const std::filesystem::path& path) const override;
	void load(const std::filesystem::path& path) override;

private:
	void update_learning_rate(int timestep);
	void update_exploration(int timestep);

private:
	const Config::DQN config_;
	ReplayBuffer& buffer_;
	std::shared_ptr<QNetModelInterface> model_;
	torch::optim::Adam optimiser_;
	torch::Tensor gamma_;
	double lr_param_;
	double exploration_param_;
	int n_updates_ = 0;
};

} // namespace drla
