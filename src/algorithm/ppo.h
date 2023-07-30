#pragma once

#include "algorithm.h"
#include "configuration/algorithm.h"
#include "model.h"
#include "rollout_buffer.h"

#include <torch/torch.h>

#include <string>
#include <vector>

namespace drla
{

class ActorCriticModel;

class PPO final : public Algorithm
{
public:
	PPO(const Config::AgentTrainAlgorithm& config, RolloutBuffer& buffer, std::shared_ptr<Model> model);

	std::string name() const override;
	std::vector<UpdateResult> update(int timestep) override;

	void save(const std::filesystem::path& path) const override;
	void load(const std::filesystem::path& path) override;

private:
	void update_learning_rate(int timestep);

private:
	const Config::PPO config_;
	RolloutBuffer& buffer_;
	std::shared_ptr<ActorCriticModelInterface> model_;
	torch::optim::Adam optimiser_;
	double lr_;
	double clip_policy_;
	double clip_vf_;
};

} // namespace drla
