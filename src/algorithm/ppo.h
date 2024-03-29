#pragma once

#include "algorithm.h"
#include "configuration/algorithm.h"
#include "model.h"
#include "optimiser.h"
#include "rollout_buffer.h"

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
	Metrics update(int timestep) override;

	void save(const std::filesystem::path& path) const override;
	void load(const std::filesystem::path& path) override;

private:
	const Config::PPO config_;
	RolloutBuffer& buffer_;
	std::shared_ptr<ActorCriticModelInterface> model_;
	Optimiser optimiser_;
};

} // namespace drla
