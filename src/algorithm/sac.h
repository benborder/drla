#pragma once

#include "algorithm.h"
#include "configuration/algorithm.h"
#include "model.h"
#include "optimiser.h"
#include "replay_buffer.h"

#include <string>
#include <vector>

namespace drla
{

class SoftActorCriticModel;

class SAC final : public Algorithm
{
public:
	SAC(
		const Config::AgentTrainAlgorithm& config,
		const ActionSpace& action_space,
		ReplayBuffer& buffer,
		std::shared_ptr<Model> model);

	std::string name() const override;
	Metrics update(int timestep) override;

	void save(const std::filesystem::path& path) const override;
	void load(const std::filesystem::path& path) override;

private:
	const Config::SAC config_;
	const ActionSpace action_space_;
	ReplayBuffer& buffer_;
	std::shared_ptr<SoftActorCriticModel> model_;
	torch::Tensor log_ent_coef_;
	Optimiser actor_optimiser_;
	Optimiser critic_optimiser_;
	Optimiser ent_coef_optimiser_;
	double target_entropy_;
};

} // namespace drla
