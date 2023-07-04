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
	std::vector<UpdateResult> update(int batch) override;

	void save(const std::filesystem::path& path) const override;
	void load(const std::filesystem::path& path) override;

private:
	void update_learning_rate(int batch);

private:
	const Config::SAC config_;
	const ActionSpace action_space_;
	ReplayBuffer& buffer_;
	std::shared_ptr<SoftActorCriticModel> model_;
	torch::Tensor log_ent_coef_;
	torch::optim::Adam actor_optimiser_;
	torch::optim::Adam critic_optimiser_;
	torch::optim::Adam ent_coef_optimiser_;
	double target_entropy_;
	double lr_param_;
};

} // namespace drla
