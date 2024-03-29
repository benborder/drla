#pragma once

#include "algorithm.h"
#include "configuration/algorithm.h"
#include "hybrid_replay_buffer.h"
#include "model.h"
#include "optimiser.h"
#include "utils.h"

#include <ATen/core/Tensor.h>

#include <string>
#include <vector>

namespace drla
{

class Dreamer : public Algorithm
{
public:
	Dreamer(
		const Config::AgentTrainAlgorithm& config,
		const ActionSpace& action_space,
		std::shared_ptr<HybridModelInterface> model,
		HybridReplayBuffer& buffer);

	std::string name() const override;
	Metrics update(int timestep) override;

	void save(const std::filesystem::path& path) const override;
	void load(const std::filesystem::path& path) override;

protected:
	void world_model_loss(const WorldModelOutput& wm_output, const HybridBatch& batch, Metrics& metrics);
	void behavioural_model_loss(const ImaginedTrajectory& imagined_trajectory, Metrics& metrics);
	torch::Tensor
	calculate_returns(const torch::Tensor& reward, const torch::Tensor& value, const torch::Tensor& non_terminal) const;

protected:
	const Config::Dreamer::TrainConfig config_;
	const ActionSpace action_space_;
	std::shared_ptr<HybridModelInterface> model_;
	HybridReplayBuffer& buffer_;

	Optimiser world_optimiser_;
	Optimiser actor_optimiser_;
	Optimiser critic_optimiser_;
	std::shared_ptr<Moments> moments_;
};

} // namespace drla
