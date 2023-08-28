#pragma once

#include "actor_net.h"
#include "configuration/model.h"
#include "fc_block.h"
#include "feature_extractor.h"
#include "model.h"
#include "types.h"

#include <torch/torch.h>

#include <memory>
#include <variant>
#include <vector>

namespace drla
{

class ActorCriticModel : public ActorCriticModelInterface
{
public:
	ActorCriticModel(
		const Config::ModelConfig& config,
		const EnvironmentConfiguration& env_config,
		int value_shape,
		bool predict_Values = false);
	ActorCriticModel(const ActorCriticModel& other, const c10::optional<torch::Device>& device);

	ModelOutput predict(const ModelInput& input) override;

	ModelOutput initial() override;

	StateShapes get_state_shape() const override;

	ActionPolicyEvaluation
	evaluate_actions(const Observations& observations, const torch::Tensor& actions, const HiddenStates& states) override;

	void save(const std::filesystem::path& path) override;
	void load(const std::filesystem::path& path) override;

	std::shared_ptr<torch::nn::Module> clone(const c10::optional<torch::Device>& device = c10::nullopt) const override;

private:
	const Config::ActorCriticConfig config_;
	const bool predict_values_;
	const bool use_gru_;

	const ActionSpace action_space_;

	FeatureExtractor feature_extractor_;
	FeatureExtractor feature_extractor_critic_;

	FCBlock shared_;
	FCBlock critic_;
	Actor actor_;
	torch::nn::GRUCell grucell_;
};
} // namespace drla
