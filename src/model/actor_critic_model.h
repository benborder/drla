#pragma once

#include "configuration/model.h"
#include "fc_block.h"
#include "feature_extractor.h"
#include "model.h"
#include "policy_action_output.h"
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

	PredictOutput predict(const Observations& observations, bool deterministic = true) override;

	ActionPolicyEvaluation evaluate_actions(const Observations& observations, const torch::Tensor& actions);

	void save(const std::filesystem::path& path) override;
	void load(const std::filesystem::path& path) override;

private:
	const Config::ActorCriticConfig config_;
	bool predict_values_;

	const ActionSpace action_space_;

	std::shared_ptr<FeatureExtractor> feature_extractor_;
	std::shared_ptr<FeatureExtractor> feature_extractor_critic_;

	FCBlock shared_;
	FCBlock critic_;
	FCBlock actor_;
	PolicyActionOutput policy_action_output_;
};
} // namespace drla
