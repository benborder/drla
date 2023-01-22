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

struct ActorOutput
{
	torch::Tensor actions;
	torch::Tensor actions_pi;
	torch::Tensor log_prob;
};

class SoftActorCriticModel : public Model
{
public:
	SoftActorCriticModel(const Config::ModelConfig& config, const EnvironmentConfiguration& env_config, int value_shape);
	SoftActorCriticModel(const SoftActorCriticModel& other, const c10::optional<torch::Device>& device);

	PredictOutput predict(const Observations& observations, bool deterministic = false) override;

	ActorOutput action_output(const Observations& observations);
	std::vector<torch::Tensor> critic(const Observations& observations, const torch::Tensor& actions);
	std::vector<torch::Tensor> critic_target(const Observations& observations, const torch::Tensor& actions);

	void update(double tau);

	void train(bool train = true) override;

	std::vector<torch::Tensor> parameters(bool recursive = true) const;
	std::vector<torch::Tensor> actor_parameters(bool recursive = true) const;
	std::vector<torch::Tensor> critic_parameters(bool recursive = true) const;

	void save(const std::filesystem::path& path) override;
	void load(const std::filesystem::path& path) override;

	std::shared_ptr<torch::nn::Module> clone(const c10::optional<torch::Device>& device = c10::nullopt) const override;

private:
	const Config::SoftActorCriticConfig config_;
	const int value_shape_;
	const ActionSpace action_space_;

	FeatureExtractor feature_extractor_actor_;
	FCBlock actor_;
	PolicyActionOutput policy_action_output_;

	struct CriticModules
	{
		FeatureExtractor feature_extractor_;
		FCBlock critic_;
	};

	std::vector<CriticModules> critics_;
	std::vector<CriticModules> critic_targets_;
};
} // namespace drla
