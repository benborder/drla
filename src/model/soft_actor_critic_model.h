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
	torch::Tensor action;
	torch::Tensor actions_pi;
	torch::Tensor log_prob;
	std::vector<torch::Tensor> state;
};

class SoftActorCriticModel : public Model
{
public:
	SoftActorCriticModel(const Config::ModelConfig& config, const EnvironmentConfiguration& env_config, int value_shape);
	SoftActorCriticModel(const SoftActorCriticModel& other, const c10::optional<torch::Device>& device);

	PredictOutput predict(const ModelInput& input) override;

	StateShapes get_state_shape() const override;

	ActorOutput action_output(const Observations& observations, const HiddenStates& state);
	std::vector<torch::Tensor>
	critic(const Observations& observations, const torch::Tensor& actions, const HiddenStates& state);
	std::vector<torch::Tensor>
	critic_target(const Observations& observations, const torch::Tensor& actions, const HiddenStates& state);

	void update(double tau);

	void train(bool train = true) override;

	std::vector<torch::Tensor> parameters(bool recursive = true) const;
	std::vector<torch::Tensor> actor_parameters(bool recursive = true) const;
	std::vector<torch::Tensor> critic_parameters(bool recursive = true) const;

	void save(const std::filesystem::path& path) override;
	void load(const std::filesystem::path& path) override;

	std::shared_ptr<torch::nn::Module> clone(const c10::optional<torch::Device>& device = c10::nullopt) const override;

private:
	struct CriticModules
	{
		FeatureExtractor feature_extractor_ = nullptr;
		torch::nn::GRUCell grucell_ = nullptr;
		FCBlock critic_ = nullptr;
	};

	std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>> critic_values(
		std::vector<CriticModules>& critics,
		const Observations& observations,
		const torch::Tensor& actions,
		const torch::Tensor& features,
		const HiddenStates& state);

private:
	const Config::SoftActorCriticConfig config_;
	const int value_shape_;
	const ActionSpace action_space_;
	const bool use_gru_;

	FeatureExtractor feature_extractor_actor_;
	torch::nn::GRUCell grucell_;
	FCBlock actor_;
	PolicyActionOutput policy_action_output_;

	std::vector<CriticModules> critics_;
	std::vector<CriticModules> critic_targets_;
};
} // namespace drla
