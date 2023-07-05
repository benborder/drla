#pragma once

#include "configuration/model.h"
#include "fc_block.h"
#include "feature_extractor.h"
#include "model.h"
#include "policy_action_output.h"
#include "res_block.h"
#include "types.h"

#include <torch/torch.h>

#include <memory>
#include <variant>
#include <vector>

namespace drla
{

class DynamicsNetworkImpl final : public torch::nn::Module
{
public:
	DynamicsNetworkImpl(
		const Config::DynamicsNetworkConfig& config,
		const std::vector<std::vector<int64_t>>& input_shape,
		const ActionSpace& action_space,
		int reward_shape);
	DynamicsNetworkImpl(const DynamicsNetworkImpl& other, const c10::optional<torch::Device>& device);

	std::pair<std::vector<torch::Tensor>, torch::Tensor> forward(const std::vector<torch::Tensor>& next_state);

	std::shared_ptr<torch::nn::Module> clone(const c10::optional<torch::Device>& device = c10::nullopt) const override;

private:
	const Config::DynamicsNetworkConfig config_;

	struct Dynamics2D
	{
		torch::nn::Conv2d conv_;
		torch::nn::BatchNorm2d bn_;
		torch::nn::Conv2d conv1x1_reward_;
		std::vector<ResBlock2d> resblocks_;
	};

	using DynamicsEncodingNet = std::variant<Dynamics2D, FCBlock>;
	std::vector<DynamicsEncodingNet> dynamics_encoding_;

	FCBlock fc_reward_;
};

TORCH_MODULE(DynamicsNetwork);

class PredictionNetworkImpl final : public torch::nn::Module

{
public:
	PredictionNetworkImpl(
		const Config::PredictionNetworkConfig& config,
		const std::vector<std::vector<int64_t>>& input_shape,
		const ActionSpace& action_space,
		int value_shape);
	PredictionNetworkImpl(const PredictionNetworkImpl& other, const c10::optional<torch::Device>& device);

	std::pair<torch::Tensor, torch::Tensor> forward(const std::vector<torch::Tensor>& input);

	std::shared_ptr<torch::nn::Module> clone(const c10::optional<torch::Device>& device = c10::nullopt) const override;

private:
	const Config::PredictionNetworkConfig config_;

	struct Prediction2D
	{
		std::vector<ResBlock2d> resblocks_;
		torch::nn::Conv2d conv1x1_value_;
		torch::nn::Conv2d conv1x1_policy_;
	};

	using PredictionEncoding = std::variant<std::monostate, Prediction2D>;
	std::vector<PredictionEncoding> prediction_encoding_;

	FCBlock fc_value_;
	FCBlock fc_policy_;
};

TORCH_MODULE(PredictionNetwork);

class MuZeroModel final : public MCTSModelInterface
{
public:
	MuZeroModel(const Config::ModelConfig& config, const EnvironmentConfiguration& env_config, int reward_shape);
	MuZeroModel(const MuZeroModel& other, const c10::optional<torch::Device>& device);

	ModelOutput predict(const ModelInput& input) override;
	ModelOutput predict(const ModelOutput& previous_output, bool deterministic = true) override;

	StateShapes get_state_shape() const override;

	torch::Tensor support_to_scalar(torch::Tensor logits) override;
	torch::Tensor scalar_to_support(torch::Tensor x) override;

	int get_stacked_observation_size() const override;

	void save(const std::filesystem::path& path) override;
	void load(const std::filesystem::path& path) override;

	std::shared_ptr<torch::nn::Module> clone(const c10::optional<torch::Device>& device = c10::nullopt) const override;

private:
	const Config::MuZeroModelConfig config_;
	const int action_space_size_;
	const int reward_shape_;

	FeatureExtractor representation_network_;
	DynamicsNetwork dynamics_network_;
	PredictionNetwork prediction_network_;
};

} // namespace drla
