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

struct ModelOutput
{
	// Value estimation
	torch::Tensor values;
	// Action distribution for action policy based models
	std::unique_ptr<Distribution> dist;
};

class QNetModel : public QNetModelInterface
{
public:
	QNetModel(const Config::ModelConfig& config, const EnvironmentConfiguration& env_config);
	QNetModel(const QNetModel& other, const c10::optional<torch::Device>& device);

	torch::Tensor forward(const Observations& observations, const HiddenStates& state);
	PredictOutput predict(const ModelInput& input) override;

	StateShapes get_state_shape() const override;

	torch::Tensor forward_target(const Observations& observations, const HiddenStates& state);
	void update(double tau);

	std::vector<torch::Tensor> parameters(bool recursive = true) const;

	void set_exploration(double exploration);

	void save(const std::filesystem::path& path) override;
	void load(const std::filesystem::path& path) override;

	std::shared_ptr<torch::nn::Module> clone(const c10::optional<torch::Device>& device = c10::nullopt) const override;

private:
	const Config::QNetModelConfig config_;
	const ActionSpace action_space_;
	const bool use_gru_;

	FeatureExtractor feature_extractor_;
	FeatureExtractor feature_extractor_target_;
	torch::nn::GRUCell grucell_;
	torch::nn::GRUCell grucell_target_;
	FCBlock q_net_;
	FCBlock q_net_target_;
	double exploration_ = 0;
};
} // namespace drla
