#pragma once

#include "drla/configuration/model.h"
#include "drla/model/feature_extractor.h"

#include <torch/torch.h>

#include <variant>
#include <vector>

namespace drla
{

class CnnExtractor : public FeatureExtractor
{
public:
	CnnExtractor(const Config::FeatureExtractorConfig& config, const ObservationShapes& observation_shape);

	torch::Tensor forward(const Observations& observations) override;
	int get_output_size() const override;

private:
	const Config::CNNConfig config_;

	using CNN = std::variant<torch::nn::Conv2d, torch::nn::Conv3d>;
	std::vector<std::vector<CNN>> conv_layers_;
	int output_size_;
};

} // namespace drla
