#pragma once

#include "drla/configuration/model.h"
#include "drla/model/feature_extractor.h"
#include "drla/model/res_block.h"

#include <torch/torch.h>

#include <variant>
#include <vector>

namespace drla
{

class CNNExtractor : public FeatureExtractorGroup
{
public:
	CNNExtractor(const Config::CNNConfig& config, const std::vector<int64_t>& observation_shape);

	torch::Tensor forward(const torch::Tensor& observation) override;
	std::vector<int64_t> get_output_shape() const override;

private:
	using Conv2d = std::pair<torch::nn::Conv2d, std::function<torch::Tensor(const torch::Tensor&)>>;
	using Layer =
		std::variant<Conv2d, torch::nn::MaxPool2d, torch::nn::AvgPool2d, torch::nn::AdaptiveAvgPool2d, ResBlock2d>;
	std::vector<Layer> cnn_layers_;
	std::vector<int64_t> out_shape_;
};

} // namespace drla
