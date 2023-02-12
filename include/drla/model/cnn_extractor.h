#pragma once

#include "drla/configuration/model.h"
#include "drla/model/feature_extractor.h"
#include "drla/model/res_block.h"

#include <torch/torch.h>

#include <functional>
#include <variant>
#include <vector>

namespace drla
{

class CNNExtractor : public FeatureExtractorGroup
{
public:
	CNNExtractor(const Config::CNNConfig& config, const std::vector<int64_t>& observation_shape);
	CNNExtractor(const CNNExtractor& other, const c10::optional<torch::Device>& device);

	torch::Tensor forward(const torch::Tensor& observation) override;
	std::vector<int64_t> get_output_shape() const override;
	std::shared_ptr<torch::nn::Module> clone(const c10::optional<torch::Device>& device = c10::nullopt) const override;

private:
	using Layer = std::variant<
		torch::nn::Conv2d,
		torch::nn::BatchNorm2d,
		torch::nn::MaxPool2d,
		torch::nn::AvgPool2d,
		torch::nn::AdaptiveAvgPool2d,
		ResBlock2d,
		std::function<torch::Tensor(const torch::Tensor&)>>;
	std::vector<Layer> cnn_layers_;
	std::vector<int64_t> out_shape_;
};

} // namespace drla
