#pragma once

#include "drla/configuration/model.h"
#include "drla/model/fc_block.h"
#include "drla/model/feature_extractor.h"

#include <torch/torch.h>

#include <vector>

namespace drla
{

class MLPExtractor : public FeatureExtractorGroup
{
public:
	MLPExtractor(const Config::MLPConfig& config, const std::vector<int64_t>& observation_shape);
	MLPExtractor(const MLPExtractor& other, const c10::optional<torch::Device>& device);

	torch::Tensor forward(const torch::Tensor& observation) override;
	std::vector<int64_t> get_output_shape() const override;
	std::shared_ptr<torch::nn::Module> clone(const c10::optional<torch::Device>& device = c10::nullopt) const override;

private:
	FCBlock hidden_;
	int output_size_;
};
} // namespace drla
