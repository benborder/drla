#pragma once

#include "drla/configuration/model.h"
#include "drla/model/fc_block.h"
#include "drla/model/feature_extractor.h"

#include <torch/torch.h>

#include <vector>

namespace drla
{

class MlpExtractor : public FeatureExtractor
{
public:
	MlpExtractor(const Config::FeatureExtractorConfig& config, const ObservationShapes& observation_shape);

	torch::Tensor forward(const Observations& observations) override;
	int get_output_size() const override;

private:
	FCBlock hidden_;
	int output_size_;
};
} // namespace drla
