#pragma once

#include "drla/configuration/model.h"
#include "drla/types.h"

#include <torch/torch.h>

#include <variant>
#include <vector>

namespace drla
{

class FeatureExtractor : public torch::nn::Module
{
public:
	~FeatureExtractor();

	virtual torch::Tensor forward(const Observations& observations) = 0;
	virtual int get_output_size() const = 0;
};

inline FeatureExtractor::~FeatureExtractor()
{
}

std::shared_ptr<FeatureExtractor>
make_feature_extractor(const Config::FeatureExtractorConfig& config, const ObservationShapes& observation_shape);

} // namespace drla
