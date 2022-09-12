#include "feature_extractor.h"

#include "cnn_extractor.h"
#include "mlp_extractor.h"

using namespace drla;

std::shared_ptr<FeatureExtractor>
drla::make_feature_extractor(const Config::FeatureExtractorConfig& config, const ObservationShapes& observation_shape)
{
	if (std::holds_alternative<Config::MLPConfig>(config))
	{
		return std::make_shared<MlpExtractor>(config, observation_shape);
	}
	else if (std::holds_alternative<Config::CNNConfig>(config))
	{
		return std::make_shared<CnnExtractor>(config, observation_shape);
	}
	else
	{
		throw std::runtime_error("Invalid feature extractor type");
	}
}
