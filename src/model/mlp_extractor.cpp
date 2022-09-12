#include "mlp_extractor.h"

#include <spdlog/spdlog.h>
#include <torch/torch.h>

using namespace drla;
using namespace torch;

MlpExtractor::MlpExtractor(const Config::FeatureExtractorConfig& config, const ObservationShapes& observation_shape)
		: hidden_(nullptr)
{
	auto mlp_config = std::get<Config::MLPConfig>(config);
	int i = 0;
	output_size_ = 0;
	for (auto& obs_shape : observation_shape)
	{
		output_size_ += std::accumulate(obs_shape.begin(), obs_shape.end(), 1, std::multiplies<>());
	}
	hidden_ = register_module(mlp_config.name, FCBlock(mlp_config, output_size_));
	if (!mlp_config.layers.empty())
	{
		output_size_ = mlp_config.layers.back().size;
	}
}

torch::Tensor MlpExtractor::forward(const Observations& observations)
{
	return hidden_(torch::cat(observations));
}

int MlpExtractor::get_output_size() const
{
	return output_size_;
}
