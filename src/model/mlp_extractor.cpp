#include "mlp_extractor.h"

#include <spdlog/spdlog.h>
#include <torch/torch.h>

using namespace drla;
using namespace torch;

MLPExtractor::MLPExtractor(const Config::MLPConfig& config, const std::vector<int64_t>& observation_shape)
		: hidden_(nullptr)
{
	int i = 0;
	output_size_ = std::accumulate(observation_shape.begin(), observation_shape.end(), 1, std::multiplies<>());
	hidden_ = register_module(config.name, FCBlock(config, output_size_));
	if (!config.layers.empty())
	{
		output_size_ = config.layers.back().size;
	}
}

torch::Tensor MLPExtractor::forward(const torch::Tensor& observation)
{
	return hidden_(observation);
}

std::vector<int64_t> MLPExtractor::get_output_shape() const
{
	return {output_size_};
}
