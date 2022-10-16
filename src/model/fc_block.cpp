#include "fc_block.h"

#include "model/utils.h"

#include <spdlog/spdlog.h>
#include <torch/torch.h>

#include <vector>

using namespace drla;
using namespace torch;

inline Config::FCConfig
configure_output(Config::FCConfig config, int output_size, Config::FCConfig::fc_layer output_layer_config)
{
	if (config.layers.empty() || config.layers.back().size != output_size)
	{
		output_layer_config.size = output_size;
		config.layers.push_back(std::move(output_layer_config));
	}
	else
	{
		config.layers.back() = std::move(output_layer_config);
	}
	return config;
}

FCBlockImpl::FCBlockImpl(const Config::FCConfig& config, int input_size) : config_(config)
{
	output_size_ = input_size;
	if (config_.layers.empty())
	{
		return;
	}

	make_fc(input_size);
}

FCBlockImpl::FCBlockImpl(
	const Config::FCConfig& config, int input_size, int output_size, Config::FCConfig::fc_layer output_layer_config)
		: config_(configure_output(config, output_size, std::move(output_layer_config)))
{
	make_fc(input_size);
}

void FCBlockImpl::make_fc(int input_size)
{
	spdlog::debug("Constructing {}", config_.name);

	size_t i = 0;
	int layer_size = config_.layers.front().use_densenet ? 0 : input_size;
	for (const auto& layer : config_.layers)
	{
		layer_size += layer.use_densenet ? input_size : 0;
		layers_.emplace_back(layer_size, layer.size);
		register_module(config_.name + std::to_string(i++), layers_.back());
		torch::nn::init::orthogonal_(layers_.back()->weight, layer.init_weight);
		torch::nn::init::constant_(layers_.back()->bias, layer.init_bias);

		spdlog::debug("Layer {}: {}", i, layer.size);
		layer_size = layer.size;
	}
	output_size_ = layer_size;
}

torch::Tensor FCBlockImpl::forward(const torch::Tensor& input)
{
	torch::Tensor x = input;
	for (size_t i = 0, ilen = layers_.size(); i < ilen; i++)
	{
		const auto& layer = config_.layers[i];
		if (layer.use_densenet && i > 0)
		{
			x = torch::cat({x, input}, -1);
		}
		x = make_activation(layer.activation)(layers_[i](x));
	}

	return x;
}

int FCBlockImpl::get_output_size() const
{
	return output_size_;
}
