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
	if (config.layers.empty() || (config.layers.back().size > 0 && config.layers.back().size != output_size))
	{
		output_layer_config.size = output_size;
		config.layers.push_back(std::move(output_layer_config));
	}
	else if (config.layers.back().size <= 0)
	{
		config.layers.back().size = output_size;
	}
	else
	{
		config.layers.back() = std::move(output_layer_config);
	}
	return config;
}

FCBlockImpl::FCBlockImpl(const FCBlockImpl& other, const c10::optional<torch::Device>& device)
		: config_(other.config_), output_size_(other.output_size_), has_multi_connected_(other.has_multi_connected_)
{
	int index = 0;
	for (auto& layer : other.layers_)
	{
		layers_.emplace_back(std::dynamic_pointer_cast<torch::nn::LinearImpl>(layer->clone(device)));
		register_module(other.named_children()[index++].key(), layers_.back());
	}
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
	int layer_size = input_size;
	int total_neurons = input_size;
	for (const auto& layer : config_.layers)
	{
		std::string type;
		switch (layer.type)
		{
			case Config::FCLayerType::kLinear:
			{
				type = "linear";
				layers_.emplace_back(layer_size, layer.size);
				total_neurons += layer.size;
				break;
			}
			case Config::FCLayerType::kInputConnected:
			{
				type = "input-connected";
				layers_.emplace_back(layer_size + input_size, layer.size);
				total_neurons += layer.size;
				break;
			}
			case Config::FCLayerType::kMultiConnected:
			{
				type = "multi-connected";
				layers_.emplace_back(total_neurons, layer.size);
				total_neurons += layer.size;
				has_multi_connected_ = true;
				break;
			}
			case Config::FCLayerType::kResidual:
			{
				type = "residual";
				if (layer.size != 0 && layer.size != input_size)
				{
					spdlog::warn("Residual: Using output size `{}` instead of specified input size `{}`", input_size, layer.size);
				}
				layers_.emplace_back(layer_size, input_size);
				total_neurons += layer.size;
				break;
			}
			case Config::FCLayerType::kForwardInput:
			{
				if (i + 1 < config_.layers.size())
				{
					spdlog::warn("Skipping layers after ForwardInput layer. The ForwardInput layer should be the final layer.");
				}
				output_size_ = layer_size + input_size;
				return;
			}
			case Config::FCLayerType::kForwardAll:
			{
				if (i + 1 < config_.layers.size())
				{
					spdlog::warn("Skipping layers after ForwardAll layer. The ForwardAll layer should be the final layer.");
				}
				has_multi_connected_ = true;
				output_size_ = total_neurons;
				return;
			}
		}
		register_module(config_.name + std::to_string(i++), layers_.back());
		torch::nn::init::orthogonal_(layers_.back()->weight, layer.init_weight);
		torch::nn::init::constant_(layers_.back()->bias, layer.init_bias);

		layer_size = layer.size;

		spdlog::debug("Layer {}: {} ({})", i, layer.type == Config::FCLayerType::kResidual ? input_size : layer.size, type);
	}

	output_size_ = layer_size;
}

torch::Tensor FCBlockImpl::forward(const torch::Tensor& input)
{
	torch::Tensor x = input;
	torch::Tensor outs = input;
	for (size_t i = 0, ilen = config_.layers.size(); i < ilen; i++)
	{
		const auto& layer = config_.layers[i];
		switch (layer.type)
		{
			case Config::FCLayerType::kLinear: x = layers_[i](x); break;
			case Config::FCLayerType::kInputConnected: x = layers_[i](i > 0 ? torch::cat({x, input}, -1) : x); break;
			case Config::FCLayerType::kMultiConnected: x = layers_[i](outs); break;
			case Config::FCLayerType::kResidual: x = layers_[i](x) + input; break;
			case Config::FCLayerType::kForwardInput: return i > 0 ? torch::cat({x, input}, -1) : x;
			case Config::FCLayerType::kForwardAll: return outs;
		}
		x = activation(x, layer.activation);
		if (has_multi_connected_)
		{
			outs = torch::cat({outs, x}, -1);
		}
	}

	return x;
}

int FCBlockImpl::get_output_size() const
{
	return output_size_;
}

std::shared_ptr<torch::nn::Module> FCBlockImpl::clone(const c10::optional<torch::Device>& device) const
{
	torch::NoGradGuard no_grad;
	return std::make_shared<FCBlockImpl>(static_cast<const FCBlockImpl&>(*this), device);
}
