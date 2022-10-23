#include "cnn_extractor.h"

#include "model/utils.h"

#include <spdlog/fmt/ranges.h>
#include <spdlog/spdlog.h>

using namespace drla;
using namespace torch;

CNNExtractor::CNNExtractor(const Config::CNNConfig& config, const std::vector<int64_t>& observation_shape)
{
	if (observation_shape.size() != 3)
	{
		spdlog::error(
			"Invalid observation shape: [{}]. The shape must be in the form of: [channels, width, height].",
			fmt::join(observation_shape, ", "));
		throw std::runtime_error("Invalid observation shape");
	}

	const int64_t config_in_channels = std::visit(
		[](const auto& config) {
			using T = std::decay_t<decltype(config)>;
			if constexpr (std::is_same_v<Config::Conv2dConfig, T>)
			{
				return config.in_channels;
			}
			return 0;
		},
		config.layers.front());
	int64_t in_channels = observation_shape[0];
	int64_t w = observation_shape[1];
	int64_t h = observation_shape[2];
	spdlog::debug("Constructing CNN:");
	if (in_channels != config_in_channels && config_in_channels != 0)
	{
		spdlog::error(
			"The environment has {} channels, but the network was configured for {} input channels.",
			in_channels,
			config_in_channels);
		throw std::runtime_error("Invalid input channels");
	}
	size_t l = 0;
	for (const auto& layer_config : config.layers)
	{
		std::visit(
			[&](const auto& config) {
				using T = std::decay_t<decltype(config)>;
				if constexpr (std::is_same_v<Config::Conv2dConfig, T>)
				{
					auto conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, config.out_channels, config.kernel_size)
																					.stride(config.stride)
																					.padding(config.padding)
																					.bias(config.use_bias));
					register_module("cnn_conv" + std::to_string(l++), conv);

					torch::nn::init::orthogonal_(conv->weight, config.init_weight);
					if (config.use_bias)
					{
						torch::nn::init::constant_(conv->bias, config.init_bias);
					}

					cnn_layers_.emplace_back(std::make_pair(std::move(conv), make_activation(config.activation)));

					w = (w - config.kernel_size + 2 * config.padding) / config.stride + 1;
					h = (h - config.kernel_size + 2 * config.padding) / config.stride + 1;

					auto conv_layer = fmt::format(
						"conv2d[ {}, {}, {}, {}, {} ]",
						in_channels,
						config.out_channels,
						config.kernel_size,
						config.padding,
						config.stride);
					spdlog::debug("{:<28}[{} {}]", conv_layer, w, h);

					in_channels = config.out_channels;
				}
				if constexpr (std::is_same_v<Config::MaxPool2dConfig, T>)
				{
					cnn_layers_.emplace_back(torch::nn::MaxPool2d(
						torch::nn::MaxPool2dOptions(config.kernel_size).stride(config.stride).padding(config.padding)));
					register_module("cnn_maxpool" + std::to_string(l++), std::get<torch::nn::MaxPool2d>(cnn_layers_.back()));

					w = (w - config.kernel_size + 2 * config.padding) / config.stride + 1;
					h = (h - config.kernel_size + 2 * config.padding) / config.stride + 1;

					auto avgpool_layer =
						fmt::format("maxpool2d[ {}, {}, {} ]", config.kernel_size, config.padding, config.stride);
					spdlog::debug("{:<28}[{} {}]", avgpool_layer, w, h);
				}
				if constexpr (std::is_same_v<Config::AvgPool2dConfig, T>)
				{
					cnn_layers_.emplace_back(torch::nn::AvgPool2d(
						torch::nn::AvgPool2dOptions(config.kernel_size).stride(config.stride).padding(config.padding)));
					register_module("cnn_avgpool" + std::to_string(l++), std::get<torch::nn::AvgPool2d>(cnn_layers_.back()));

					w = (w - config.kernel_size + 2 * config.padding) / config.stride + 1;
					h = (h - config.kernel_size + 2 * config.padding) / config.stride + 1;

					auto avgpool_layer =
						fmt::format("avgpool2d[ {}, {}, {} ]", config.kernel_size, config.padding, config.stride);
					spdlog::debug("{:<28}[{} {}]", avgpool_layer, w, h);
				}
				if constexpr (std::is_same_v<Config::AdaptiveAvgPool2dConfig, T>)
				{
					cnn_layers_.emplace_back(torch::nn::AdaptiveAvgPool2d(config.size));
					register_module("cnn_adaptavgpool" + std::to_string(l++), std::get<torch::nn::AvgPool2d>(cnn_layers_.back()));

					w = config.size[1];
					h = config.size[0];

					spdlog::debug("{:<28}[{} {}]", "adaptive_avgpool2d", w, h);
				}
				if constexpr (std::is_same_v<Config::ResBlock2dConfig, T>)
				{
					cnn_layers_.emplace_back(ResBlock2d(in_channels, config));
					register_module("resnet_resblock" + std::to_string(l++), std::get<ResBlock2d>(cnn_layers_.back()));

					spdlog::debug("{:<28}[{} {}]", "resblock2d", w, h);
				}
			},
			layer_config);
	}

	out_shape_ = {in_channels, w, h};
}

torch::Tensor CNNExtractor::forward(const torch::Tensor& observation)
{
	auto x = observation.to(torch::kFloat);
	for (auto& cnn_layer : cnn_layers_)
	{
		x = std::visit(
			[&x](auto& layer) {
				using T = std::decay_t<decltype(layer)>;
				if constexpr (std::is_same_v<Conv2d, T>)
				{
					return layer.second(layer.first(x));
				}
				else
				{
					return layer(x);
				}
			},
			cnn_layer);
	}

	return x;
}

std::vector<int64_t> CNNExtractor::get_output_shape() const
{
	return out_shape_;
}
