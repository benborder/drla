#include "cnn_extractor.h"

#include <spdlog/fmt/ranges.h>
#include <spdlog/spdlog.h>

using namespace drla;
using namespace torch;

CnnExtractor::CnnExtractor(const Config::FeatureExtractorConfig& config, const ObservationShapes& observation_shape)
		: config_(std::get<Config::CNNConfig>(config))
{
	output_size_ = 0;
	size_t conv_index = 0;
	for (const auto& obs_shape : observation_shape)
	{
		std::vector<int64_t> out_shape;
		int64_t elements = 0;

		if (obs_shape.size() == 1)
		{
			out_shape = {obs_shape[0]};
			elements = obs_shape[0];
		}
		else if (obs_shape.size() == 2)
		{
			out_shape = {obs_shape[0], obs_shape[1]};
			elements = obs_shape[0] * obs_shape[1];
		}
		else if (obs_shape.size() == 3)
		{
			const auto& conv_layer_config = config_.conv_layers[conv_index];
			const auto config_in_channels = (int64_t)conv_layer_config.front().in_channels;

			int64_t in_channels = obs_shape[0];
			int64_t w = obs_shape[1];
			int64_t h = obs_shape[2];
			spdlog::debug("Constructing 2D CNN{}:", conv_index);
			if (in_channels != config_in_channels && config_in_channels != 0)
			{
				spdlog::error(
						"The environment has {} channels, but the network was configured for {}.", in_channels, config_in_channels);
				throw std::runtime_error("Invalid input channels");
			}
			size_t l = 0;
			std::vector<CNN> conv_net;
			for (const auto& layer_config : conv_layer_config)
			{
				conv_net.emplace_back(
						torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, layer_config.out_channels, layer_config.kernel_size)
																	.stride(layer_config.stride)));
				register_module(
						"conv" + std::to_string(conv_index) + std::to_string(l++), std::get<torch::nn::Conv2d>(conv_net.back()));

				torch::nn::init::orthogonal_(std::get<torch::nn::Conv2d>(conv_net.back())->weight, layer_config.init_weight);
				torch::nn::init::constant_(std::get<torch::nn::Conv2d>(conv_net.back())->bias, layer_config.init_bias);

				w = (w - layer_config.kernel_size + 2 * layer_config.padding) / layer_config.stride + 1;
				h = (h - layer_config.kernel_size + 2 * layer_config.padding) / layer_config.stride + 1;

				auto cnn_layer = fmt::format(
						"[ {}, {}, {}, {}, {} ]",
						in_channels,
						layer_config.out_channels,
						layer_config.kernel_size,
						layer_config.padding,
						layer_config.stride);
				spdlog::debug("{:<24}[{} {}]", cnn_layer, w, h);

				in_channels = layer_config.out_channels;
			}

			out_shape = {in_channels, w, h};
			elements = in_channels * w * h;
			conv_layers_.push_back(std::move(conv_net));
			conv_index++;
		}
		else if (obs_shape.size() == 4)
		{
			const auto& conv_layer_config = config_.conv_layers[conv_index];
			const auto config_in_channels = (int64_t)conv_layer_config.front().in_channels;

			int64_t in_channels = obs_shape[0];
			int64_t w = obs_shape[1];
			int64_t h = obs_shape[2];
			int64_t l = obs_shape[3];
			spdlog::debug("Constructing 3D CNN{}:", conv_index);
			if (config_in_channels != 0 && in_channels != config_in_channels)
			{
				spdlog::error(
						"The environment has {} channels, but the network was configured for {}.", in_channels, config_in_channels);
				throw std::runtime_error("Invalid input channels");
			}
			size_t i = 0;
			std::vector<CNN> conv_net;
			for (const auto& layer_config : conv_layer_config)
			{
				conv_net.emplace_back(
						torch::nn::Conv3d(torch::nn::Conv3dOptions(in_channels, layer_config.out_channels, layer_config.kernel_size)
																	.stride(layer_config.stride)));
				register_module(
						"conv" + std::to_string(conv_index) + std::to_string(i++), std::get<torch::nn::Conv3d>(conv_net.back()));

				torch::nn::init::orthogonal_(std::get<torch::nn::Conv3d>(conv_net.back())->weight, layer_config.init_weight);
				torch::nn::init::constant_(std::get<torch::nn::Conv3d>(conv_net.back())->bias, layer_config.init_bias);

				w = (w - layer_config.kernel_size + 2 * layer_config.padding) / layer_config.stride + 1;
				h = (h - layer_config.kernel_size + 2 * layer_config.padding) / layer_config.stride + 1;
				l = (l - layer_config.kernel_size + 2 * layer_config.padding) / layer_config.stride + 1;

				auto cnn_layer = fmt::format(
						"[ {}, {}, {}, {}, {} ]",
						in_channels,
						layer_config.out_channels,
						layer_config.kernel_size,
						layer_config.padding,
						layer_config.stride);
				spdlog::debug("{:<24}[{} {} {}]", cnn_layer, w, h, l);

				in_channels = layer_config.out_channels;
			}

			out_shape = {in_channels, w, h, l};
			elements = in_channels * w * h * l;
			conv_layers_.push_back(std::move(conv_net));
			conv_index++;
		}
		else
		{
			spdlog::error("Invalid observation shape: {}", fmt::join(obs_shape, ", "));
			throw std::runtime_error("Invalid observation shape");
		}

		output_size_ += elements;
		spdlog::debug(
				"{:<24}[{}] -> [{}] -> {}", "Observation:", fmt::join(obs_shape, ", "), fmt::join(out_shape, ", "), elements);
	}
	spdlog::debug("{:<24}[{}]", "Observation features:", output_size_);
}

torch::Tensor CnnExtractor::forward(const Observations& observations)
{
	int64_t hidden_index = 0;
	torch::Tensor hidden = torch::empty({observations.front().size(0), output_size_}, observations.front().device());
	for (size_t i = 0, j = 0; i < observations.size(); i++)
	{
		auto x = observations[i].to(torch::kFloat);
		if (x.dim() >= 4)
		{
			for (auto& conv : conv_layers_[j])
			{
				x = std::visit([&](auto& conv) { return torch::relu(conv(x)); }, conv);
			}
			++j;
		}
		auto fcsize = x.numel() / x.size(0);
		hidden.index(
				{torch::indexing::Slice(0, torch::indexing::None),
				 torch::indexing::Slice(hidden_index, hidden_index + fcsize)}) = x.view({x.size(0), fcsize});
		hidden_index += fcsize;
	}

	assert(hidden.size(1) == output_size_);

	return hidden;
}

int CnnExtractor::get_output_size() const
{
	return output_size_;
}
