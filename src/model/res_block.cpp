#include "res_block.h"

#include <spdlog/spdlog.h>
#include <torch/torch.h>

#include <vector>

using namespace drla;
using namespace torch;

ResBlock2dImpl::ResBlock2dImpl(int channels, Config::ResBlock2dConfig config)
{
	for (int i = 0; i < config.layers; ++i)
	{
		auto layer = ResLayer{
			torch::nn::Conv2d(
				torch::nn::Conv2dOptions(channels, channels, config.kernel_size).stride(config.stride).padding(1).bias(false)),
			nullptr};
		register_module("conv" + std::to_string(i), layer.conv);
		torch::nn::init::orthogonal_(layer.conv->weight, config.init_weight);
		if (config.normalise)
		{
			layer.bn = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(channels));
			register_module("bn" + std::to_string(i), layer.bn);
		}
		res_layers_.push_back(std::move(layer));
	}
}

ResBlock2dImpl::ResBlock2dImpl(const ResBlock2dImpl& other, const c10::optional<torch::Device>& device)
{
	int index = 0;
	for (auto& res_layer : other.res_layers_)
	{
		auto layer = ResLayer{std::dynamic_pointer_cast<torch::nn::Conv2dImpl>(res_layer.conv->clone(device)), nullptr};
		register_module(other.named_children()[index++].key(), layer.conv);
		if (!res_layer.bn.is_empty())
		{
			layer.bn = std::dynamic_pointer_cast<torch::nn::BatchNorm2dImpl>(res_layer.bn->clone(device));
			register_module(other.named_children()[index++].key(), layer.bn);
		}

		res_layers_.emplace_back(std::move(layer));
	}
}

torch::Tensor ResBlock2dImpl::forward(torch::Tensor x)
{
	auto v = x;
	auto last_iter = std::prev(res_layers_.end());
	for (auto layer_iter = res_layers_.begin(); layer_iter != res_layers_.end(); ++layer_iter)
	{
		v = layer_iter->conv(v);
		if (!layer_iter->bn.is_empty())
		{
			v = layer_iter->bn(v);
		}
		if (layer_iter == last_iter)
		{
			v += x;
		}
		v = torch::relu(v);
	}
	return v;
}

std::shared_ptr<torch::nn::Module> ResBlock2dImpl::clone(const c10::optional<torch::Device>& device) const
{
	torch::NoGradGuard no_grad;
	return std::make_shared<ResBlock2dImpl>(static_cast<const ResBlock2dImpl&>(*this), device);
}
