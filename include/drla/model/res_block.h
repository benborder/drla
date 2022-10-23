#pragma once

#include "drla/configuration/model.h"
#include "drla/types.h"

#include <torch/torch.h>

#include <vector>

namespace drla
{

/// @brief Residual Block for 2D tensors.
class ResBlock2dImpl : public torch::nn::Module
{
public:
	ResBlock2dImpl(int channels, Config::ResBlock2dConfig config = {});

	torch::Tensor forward(torch::Tensor x);

private:
	struct ResLayer
	{
		torch::nn::Conv2d conv;
		torch::nn::BatchNorm2d bn = nullptr;
	};

	std::vector<ResLayer> res_layers_;
};

TORCH_MODULE(ResBlock2d);

} // namespace drla
