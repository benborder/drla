#pragma once

#include "drla/configuration/model.h"
#include "drla/types.h"

#include <ATen/core/Tensor.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/batchnorm.h>
#include <torch/nn/modules/conv.h>

#include <vector>

namespace drla
{

/// @brief Residual Block for 2D tensors. The input is the same size as the output.
class ResBlock2dImpl : public torch::nn::Module
{
public:
	/// @brief Constructs a new ResBlock2dImpl object with the given ResBlock2dConfig and channels.
	/// @param channels The number of channels the block has
	/// @param config The configuration of the internal CNN layers, see `ResBlock2dConfig` for more details.
	ResBlock2dImpl(int channels, Config::ResBlock2dConfig config = {});
	/// @brief Constructs a ResBlock2dImpl as a deep copy of other on the specified device
	/// @param other The object to clone from
	/// @param device The device to clone to
	ResBlock2dImpl(const ResBlock2dImpl& other, const c10::optional<torch::Device>& device);

	/// @brief Performs a forward pass through the residual block
	/// @param x The input tensor
	/// @return The output from the residual block
	torch::Tensor forward(torch::Tensor x);
	/// @brief Clones the fully connected block, creating a deep copy.
	/// @param device An optional device parameter to use for the cloned block.
	/// @return A shared pointer to the cloned fully connected block.
	std::shared_ptr<torch::nn::Module> clone(const c10::optional<torch::Device>& device = c10::nullopt) const override;

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
