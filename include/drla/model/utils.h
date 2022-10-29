#pragma once

#include "drla/model.h"

#include <torch/torch.h>

#include <functional>

namespace drla
{

inline torch::Tensor flatten(const std::vector<torch::Tensor>& features)
{
	torch::Tensor flattened = torch::empty({features.front().size(0), 0}, features.front().device());
	for (auto& feature : features) { flattened = torch::cat({flattened, feature.view({feature.size(0), -1})}, 1); }
	return flattened;
}

inline std::function<torch::Tensor(const torch::Tensor&)> make_activation(Config::Activation activation)
{
	switch (activation)
	{
		case Config::Activation::kNone: break;
		case Config::Activation::kReLU: return [](const torch::Tensor& x) { return torch::relu(x); };
		case Config::Activation::kLeakyReLU: return [](const torch::Tensor& x) { return torch::leaky_relu(x); };
		case Config::Activation::kTanh: return [](const torch::Tensor& x) { return torch::tanh(x); };
		case Config::Activation::kSigmoid: return [](const torch::Tensor& x) { return torch::sigmoid(x); };
		case Config::Activation::kELU: return [](const torch::Tensor& x) { return torch::elu(x); };
		case Config::Activation::kSoftplus: return [](const torch::Tensor& x) { return torch::softplus(x); };
	}
	return [](const torch::Tensor& x) { return x; };
}

inline torch::Tensor activation(torch::Tensor x, Config::Activation activation)
{
	switch (activation)
	{
		case Config::Activation::kNone: break;
		case Config::Activation::kReLU: return torch::relu(x);
		case Config::Activation::kLeakyReLU: return torch::leaky_relu(x);
		case Config::Activation::kTanh: return torch::tanh(x);
		case Config::Activation::kSigmoid: return torch::sigmoid(x);
		case Config::Activation::kELU: return torch::elu(x);
		case Config::Activation::kSoftplus: return torch::softplus(x);
	}
	return x;
}

} // namespace drla
