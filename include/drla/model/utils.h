#pragma once

#include "drla/model.h"

#include <torch/torch.h>

#include <functional>
#include <string>

namespace drla
{

/// @brief Flattens the input tensor list to a single 1D output tensor
/// @param features The input tensor list to flatten
/// @return The flattened output tensor
inline torch::Tensor flatten(const std::vector<torch::Tensor>& features)
{
	torch::Tensor flattened = torch::empty({features.front().size(0), 0}, features.front().device());
	for (auto& feature : features) { flattened = torch::cat({flattened, feature.view({feature.size(0), -1})}, 1); }
	return flattened;
}

/// @brief Creates and activation function based on the type specified
/// @param activation The activation function type
/// @return A std::function of the activation function
inline ActivationFunction make_activation(Config::Activation activation)
{
	switch (activation)
	{
		case Config::Activation::kNone: break;
		case Config::Activation::kReLU: return [](const torch::Tensor& x) { return torch::relu(x); };
		case Config::Activation::kLeakyReLU: return [](const torch::Tensor& x) { return torch::leaky_relu(x); };
		case Config::Activation::kTanh: return [](const torch::Tensor& x) { return torch::tanh(x); };
		case Config::Activation::kSigmoid: return [](const torch::Tensor& x) { return torch::sigmoid(x); };
		case Config::Activation::kELU: return [](const torch::Tensor& x) { return torch::elu(x); };
		case Config::Activation::kSiLU: return [](const torch::Tensor& x) { return torch::silu(x); };
		case Config::Activation::kSoftplus: return [](const torch::Tensor& x) { return torch::softplus(x); };
	}
	return [](const torch::Tensor& x) { return x; };
}

/// @brief Performs the specified activation function on the input x
/// @param x The input to perform activation on
/// @param activation The activation function type to use
/// @return The activated output
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
		case Config::Activation::kSiLU: return torch::silu(x);
		case Config::Activation::kSoftplus: return torch::softplus(x);
	}
	return x;
}

/// @brief Gets the string of the specified activation function
/// @param activation The activation function type
/// @return A string of the activation function type
inline std::string activation_name(Config::Activation activation)
{
	switch (activation)
	{
		case Config::Activation::kNone: break;
		case Config::Activation::kReLU: return "ReLU";
		case Config::Activation::kLeakyReLU: return "Leaky ReLU";
		case Config::Activation::kTanh: return "Tanh";
		case Config::Activation::kSigmoid: return "Sigmoid";
		case Config::Activation::kELU: return "ELU";
		case Config::Activation::kSiLU: return "SiLU";
		case Config::Activation::kSoftplus: return "Softplus";
	}
	return "Unity";
}

/// @brief Performs initialisation on the specified tensor
/// @param weight The tensor to perform initialisation on
/// @param type The type of initialisation to perform
/// @param init_value The value to use for initialisation. This value is dependent on the activation type and not
/// necessarily used.
inline void weight_init(torch::Tensor weight, Config::InitType type, double init_value)
{
	switch (type)
	{
		case Config::InitType::kDefault: return;
		case Config::InitType::kConstant: torch::nn::init::constant_(weight, init_value); return;
		case Config::InitType::kOrthogonal: torch::nn::init::orthogonal_(weight, init_value); return;
		case Config::InitType::kKaimingUniform: torch::nn::init::kaiming_uniform_(weight, init_value); return;
		case Config::InitType::kKaimingNormal: torch::nn::init::kaiming_normal_(weight, init_value); return;
		case Config::InitType::kXavierUniform: torch::nn::init::xavier_uniform_(weight, init_value); return;
		case Config::InitType::kXavierNormal: torch::nn::init::xavier_normal_(weight, init_value); return;
	}
}

inline int64_t flatten(const std::vector<int64_t>& x)
{
	return std::accumulate(x.begin(), x.end(), 1, std::multiplies<>());
}

inline int64_t flatten(const std::vector<std::vector<int64_t>>& x)
{
	int64_t output_size = 0;
	for (auto& shape : x) { output_size += flatten(shape); }
	return output_size;
}

inline torch::Tensor normalise(const torch::Tensor& x)
{
	torch::Tensor min;
	torch::Tensor max;
	if (x.dim() == 4)
	{
		min = std::get<0>(x.view({x.size(0), x.size(1), -1}).min(2)).view({x.size(0), x.size(1), 1, 1});
		max = std::get<0>(x.view({x.size(0), x.size(1), -1}).max(2)).view({x.size(0), x.size(1), 1, 1});
	}
	else
	{
		std::vector<int64_t> shape(x.dim(), 1);
		shape[0] = x.size(0);
		min = std::get<0>(x.view({x.size(0), -1}).min(1)).view(shape);
		max = std::get<0>(x.view({x.size(0), -1}).max(1)).view(shape);
	}
	return (x - min) / (max - min).clamp_min(1e-5);
}

inline std::vector<std::vector<int64_t>> condense_shape(const std::vector<std::vector<int64_t>>& input_shape)
{
	std::vector<std::vector<int64_t>> output_shapes;

	for (auto& shape : input_shape)
	{
		auto size = shape.size();

		bool matched = false;
		for (auto& output_shape : output_shapes)
		{
			if (output_shape == shape)
			{
				matched = true;
				// Image based tensor dims are assumed to be of the format [channels, height, width] and appended in the
				// channels dim
				if (size == 3)
				{
					output_shape[0] += shape[0];
				}
				// data based tensor dims are assumed to be of the format [..., data] and appended in the data dim
				else
				{
					output_shape[size - 1] += shape[size - 1];
				}
			}
		}
		if (!matched)
		{
			output_shapes.push_back(shape);
		}
	}

	return output_shapes;
}

inline std::vector<torch::Tensor> condense(const std::vector<torch::Tensor>& input)
{
	std::vector<torch::Tensor> output;

	for (auto& input_tensor : input)
	{
		bool matched = false;
		for (auto& output_tensor : output)
		{
			if (output_tensor.sizes() == input_tensor.sizes())
			{
				matched = true;
				// Image based tensor dims are assumed to be of the format [batch, channels, height, width] and appended in the
				// channels dim. Other tensors are assumed to be of the format [batch, data] and appended in the second dim
				output_tensor = torch::cat({output_tensor, input_tensor}, 1);
			}
		}
		if (!matched)
		{
			output.push_back(input_tensor);
		}
	}

	return output;
}

/// @brief Gets the stacked observation shape from the input single observation
/// @param shape The input single observation shape to base the stack from
/// @param stack_size The size of the stack to create. The stack size cannot be negative.
/// @return The stacked observation shape
inline ObservationShapes stacked_observation_shape(const ObservationShapes& shape, int stack_size)
{
	stack_size = std::max(stack_size, 0);
	ObservationShapes stacked_shape;
	for (const auto& dims : shape)
	{
		auto stacked_dims = dims;
		if (dims.size() < 3)
		{
			stacked_dims.back() *= stack_size + 1;
			stacked_dims.back() += stack_size;
		}
		else
		{
			stacked_dims.front() *= stack_size + 1;
			stacked_dims.front() += stack_size;
		}
		stacked_shape.push_back(stacked_dims);
	}
	return stacked_shape;
}

} // namespace drla
