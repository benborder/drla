#pragma once

#include "drla/configuration/model.h"
#include "drla/types.h"

#include <torch/torch.h>

#include <functional>
#include <string>
#include <vector>

namespace drla
{

/// @brief Flattens the input tensor list to a single 1D output tensor
/// @param features The input tensor list to flatten
/// @return The flattened output tensor
torch::Tensor flatten(const std::vector<torch::Tensor>& x, int dims = 0);

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

/// @brief Flattens the shape into a scalar
/// @param x The input shape to flatten
/// @return The flattened size
inline int64_t flatten(const std::vector<int64_t>& x)
{
	return std::accumulate(x.begin(), x.end(), 1, std::multiplies<>());
}

/// @brief Flattens a list of shapes into a single scalar
/// @param x The list of input shapes to flatten
/// @return The flattened size
inline int64_t flatten(const std::vector<std::vector<int64_t>>& x)
{
	int64_t output_size = 0;
	for (auto& shape : x) { output_size += flatten(shape); }
	return output_size;
}

/// @brief Normalises a tensor along its data dims (1D or 2D)
/// @param x The tensor to normalise
/// @param dims The number of dims to ignore when normalising starting from the first dim 0 (i.e. if dims=1,
/// normalisation will only occur along each element in the 0 dim, for example a batch dim)
/// @return The normalised tensor
torch::Tensor normalise(const torch::Tensor& x, int dims);

/// @brief Concatenates shapes in a vector that are identical, combining them together. If no shapes are matched, the
/// list is unchanged.
/// @param input_shape The input list of shapes
/// @return The output list of condensed shapes
std::vector<std::vector<int64_t>> condense_shape(const std::vector<std::vector<int64_t>>& input_shape);

/// @brief Concatenates tensors in a vector that have the same dimensions, combining them together. If no tensors are
/// matched, the list is unchanged.
/// @param input The input list of tensors
/// @param dim The dimension to concatenate along
/// @return The output list of condensed tensors
std::vector<torch::Tensor> condense(const std::vector<torch::Tensor>& input, int dim = 1);

/// @brief Gets the stacked observation shape from the input single observation
/// @param shape The input single observation shape to base the stack from
/// @param stack_size The size of the stack to create. The stack size cannot be negative.
/// @return The stacked observation shape
ObservationShapes stacked_observation_shape(const ObservationShapes& shape, int stack_size);

inline void
update_params(const std::vector<torch::Tensor>& current, const std::vector<torch::Tensor>& target, double tau)
{
	for (size_t i = 0; i < current.size(); i++) { target[i].mul_(1.0 - tau).add_(current[i], tau); }
}

} // namespace drla
