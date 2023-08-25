#pragma once

#include "drla/configuration/model.h"
#include "drla/types.h"

#include <torch/torch.h>

#include <map>
#include <vector>

namespace drla
{

/// @brief The FCBlockImpl serves as a building block for neural networks. It constructs a Feedforward Neural Network
/// with a fully connected layer, where the user can specify how many layers and neurons to have.
class FCBlockImpl : public torch::nn::Module
{
public:
	/// @brief Construct a new FCBlockImpl object with the given FCConfig and input_size.
	/// @param config The FCConfig to use for creating the FC layers.
	/// @param name Name of the block.
	/// @param input_size The input size of the first layer.
	FCBlockImpl(const Config::FCConfig& config, const std::string& name, int input_size);
	/// @brief Construct a new FCBlockImpl object with the given FCConfig and input_size and an output layer.
	/// @param config The FCConfig to use for creating the FC layers.
	/// @param name The name of the block.
	/// @param input_size The input size of the first layer.
	/// @param output_layer_config The configuration for the output layer.
	FCBlockImpl(
		const Config::FCConfig& config, const std::string& name, int input_size, Config::LinearConfig output_layer_config);
	/// @brief Constructs a FCBlockImpl as a deep copy of other on the specified device
	/// @param other The object to clone from
	/// @param device The device to clone to
	FCBlockImpl(const FCBlockImpl& other, const c10::optional<torch::Device>& device);

	/// @brief Computes the forward pass through the fully-connected block.
	/// @param input Tensor of shape (batch_size, input_size) representing the input features.
	/// @return Tensor of shape (batch_size, output_size) representing the output of the fully-connected block.
	torch::Tensor forward(const torch::Tensor& input);
	/// @brief Gets the output size of the fully connected block.
	/// @return The output size of the fully connected block.
	int get_output_size() const;
	/// @brief Clones the fully connected block, creating a deep copy.
	/// @param device An optional device parameter to use for the cloned block.
	/// @return A shared pointer to the cloned fully connected block.
	std::shared_ptr<torch::nn::Module> clone(const c10::optional<torch::Device>& device = c10::nullopt) const override;

private:
	void make_fc(int input_size, const std::string& name);

	const Config::FCConfig config_;

	using Layer = std::variant<torch::nn::Linear, torch::nn::LayerNorm, ActivationFunction>;
	std::vector<Layer> layers_;
	int output_size_;
	std::multimap<int, Config::LayerConnectionConfig> connections_;
};

TORCH_MODULE(FCBlock);

} // namespace drla
