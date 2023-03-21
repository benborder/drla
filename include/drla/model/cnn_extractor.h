#pragma once

#include "drla/configuration/model.h"
#include "drla/model/feature_extractor.h"
#include "drla/model/res_block.h"

#include <torch/torch.h>

#include <functional>
#include <variant>
#include <vector>

namespace drla
{

/// @brief The CNNExtractor performs feature extraction via 2D CNN, batch norm, pooling and residual blocks. The layers
/// can be configured and the final output shape is automatically calculated.
class CNNExtractor : public FeatureExtractorGroup
{
public:
	/// @brief Construct a new CNNExtractor object with the given CNNConfig and observation shape
	/// @param config CNNConfig to use for creating the CNN extractor and its layers
	/// @param observation_shape The input observation shape to the CNN extractor
	CNNExtractor(const Config::CNNConfig& config, const std::vector<int64_t>& observation_shape);
	/// @brief Constructs a CNNExtractor as a deep copy of other on the specified device
	/// @param other The object to clone from
	/// @param device The device to clone to
	CNNExtractor(const CNNExtractor& other, const c10::optional<torch::Device>& device);

	/// @brief Computes a forward pass through the CNN extractor, returning the output tensor
	/// @param observation The observation tensor list to extract features from. Each tensor in the list has the shape
	/// [batch, channel, ...]
	/// @return The extracted features in latent space, having the shape [batch, ...], where ... can be obtained from
	/// `get_output_shape()`.
	torch::Tensor forward(const torch::Tensor& observation) override;
	/// @brief Gets the shape of this extractor forward pass output tensor
	/// @return The shape of the forward pass tensor output
	std::vector<int64_t> get_output_shape() const override;
	/// @brief Clones the CNNExtractor, creating a deep copy.
	/// @param device An optional device parameter to use for the cloned block.
	/// @return A shared pointer to the cloned CNNExtractor.
	std::shared_ptr<torch::nn::Module> clone(const c10::optional<torch::Device>& device = c10::nullopt) const override;

private:
	using Layer = std::variant<
		torch::nn::Conv2d,
		torch::nn::BatchNorm2d,
		torch::nn::MaxPool2d,
		torch::nn::AvgPool2d,
		torch::nn::AdaptiveAvgPool2d,
		ResBlock2d,
		std::function<torch::Tensor(const torch::Tensor&)>>;
	std::vector<Layer> cnn_layers_;
	std::vector<int64_t> out_shape_;
};

} // namespace drla
