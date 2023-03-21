#pragma once

#include "drla/configuration/model.h"
#include "drla/model/fc_block.h"
#include "drla/model/feature_extractor.h"

#include <torch/torch.h>

#include <vector>

namespace drla
{

/// @brief The MLPExtractor performs feature extraction via a MLP network. it is basically a wrapper around FCBlock and
/// inherits the same configuration
class MLPExtractor : public FeatureExtractorGroup
{
public:
	/// @brief Construct a new MLPExtractor object with the given MLPConfig and observation shape
	/// @param config MLPConfig to use for creating the MLP extractor and its layers
	/// @param observation_shape The input observation shape to the MLP extractor
	MLPExtractor(const Config::MLPConfig& config, const std::vector<int64_t>& observation_shape);
	/// @brief Constructs a MLPExtractor as a deep copy of other on the specified device
	/// @param other The object to clone from
	/// @param device The device to clone to
	MLPExtractor(const MLPExtractor& other, const c10::optional<torch::Device>& device);

	/// @brief Computes a forward pass through the CNN extractor, returning the output tensor
	/// @param observation The observation tensor list to extract features from. Each tensor in the list has the shape
	/// [batch, channel, data]
	/// @return The extracted features in latent space, having the shape [batch, data], where ... can be obtained from
	/// `get_output_shape()`.
	torch::Tensor forward(const torch::Tensor& observation) override;
	/// @brief Gets the shape of this extractor forward pass output tensor
	/// @return The shape of the forward pass tensor output
	std::vector<int64_t> get_output_shape() const override;
	/// @brief Clones the MLPExtractor, creating a deep copy.
	/// @param device An optional device parameter to use for the cloned block.
	/// @return A shared pointer to the cloned MLPExtractor.
	std::shared_ptr<torch::nn::Module> clone(const c10::optional<torch::Device>& device = c10::nullopt) const override;

private:
	FCBlock hidden_;
	int output_size_;
};
} // namespace drla
