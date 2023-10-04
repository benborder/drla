#pragma once

#include "drla/configuration/model.h"
#include "drla/types.h"

#include <torch/torch.h>

#include <memory>
#include <variant>
#include <vector>

namespace drla
{

/// @brief The FeatureExtractorGroup defines the interface for feature extraction modules. See `MLPExtractor` and
/// `CNNExtractor` for available modules.
class FeatureExtractorGroup : public torch::nn::Module
{
public:
	~FeatureExtractorGroup() = default;

	/// @brief Computes a forward pass through the feature extractor, returning the output tensor
	/// @param observation The observation tensor list to extract features from.Each tensor in the list has the shape
	/// [batch, channel, ...]
	/// @return The extracted features in latent space, having the shape [batch, ...], where ... can be obtained from
	/// `get_output_shape()`.
	virtual torch::Tensor forward(const torch::Tensor& observation) = 0;
	/// @brief Gets the shape of this extractor forward pass output tensor
	/// @return The shape of the forward pass tensor output
	virtual std::vector<int64_t> get_output_shape() const = 0;
	/// @brief Clones the feature extractor, creating a deep copy.
	/// @param device An optional device parameter to use for the cloned extractor.
	/// @return A shared pointer to the cloned feature extractor.
	virtual std::shared_ptr<torch::nn::Module> clone(const c10::optional<torch::Device>& device = c10::nullopt) const = 0;
};

/// @brief Provides feature extraction functionality, accepting an observation group list of tensors and outputting a
/// list of tensors representing the extracted features in latent space.
class FeatureExtractorImpl : public torch::nn::Module
{
public:
	/// @brief Constructs feature extractors according to the specified FeatureExtractorConfig and supplied observation
	/// shape. A single observation group is assigned to a single feature extractor group.
	/// @param config The configuration for the feature extractors
	/// @param observation_shape The observation shape for each observation group.
	FeatureExtractorImpl(const Config::FeatureExtractorConfig& config, const ObservationShapes& observation_shape);
	/// @brief Clones all feature extractors, creating a deep copy
	/// @param other The FeatureExtractorImpl to clone
	/// @param device An optional device parameter to use for the cloned extractor.
	FeatureExtractorImpl(const FeatureExtractorImpl& other, const c10::optional<torch::Device>& device);

	/// @brief Computes a forward pass through all feature extractors, returning the output tensor list
	/// @param observation The observation tensor list to extract features from. Each tensor in the list has the shape
	/// [batch, channel, ...]
	/// @return The extracted features in latent space, having the shape [batch, ...], where ... can be obtained from
	/// `get_output_shape()`.
	std::vector<torch::Tensor> forward(const Observations& observations);
	/// @brief Gets the shape of this extractors forward pass output tensor
	/// @return The shape of the forward pass tensor output
	std::vector<std::vector<int64_t>> get_output_shape() const;
	/// @brief Returns the total size of all outputs combined in terms of the number of elements
	/// @return The number of elements in the output of a forward pass
	int get_output_size() const;
	/// @brief Clones the feature extractor, creating a deep copy.
	/// @param device An optional device parameter to use for the cloned block.
	/// @return A shared pointer to the cloned feature extractor.
	std::shared_ptr<torch::nn::Module> clone(const c10::optional<torch::Device>& device = c10::nullopt) const override;

private:
	std::vector<std::shared_ptr<FeatureExtractorGroup>> feature_extractors_;
	std::vector<std::vector<int64_t>> output_shape_;
	int output_size_;
};

TORCH_MODULE(FeatureExtractor);

/// @brief Configures the FeatureExtractor to creates a multi encoder architecture
/// @param config The multi encoder configuration. Can use the simpler MultiEncoderNetworkConfig or create a custom
/// config using FeatureExtractorConfig
/// @param input_shape The input observation shape to the encoder
/// @return The FeatureExtractor configured into an encoder
FeatureExtractor make_multi_encoder(const Config::MultiEncoderConfig& config, const ObservationShapes& input_shape);

/// @brief Configures the FeatureExtractor to creates a multi decoder architecture
/// @param config The multi decoder configuration. Can use the simpler MultidecoderNetworkConfig or create a custom
/// config using FeatureExtractorConfig
/// @param input_shape The input latent shapes to the decoder
/// @param output_shape The output observation shapes of the decoder
/// @return The FeatureExtractor configured into a decoder
FeatureExtractor make_multi_decoder(
	const Config::MultiDecoderConfig& config,
	const ObservationShapes& input_shape,
	const ObservationShapes& output_shape);

} // namespace drla
