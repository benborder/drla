#pragma once

#include "drla/configuration/model.h"
#include "drla/model/fc_block.h"
#include "drla/types.h"

#include <ATen/core/Tensor.h>
#include <torch/nn/module.h>

#include <memory>

namespace drla
{

class Distribution;

/// @brief A module which automatically switches between the types of output distributions based on the action space
/// configured at construction.
class ActorImpl : public torch::nn::Module
{
public:
	/// @brief Constructs a new ActorImpl object based on the supplied configuration options
	/// @param config The configuration, see `ActorConfig` for more details.
	/// @param inputs The flattened input size to use for the forward pass
	/// @param action_space The action space of the environment
	/// @param use_logits Indicated if the input is logits or probabilities. Defaults to logits.
	ActorImpl(const Config::ActorConfig& config, int inputs, const ActionSpace& action_space, bool use_logits = true);
	/// @brief Constructs a ActorImpl as a deep copy of other on the specified device
	/// @param other The object to clone from
	/// @param device The device to clone to
	ActorImpl(const ActorImpl& other, const c10::optional<torch::Device>& device);

	/// @brief Creates a distribution based on the latent input tensor and the configured action space of the environment
	/// @param latent The latent input to pass through the mlp and produce an output distribution
	/// @return An abstracted distribution which can be used to sample an action.
	std::unique_ptr<Distribution> forward(torch::Tensor latent);
	/// @brief Clones the ActorImpl, creating a deep copy.
	/// @param device An optional device parameter to use for the cloned block.
	/// @return A shared pointer to the cloned ActorImpl.
	std::shared_ptr<torch::nn::Module> clone(const c10::optional<torch::Device>& device = c10::nullopt) const override;

private:
	const Config::ActorConfig config_;
	ActionSpace action_space_;
	int num_actions_;
	int output_size_;
	bool use_logits_;
	FCBlock mlp_;
};

TORCH_MODULE(Actor);

} // namespace drla
