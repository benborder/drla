#pragma once

#include "drla/configuration/model.h"
#include "drla/types.h"

#include <torch/torch.h>

#include <memory>

namespace drla
{

class Distribution;

/// @brief A module which automatically switches between the types of output distributions based on the action space
/// configured at construction.
class PolicyActionOutputImpl : public torch::nn::Module
{
public:
	/// @brief Constructs a new PolicyActionOutputImpl object based on the supplied configuration options
	/// @param config The configuration, see `PolicyActionOutputConfig` for more details.
	/// @param inputs The flattened input size to use for the forward pass
	/// @param action_space The action space of the environment
	/// @param use_logits Indicated if the input is logits or probabilities. Defaults to logits.
	PolicyActionOutputImpl(
		const Config::PolicyActionOutputConfig& config,
		int inputs,
		const ActionSpace& action_space,
		bool use_logits = true);
	/// @brief Constructs a PolicyActionOutputImpl as a deep copy of other on the specified device
	/// @param other The object to clone from
	/// @param device The device to clone to
	PolicyActionOutputImpl(const PolicyActionOutputImpl& other, const c10::optional<torch::Device>& device);

	/// @brief Creates a distribution based on the latent input tensor and the configured action space of the environment
	/// @param latent_pi The latent input, either logits or probability based on what was configured.
	/// @return An abstracted distribution which can be used to sample an action.
	std::unique_ptr<Distribution> forward(torch::Tensor latent_pi);
	/// @brief Clones the PolicyActionOutputImpl, creating a deep copy.
	/// @param device An optional device parameter to use for the cloned block.
	/// @return A shared pointer to the cloned PolicyActionOutputImpl.
	std::shared_ptr<torch::nn::Module> clone(const c10::optional<torch::Device>& device = c10::nullopt) const override;

private:
	const Config::PolicyActionOutputConfig config_;
	ActionSpace action_space_;
	int num_actions_;
	bool use_logits_;

	torch::nn::Linear action_net_;
	torch::nn::Linear log_std_;
};

TORCH_MODULE(PolicyActionOutput);

} // namespace drla
