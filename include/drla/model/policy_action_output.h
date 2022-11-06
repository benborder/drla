#pragma once

#include "drla/configuration/model.h"
#include "drla/types.h"

#include <torch/torch.h>

#include <memory>

// Maybe rename file to policy action output
namespace drla
{

class Distribution;

class PolicyActionOutputImpl : public torch::nn::Module
{
public:
	PolicyActionOutputImpl(
		const Config::PolicyActionOutputConfig& config,
		int inputs,
		const ActionSpace& action_space,
		bool use_logits = true);
	PolicyActionOutputImpl(const PolicyActionOutputImpl& other, const c10::optional<torch::Device>& device);

	std::unique_ptr<Distribution> forward(torch::Tensor latent_pi);
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
