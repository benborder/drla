#include "policy_action_output.h"

#include "bernoulli.h"
#include "categorical.h"
#include "diagonal_gaussian.h"
#include "model/utils.h"

#include <spdlog/spdlog.h>
#include <torch/torch.h>

#include <memory>

using namespace torch;
using namespace drla;

PolicyActionOutputImpl::PolicyActionOutputImpl(
	const Config::PolicyActionOutputConfig& config, int inputs, const ActionSpace& action_space, bool use_logits)
		: config_(config)
		, action_space_(action_space)
		, num_actions_(std::accumulate(action_space_.shape.begin(), action_space_.shape.end(), 0))
		, use_logits_(use_logits)
		, action_net_(inputs, num_actions_)
		, log_std_(nullptr)
{
	register_module("action_net_output", action_net_);
	if (action_space_.type == ActionSpaceType::kBox)
	{
		log_std_ = torch::nn::Linear(inputs, num_actions_);
		register_module("log_std", log_std_);
		weight_init(log_std_->weight, config_.init_weight_type, config_.init_weight);
		weight_init(log_std_->bias, config_.init_bias_type, config_.init_bias);
	}
	weight_init(action_net_->weight, config_.init_weight_type, config_.init_weight);
	weight_init(action_net_->bias, config_.init_bias_type, config_.init_bias);

	spdlog::debug("Output:  {}", num_actions_);
}

PolicyActionOutputImpl::PolicyActionOutputImpl(
	const PolicyActionOutputImpl& other, const c10::optional<torch::Device>& device)
		: config_(other.config_)
		, action_space_(other.action_space_)
		, num_actions_(other.num_actions_)
		, use_logits_(other.use_logits_)
		, action_net_(std::dynamic_pointer_cast<torch::nn::LinearImpl>(other.action_net_->clone(device)))
		, log_std_(nullptr)
{
	register_module("action_net_output", action_net_);
	if (action_space_.type == ActionSpaceType::kBox)
	{
		log_std_ = std::dynamic_pointer_cast<torch::nn::LinearImpl>(other.log_std_->clone(device));
		register_module("log_std", log_std_);
	}
}

std::unique_ptr<Distribution> PolicyActionOutputImpl::forward(torch::Tensor latent_pi)
{
	auto latent_actions = activation(action_net_(latent_pi), config_.activation);

	std::optional<torch::Tensor> probs;
	std::optional<torch::Tensor> logits;
	if (use_logits_)
	{
		logits = {latent_actions};
	}
	else
	{
		probs = {latent_actions};
	}
	switch (action_space_.type)
	{
		case ActionSpaceType::kDiscrete:
		{
			return std::make_unique<Categorical>(probs, logits);
		}
		case ActionSpaceType::kBox:
		{
			return std::make_unique<DiagonalGaussian>(latent_actions, log_std_(latent_pi));
		}
		case ActionSpaceType::kMultiBinary:
		{
			return std::make_unique<Bernoulli>(probs, logits);
		}
		case ActionSpaceType::kMultiDiscrete:
		{
			return std::make_unique<MultiCategorical>(action_space_.shape, probs, logits);
		}
		default:
		{
			throw std::runtime_error(
				"Invalid action space type for action policy output layer: " +
				std::to_string(static_cast<int>(action_space_.type)));
		}
	}
}

std::shared_ptr<torch::nn::Module> PolicyActionOutputImpl::clone(const c10::optional<torch::Device>& device) const
{
	torch::NoGradGuard no_grad;
	return std::make_shared<PolicyActionOutputImpl>(static_cast<const PolicyActionOutputImpl&>(*this), device);
}
