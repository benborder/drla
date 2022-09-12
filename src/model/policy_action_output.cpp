#include "policy_action_output.h"

#include "bernoulli.h"
#include "categorical.h"
#include "diagonal_gaussian.h"

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
		torch::nn::init::orthogonal_(log_std_->weight, config_.init_weight);
		torch::nn::init::constant_(log_std_->bias, config_.init_bias);
	}
	torch::nn::init::orthogonal_(action_net_->weight, config_.init_weight);
	torch::nn::init::constant_(action_net_->bias, config_.init_bias);

	spdlog::debug("Output:  {}", num_actions_);
}

std::unique_ptr<Distribution> PolicyActionOutputImpl::forward(torch::Tensor latent_pi)
{
	auto latent_actions = action_net_(latent_pi);
	switch (config_.activation)
	{
		case Config::Activation::kNone:
		{
			break;
		}
		case Config::Activation::kReLU:
		{
			latent_actions = torch::relu(latent_actions);
			break;
		}
		case Config::Activation::kLeakyReLU:
		{
			latent_actions = torch::leaky_relu(latent_actions);
			break;
		}
		case Config::Activation::kTanh:
		{
			latent_actions = torch::tanh(latent_actions);
			break;
		}
		case Config::Activation::kSigmoid:
		{
			latent_actions = torch::sigmoid(latent_actions);
			break;
		}
		case Config::Activation::kELU:
		{
			latent_actions = torch::elu(latent_actions);
			break;
		}
		case Config::Activation::kSoftplus:
		{
			latent_actions = torch::softplus(latent_actions);
			break;
		}
	}
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
