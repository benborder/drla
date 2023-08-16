#include "actor_net.h"

#include "bernoulli.h"
#include "categorical.h"
#include "diagonal_gaussian.h"
#include "model/utils.h"

#include <spdlog/spdlog.h>
#include <torch/torch.h>

#include <memory>

using namespace torch;
using namespace drla;

ActorImpl::ActorImpl(const Config::ActorConfig& config, int inputs, const ActionSpace& action_space, bool use_logits)
		: config_(config)
		, action_space_(action_space)
		, num_actions_(std::accumulate(action_space_.shape.begin(), action_space_.shape.end(), 0))
		, output_size_(action_space_.type == ActionSpaceType::kBox ? 2 * num_actions_ : num_actions_)
		, use_logits_(use_logits)
		, mlp_(
				config_.mlp,
				"actor",
				inputs,
				output_size_,
				Config::LinearConfig{
					output_size_,
					Config::Activation::kNone,
					config.init_weight_type,
					config.init_weight,
					config.init_bias_type,
					config.init_bias})
{
	register_module("mlp", mlp_);
}

ActorImpl::ActorImpl(const ActorImpl& other, const c10::optional<torch::Device>& device)
		: config_(other.config_)
		, action_space_(other.action_space_)
		, num_actions_(other.num_actions_)
		, output_size_(other.output_size_)
		, use_logits_(other.use_logits_)
		, mlp_(std::dynamic_pointer_cast<FCBlockImpl>(other.mlp_->clone(device)))
{
	register_module("mlp", mlp_);
}

std::unique_ptr<Distribution> ActorImpl::forward(torch::Tensor latent)
{
	auto output = mlp_(latent);

	torch::Tensor probs;
	torch::Tensor logits;
	if (use_logits_)
	{
		logits = output;
	}
	else
	{
		probs = output;
	}
	switch (action_space_.type)
	{
		case ActionSpaceType::kDiscrete:
		{
			return std::make_unique<Categorical>(probs, logits);
		}
		case ActionSpaceType::kBox:
		{
			auto sz = output.sizes().vec();
			sz.back() = 2;
			sz.push_back(-1);
			output = output.view(sz);
			return std::make_unique<DiagonalGaussian>(output.narrow(-2, 0, 1), output.narrow(-2, 1, 1));
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

std::shared_ptr<torch::nn::Module> ActorImpl::clone(const c10::optional<torch::Device>& device) const
{
	torch::NoGradGuard no_grad;
	return std::make_shared<ActorImpl>(static_cast<const ActorImpl&>(*this), device);
}
