#include "actor_critic_model.h"

#include "distribution.h"
#include "model/utils.h"

#include <spdlog/spdlog.h>

#include <filesystem>

using namespace drla;
using namespace torch;

ActorCriticModel::ActorCriticModel(
	const Config::ModelConfig& config, const EnvironmentConfiguration& env_config, int value_shape, bool predict_values)
		: config_(std::get<Config::ActorCriticConfig>(config))
		, predict_values_(config_.predict_values || predict_values)
		, action_space_(env_config.action_space)
		, feature_extractor_(config_.feature_extractor, env_config.observation_shapes)
		, feature_extractor_critic_(nullptr)
		, shared_(nullptr)
		, critic_(nullptr)
		, actor_(nullptr)
		, policy_action_output_(nullptr)
{
	register_module("feature_extractor", feature_extractor_);
	int input_size = feature_extractor_->get_output_size();
	if (config_.use_shared_extractor)
	{
		shared_ = register_module(config_.shared.name, FCBlock(config_.shared, input_size));
		input_size = shared_->get_output_size();
	}
	else
	{
		feature_extractor_critic_ = register_module(
			"feature_extractor_critic", FeatureExtractor(config_.feature_extractor, env_config.observation_shapes));
	}

	critic_ = register_module(config_.critic.name, FCBlock(config_.critic, input_size, value_shape));
	actor_ = register_module(config_.actor.name, FCBlock(config_.actor, input_size));
	if (config_.actor.layers.empty())
	{
		spdlog::debug("Constructing {}", config_.actor.name);
	}
	policy_action_output_ = register_module(
		"policy_action_output",
		PolicyActionOutput(config_.policy_action_output, actor_->get_output_size(), env_config.action_space));

	int parameter_size = 0;
	auto params = parameters();
	for (auto& p : params)
	{
		if (p.requires_grad())
		{
			parameter_size += p.numel();
		}
	}
	spdlog::debug("Total parameters: {}", parameter_size);
}

PredictOutput ActorCriticModel::predict(const Observations& observations, bool deterministic)
{
	auto features = flatten(feature_extractor_(observations));
	torch::Tensor hidden;
	torch::Tensor values;
	if (config_.use_shared_extractor)
	{
		hidden = shared_(features);
		if (predict_values_)
		{
			values = critic_(hidden);
		}
	}
	else
	{
		hidden = features;
		if (predict_values_)
		{
			values = critic_(flatten(feature_extractor_critic_(observations)));
		}
	}
	auto dist = policy_action_output_(actor_(hidden));
	auto action = dist->sample(deterministic);

	if (deterministic)
	{
		return {values, action, {}};
	}
	else
	{
		auto action_log_probs = dist->action_log_prob(action);

		if (is_action_discrete(action_space_))
		{
			action = action.unsqueeze(-1);
			action_log_probs = action_log_probs.unsqueeze(-1);
		}
		else
		{
			action_log_probs = action_log_probs.sum(-1, true);
		}

		return {values, action, action_log_probs};
	}
}

ActionPolicyEvaluation
ActorCriticModel::evaluate_actions(const Observations& observations, const torch::Tensor& actions)
{
	auto features = flatten(feature_extractor_(observations));
	torch::Tensor hidden;
	torch::Tensor values;
	if (config_.use_shared_extractor)
	{
		hidden = shared_(features);
		values = critic_(hidden);
	}
	else
	{
		hidden = features;
		values = critic_(flatten(feature_extractor_critic_(observations)));
	}
	auto dist = policy_action_output_(actor_(hidden));

	torch::Tensor action_log_probs;
	if (is_action_discrete(action_space_))
	{
		action_log_probs = dist->action_log_prob(actions.squeeze(-1)).view({actions.size(0), -1}).sum(-1).unsqueeze(-1);
	}
	else
	{
		action_log_probs = dist->action_log_prob(actions).sum(-1, true);
	}

	return {values, action_log_probs, dist->entropy().mean()};
}

void ActorCriticModel::save(const std::filesystem::path& path)
{
	torch::save(std::dynamic_pointer_cast<torch::nn::Module>(shared_from_this()), path / "model.pt");
}

void ActorCriticModel::load(const std::filesystem::path& path)
{
	auto model_path = path / "model.pt";
	if (std::filesystem::exists(model_path))
	{
		auto model = std::dynamic_pointer_cast<torch::nn::Module>(shared_from_this());
		torch::load(model, model_path);
		spdlog::debug("Actor Critic model loaded");
	}
}
