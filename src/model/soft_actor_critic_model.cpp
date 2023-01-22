#include "soft_actor_critic_model.h"

#include "distribution.h"
#include "utils.h"

#include <spdlog/spdlog.h>

#include <filesystem>

using namespace drla;
using namespace torch;

SoftActorCriticModel::SoftActorCriticModel(
	const Config::ModelConfig& config, const EnvironmentConfiguration& env_config, int value_shape)
		: config_(std::get<Config::SoftActorCriticConfig>(config))
		, value_shape_(value_shape)
		, action_space_(env_config.action_space)
		, feature_extractor_actor_(config_.feature_extractor, env_config.observation_shapes)
		, actor_(nullptr)
		, policy_action_output_(nullptr)
{
	register_module("feature_extractor_actor", feature_extractor_actor_);
	actor_ = register_module("actor", FCBlock(config_.actor, "actor", feature_extractor_actor_->get_output_size()));

	policy_action_output_ = register_module(
		"policy_action_output",
		PolicyActionOutput(config_.policy_action_output, actor_->get_output_size(), action_space_, true));

	auto actions = std::accumulate(action_space_.shape.begin(), action_space_.shape.end(), 0);
	auto make_critic = [&](std::string postfix) {
		CriticModules cm{nullptr, nullptr};
		if (config_.shared_feature_extractor)
		{
			cm.feature_extractor_ = feature_extractor_actor_;
		}
		else
		{
			cm.feature_extractor_ = register_module(
				"feature_extractor_critic_" + postfix,
				FeatureExtractor(config_.feature_extractor, env_config.observation_shapes));
		}

		int critic_input = cm.feature_extractor_->get_output_size();
		int critic_output = actions * value_shape_;
		if (!is_action_discrete(action_space_))
		{
			critic_input += actions;
			critic_output = value_shape_;
		}
		auto name = "critic_" + postfix;
		cm.critic_ = register_module(name, FCBlock(config_.critic, name, critic_input, critic_output));
		return cm;
	};
	for (size_t i = 0; i < config_.n_critics; ++i)
	{
		critics_.push_back(make_critic(std::to_string(i)));
		critic_targets_.push_back(make_critic("target_" + std::to_string(i)));
	}

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
	// make the target initially equal the critic
	update(1.0);
}

SoftActorCriticModel::SoftActorCriticModel(
	const SoftActorCriticModel& other, const c10::optional<torch::Device>& device)
		: config_(other.config_)
		, value_shape_(other.value_shape_)
		, action_space_(other.action_space_)
		, feature_extractor_actor_(nullptr)
		, actor_(nullptr)
		, policy_action_output_(nullptr)
{
	feature_extractor_actor_ =
		std::dynamic_pointer_cast<FeatureExtractorImpl>(other.feature_extractor_actor_->clone(device));
	register_module("feature_extractor_actor", feature_extractor_actor_);

	policy_action_output_ = std::dynamic_pointer_cast<PolicyActionOutputImpl>(other.policy_action_output_->clone(device));
	register_module("policy_action_output", policy_action_output_);

	auto make_critic = [&](const CriticModules& other, std::string postfix) {
		CriticModules cm{nullptr, nullptr};
		if (config_.shared_feature_extractor)
		{
			cm.feature_extractor_ = feature_extractor_actor_;
		}
		else
		{
			cm.feature_extractor_ = std::dynamic_pointer_cast<FeatureExtractorImpl>(other.feature_extractor_->clone(device));
			register_module("feature_extractor_actor" + postfix, cm.feature_extractor_);
		}

		cm.critic_ = std::dynamic_pointer_cast<FCBlockImpl>(other.critic_->clone(device));
		register_module("critic" + postfix, cm.critic_);

		return cm;
	};

	for (size_t i = 0; i < config_.n_critics; ++i)
	{
		critics_.push_back(make_critic(other.critics_[i], "critic_" + std::to_string(i)));
		critic_targets_.push_back(make_critic(other.critic_targets_[i], "critic_target_" + std::to_string(i)));
	}

	actor_ = std::dynamic_pointer_cast<FCBlockImpl>(other.actor_->clone(device));
	register_module("actor", actor_);
}

PredictOutput SoftActorCriticModel::predict(const Observations& observations, bool deterministic)
{
	torch::Tensor features = flatten(feature_extractor_actor_(observations));
	auto latent_pi = actor_(features);
	auto dist = policy_action_output_(latent_pi);
	auto action = dist->sample();

	if (is_action_discrete(action_space_))
	{
		action = action.unsqueeze(-1);
	}

	torch::Tensor value;
	if (config_.predict_values)
	{
		std::vector<torch::Tensor> qvalues;
		for (auto& critic_module : critics_)
		{
			torch::NoGradGuard no_grad_guard;
			torch::Tensor q_value_input = features;
			if (!config_.shared_feature_extractor)
			{
				q_value_input = flatten(critic_module.feature_extractor_(observations));
			}
			if (!is_action_discrete(action_space_))
			{
				q_value_input = torch::cat({q_value_input, dist->get_action_output()}, 1);
			}
			qvalues.push_back(critic_module.critic_(q_value_input));
		}

		value = std::get<0>(torch::min(torch::stack(qvalues), 0));
		// Select the value based on the action (rather than returning all qvalues)
		value = value.gather(1, action.to(torch::kLong));
	}
	else
	{
		value = torch::zeros({action.size(0), value_shape_});
	}

	return {action, value};
}

ActorOutput SoftActorCriticModel::action_output(const Observations& observations)
{
	torch::Tensor features = flatten(feature_extractor_actor_(observations));
	auto latent_pi = actor_(features);
	auto dist = policy_action_output_(latent_pi);
	auto action = dist->sample();
	torch::Tensor log_probs;
	if (is_action_discrete(action_space_))
	{
		auto logits = dist->get_action_output();
		log_probs = torch::log_softmax(logits, -1);
	}
	else
	{
		log_probs = dist->action_log_prob(action).sum(-1, true);
	}
	return {action.unsqueeze(-1), dist->get_action_output(), log_probs};
}

std::vector<torch::Tensor> SoftActorCriticModel::critic(const Observations& observations, const torch::Tensor& actions)
{
	std::vector<torch::Tensor> qvalues;
	torch::Tensor features;
	if (config_.shared_feature_extractor)
	{
		torch::NoGradGuard no_grad_guard;
		features = flatten(critics_.front().feature_extractor_(observations));
	}
	for (auto& critic : critics_)
	{
		if (!config_.shared_feature_extractor)
		{
			features = flatten(critic.feature_extractor_(observations));
		}
		torch::Tensor q_value_input;
		if (is_action_discrete(action_space_))
		{
			q_value_input = features;
		}
		else
		{
			q_value_input = torch::cat({features, actions}, 1);
		}

		qvalues.push_back(critic.critic_(q_value_input));
	}
	return qvalues;
}

std::vector<torch::Tensor>
SoftActorCriticModel::critic_target(const Observations& observations, const torch::Tensor& actions)
{
	torch::NoGradGuard no_grad_guard;
	std::vector<torch::Tensor> qvalues;
	torch::Tensor features;
	if (config_.shared_feature_extractor)
	{
		features = flatten(critic_targets_.front().feature_extractor_(observations));
	}
	for (auto& critic : critic_targets_)
	{
		if (!config_.shared_feature_extractor)
		{
			features = flatten(critic.feature_extractor_(observations));
		}
		torch::Tensor q_value_input;
		if (is_action_discrete(action_space_))
		{
			q_value_input = features;
		}
		else
		{
			q_value_input = torch::cat({features, actions}, 1);
		}

		qvalues.push_back(critic.critic_(q_value_input));
	}

	return qvalues;
}

void SoftActorCriticModel::update(double tau)
{
	torch::NoGradGuard no_grad;
	for (size_t i = 0; i < config_.n_critics; ++i)
	{
		const auto current_params = critics_[i].critic_->parameters();
		auto target_params = critic_targets_[i].critic_->parameters();

		for (size_t p = 0; p < current_params.size(); ++p)
		{
			target_params[p].mul_(1.0 - tau).add_(current_params[p], tau);
		}

		const auto feature_current_params = critics_[i].feature_extractor_->parameters();
		auto feature_target_params = critic_targets_[i].feature_extractor_->parameters();

		for (size_t p = 0; p < feature_current_params.size(); ++p)
		{
			feature_target_params[p].mul_(1.0 - tau).add_(feature_current_params[p], tau);
		}
	}
}

void SoftActorCriticModel::train(bool train)
{
	feature_extractor_actor_->train(train);
	actor_->train(train);
	policy_action_output_->train(train);
	for (auto& critic : critics_)
	{
		if (!config_.shared_feature_extractor)
		{
			critic.feature_extractor_->train(train);
		}
		critic.critic_->train(train);
	}
	// Training should never be enabled for the target
	for (auto& critic : critic_targets_)
	{
		if (!config_.shared_feature_extractor)
		{
			critic.feature_extractor_->train(false);
		}
		critic.critic_->train(false);
	}
}

std::vector<torch::Tensor> SoftActorCriticModel::parameters(bool recursive) const
{
	std::vector<torch::Tensor> params;

	const auto actor_params = actor_parameters(recursive);
	params.insert(params.end(), actor_params.begin(), actor_params.end());

	const auto critic_params = critic_parameters(recursive);
	params.insert(params.end(), critic_params.begin(), critic_params.end());

	return params;
}

std::vector<torch::Tensor> SoftActorCriticModel::actor_parameters(bool recursive) const
{
	std::vector<torch::Tensor> params;

	const auto feature_actor_params = feature_extractor_actor_->parameters(recursive);
	params.insert(params.end(), feature_actor_params.begin(), feature_actor_params.end());
	const auto actor_params = actor_->parameters(recursive);
	params.insert(params.end(), actor_params.begin(), actor_params.end());
	const auto policy_action_output_params = policy_action_output_->parameters(recursive);
	params.insert(params.end(), policy_action_output_params.begin(), policy_action_output_params.end());

	return params;
}

std::vector<torch::Tensor> SoftActorCriticModel::critic_parameters(bool recursive) const
{
	std::vector<torch::Tensor> params;

	for (auto& critic : critics_)
	{
		if (!config_.shared_feature_extractor)
		{
			const auto feature_critic_params = critic.feature_extractor_->parameters(recursive);
			params.insert(params.end(), feature_critic_params.begin(), feature_critic_params.end());
		}
		const auto critic_params = critic.critic_->parameters(recursive);
		params.insert(params.end(), critic_params.begin(), critic_params.end());
	}

	return params;
}

void SoftActorCriticModel::save(const std::filesystem::path& path)
{
	torch::save(std::dynamic_pointer_cast<torch::nn::Module>(shared_from_this()), path / "model.pt");
}

void SoftActorCriticModel::load(const std::filesystem::path& path)
{
	auto model_path = path / "model.pt";
	if (std::filesystem::exists(model_path))
	{
		auto model = std::dynamic_pointer_cast<torch::nn::Module>(shared_from_this());
		torch::load(model, model_path);
		spdlog::debug("Soft Actor Critic model loaded");
	}
}

std::shared_ptr<torch::nn::Module> SoftActorCriticModel::clone(const c10::optional<torch::Device>& device) const
{
	torch::NoGradGuard no_grad;
	return std::make_shared<SoftActorCriticModel>(static_cast<const SoftActorCriticModel&>(*this), device);
}
