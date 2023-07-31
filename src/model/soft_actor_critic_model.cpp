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
		, use_gru_(config_.gru_hidden_size > 0)
		, feature_extractor_actor_(config_.feature_extractor, env_config.observation_shapes)
		, grucell_(nullptr)
		, actor_(nullptr)
{
	register_module("feature_extractor_actor", feature_extractor_actor_);
	int input_size = feature_extractor_actor_->get_output_size();
	if (use_gru_)
	{
		grucell_ =
			register_module("grucell", torch::nn::GRUCell(torch::nn::GRUCellOptions(input_size, config_.gru_hidden_size)));
		input_size = config_.gru_hidden_size;
	}
	actor_ = register_module("actor", Actor(config_.actor, input_size, action_space_, true));

	auto actions = std::accumulate(action_space_.shape.begin(), action_space_.shape.end(), 0);
	auto make_critic = [&](std::string postfix) {
		CriticModules cm;
		int cm_input_size;
		if (config_.shared_feature_extractor)
		{
			cm.feature_extractor_ = feature_extractor_actor_;
			cm_input_size = cm.feature_extractor_->get_output_size();
		}
		else
		{
			cm.feature_extractor_ = register_module(
				"feature_extractor_critic_" + postfix,
				FeatureExtractor(config_.feature_extractor, env_config.observation_shapes));
			cm_input_size = cm.feature_extractor_->get_output_size();
			if (use_gru_)
			{
				cm.grucell_ = register_module(
					"grucell" + postfix, torch::nn::GRUCell(torch::nn::GRUCellOptions(cm_input_size, config_.gru_hidden_size)));
				cm_input_size = config_.gru_hidden_size;
			}
		}

		int critic_output = actions * value_shape_;
		if (!is_action_discrete(action_space_))
		{
			cm_input_size += actions;
			critic_output = value_shape_;
		}
		auto name = "critic_" + postfix;
		cm.critic_ = register_module(name, FCBlock(config_.critic, name, cm_input_size, critic_output));
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
		, use_gru_(other.use_gru_)
		, feature_extractor_actor_(
				std::dynamic_pointer_cast<FeatureExtractorImpl>(other.feature_extractor_actor_->clone(device)))
		, grucell_(use_gru_ ? std::dynamic_pointer_cast<torch::nn::GRUCellImpl>(other.grucell_->clone(device)) : nullptr)
		, actor_(std::dynamic_pointer_cast<ActorImpl>(other.actor_->clone(device)))
{
	register_module("feature_extractor_actor", feature_extractor_actor_);
	register_module("actor", actor_);
	if (use_gru_)
	{
		register_module("grucell", grucell_);
	}

	auto make_critic = [&](const CriticModules& other, std::string postfix) {
		CriticModules cm;
		if (config_.shared_feature_extractor)
		{
			cm.feature_extractor_ = feature_extractor_actor_;
		}
		else
		{
			cm.feature_extractor_ = std::dynamic_pointer_cast<FeatureExtractorImpl>(other.feature_extractor_->clone(device));
			register_module("feature_extractor_actor" + postfix, cm.feature_extractor_);
			if (use_gru_)
			{
				cm.grucell_ = std::dynamic_pointer_cast<torch::nn::GRUCellImpl>(other.grucell_->clone(device));
				register_module("grucell" + postfix, cm.grucell_);
			}
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
}

ModelOutput SoftActorCriticModel::predict(const ModelInput& input)
{
	torch::Tensor features = flatten(feature_extractor_actor_(input.observations));
	ModelOutput output;
	if (use_gru_)
	{
		features = grucell_(features, input.prev_output.state.at(0));
		output.state = {features};
	}
	auto dist = actor_(features);

	if (input.deterministic)
	{
		output.action = dist->mode();
	}
	else
	{
		output.action = dist->sample();
	}

	if (is_action_discrete(action_space_))
	{
		output.action.unsqueeze_(-1);
	}

	if (config_.predict_values || is_training())
	{
		torch::NoGradGuard no_grad_guard;
		auto [qvalues, state] =
			critic_values(critics_, input.observations, dist->get_action_output(), features, input.prev_output.state);
		if (use_gru_)
		{
			output.state.insert(output.state.end(), state.begin(), state.end());
		}

		output.values = std::get<0>(torch::min(torch::stack(qvalues), 0));
		// Select the value based on the action (rather than returning all qvalues)
		output.values = output.values.gather(1, output.action.to(torch::kLong));
	}
	else
	{
		output.values = torch::zeros({output.action.size(0), value_shape_});
	}

	return output;
}

ModelOutput SoftActorCriticModel::initial() const
{
	ModelOutput output;
	auto device = actor_->parameters().front().device();
	if (is_action_discrete(action_space_))
	{
		output.action = torch::zeros(static_cast<int>(action_space_.shape.size()));
	}
	else
	{
		output.action = torch::zeros(action_space_.shape);
	}
	output.values = torch::zeros(value_shape_, device);
	if (use_gru_)
	{
		output.state = {torch::zeros(config_.gru_hidden_size, device)};
		for (size_t i = 0; i < critics_.size(); ++i)
		{
			output.state.push_back(torch::zeros(config_.gru_hidden_size, device));
		}
	}
	return output;
}

StateShapes SoftActorCriticModel::get_state_shape() const
{
	if (use_gru_)
	{
		StateShapes shape = {config_.gru_hidden_size};
		for (size_t i = 0; i < critics_.size(); ++i) { shape.push_back(config_.gru_hidden_size); }
		return shape;
	}
	else
	{
		return {};
	}
}

ActorOutput SoftActorCriticModel::action_output(const Observations& observations, const HiddenStates& state)
{
	torch::Tensor features = flatten(feature_extractor_actor_(observations));
	if (grucell_)
	{
		features = grucell_(features, state.at(0));
	}
	auto dist = actor_(features);
	ActorOutput output;
	output.action = dist->sample();
	output.state = {features};
	torch::Tensor log_probs;
	if (is_action_discrete(action_space_))
	{
		auto logits = dist->get_action_output();
		output.log_prob = torch::log_softmax(logits, -1);
	}
	else
	{
		output.log_prob = dist->log_prob(output.action).sum(-1, true);
	}
	output.action.unsqueeze_(-1);
	output.actions_pi = dist->get_action_output();
	return output;
}

std::vector<torch::Tensor>
SoftActorCriticModel::critic(const Observations& observations, const torch::Tensor& actions, const HiddenStates& state)
{
	torch::Tensor features;
	if (config_.shared_feature_extractor)
	{
		torch::NoGradGuard no_grad_guard;
		features = flatten(critics_.front().feature_extractor_(observations));
		if (use_gru_)
		{
			features = grucell_(features, state.at(0));
		}
	}
	auto [qvalues, _] = critic_values(critics_, observations, actions, features, state);
	return qvalues;
}

std::vector<torch::Tensor> SoftActorCriticModel::critic_target(
	const Observations& observations, const torch::Tensor& actions, const HiddenStates& state)
{
	torch::NoGradGuard no_grad_guard;
	torch::Tensor features;
	if (config_.shared_feature_extractor)
	{
		features = flatten(critic_targets_.front().feature_extractor_(observations));
		if (use_gru_)
		{
			features = grucell_(features, state.at(0));
		}
	}
	auto [qvalues, _] = critic_values(critics_, observations, actions, features, state);
	return qvalues;
}

std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>> SoftActorCriticModel::critic_values(
	std::vector<CriticModules>& critics,
	const Observations& observations,
	const torch::Tensor& actions,
	const torch::Tensor& features,
	const HiddenStates& state)
{
	std::vector<torch::Tensor> qvalues;
	std::vector<torch::Tensor> output_state;
	for (size_t c = 0, len = critics.size(); c < len; ++c)
	{
		auto& critic_module = critics[c];

		torch::Tensor q_value_input = features;
		if (!config_.shared_feature_extractor)
		{
			q_value_input = flatten(critic_module.feature_extractor_(observations));
			if (use_gru_)
			{
				q_value_input = critic_module.grucell_(q_value_input, state.at(c + 1));
				output_state.push_back(q_value_input);
			}
		}
		if (!is_action_discrete(action_space_))
		{
			q_value_input = torch::cat({q_value_input, actions}, 1);
		}
		qvalues.push_back(critic_module.critic_(q_value_input));
	}
	return {qvalues, output_state};
}

inline void
update_params(const std::vector<torch::Tensor>& current, const std::vector<torch::Tensor>& target, double tau)
{
	for (size_t i = 0; i < current.size(); i++) { target[i].mul_(1.0 - tau).add_(current[i], tau); }
}

void SoftActorCriticModel::update(double tau)
{
	torch::NoGradGuard no_grad;
	for (size_t i = 0; i < config_.n_critics; ++i)
	{
		auto& current = critics_[i];
		auto& target = critic_targets_[i];
		update_params(current.critic_->parameters(), target.critic_->parameters(), tau);
		if (!config_.shared_feature_extractor)
		{
			update_params(current.feature_extractor_->parameters(), target.feature_extractor_->parameters(), tau);
			if (use_gru_)
			{
				update_params(current.grucell_->parameters(), target.grucell_->parameters(), tau);
			}
		}
	}
}

void SoftActorCriticModel::train(bool train)
{
	feature_extractor_actor_->train(train);
	actor_->train(train);
	if (use_gru_)
	{
		grucell_->train(train);
	}
	for (auto& critic : critics_)
	{
		if (!config_.shared_feature_extractor)
		{
			critic.feature_extractor_->train(train);
			if (use_gru_)
			{
				critic.grucell_->train(train);
			}
		}
		critic.critic_->train(train);
	}
	// Training should never be enabled for the target
	for (auto& critic : critic_targets_)
	{
		if (!config_.shared_feature_extractor)
		{
			critic.feature_extractor_->train(false);
			if (use_gru_)
			{
				critic.grucell_->train(false);
			}
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
	if (use_gru_)
	{
		const auto grucell_params = grucell_->parameters(recursive);
		params.insert(params.end(), grucell_params.begin(), grucell_params.end());
	}
	const auto actor_params = actor_->parameters(recursive);
	params.insert(params.end(), actor_params.begin(), actor_params.end());

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
			if (use_gru_)
			{
				const auto grucell_critic_params = critic.grucell_->parameters(recursive);
				params.insert(params.end(), grucell_critic_params.begin(), grucell_critic_params.end());
			}
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
