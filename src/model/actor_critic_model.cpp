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
		, use_gru_(config_.gru_hidden_size > 0)
		, action_space_(env_config.action_space)
		, feature_extractor_(config_.feature_extractor, env_config.observation_shapes)
		, feature_extractor_critic_(nullptr)
		, shared_(nullptr)
		, critic_(nullptr)
		, actor_(nullptr)
		, grucell_(nullptr)
{
	register_module("feature_extractor", feature_extractor_);
	int input_size = feature_extractor_->get_output_size();
	if (config_.use_shared_extractor || use_gru_)
	{
		shared_ = register_module("shared", FCBlock(config_.shared, "shared", input_size));
		input_size = shared_->get_output_size();
	}
	else
	{
		feature_extractor_critic_ = register_module(
			"feature_extractor_critic", FeatureExtractor(config_.feature_extractor, env_config.observation_shapes));
	}

	if (use_gru_)
	{
		grucell_ =
			register_module("grucell", torch::nn::GRUCell(torch::nn::GRUCellOptions(input_size, config_.gru_hidden_size)));
		input_size = config_.gru_hidden_size;
	}

	critic_ = register_module("critic", FCBlock(config_.critic, "critic", input_size, Config::LinearConfig{value_shape}));
	actor_ = register_module("actor", Actor(config_.actor, input_size, env_config.action_space));

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

ActorCriticModel::ActorCriticModel(const ActorCriticModel& other, const c10::optional<torch::Device>& device)
		: config_(other.config_)
		, predict_values_(other.predict_values_)
		, use_gru_(other.use_gru_)
		, action_space_(other.action_space_)
		, feature_extractor_(std::dynamic_pointer_cast<FeatureExtractorImpl>(other.feature_extractor_->clone(device)))
		, feature_extractor_critic_(nullptr)
		, shared_(nullptr)
		, critic_(std::dynamic_pointer_cast<FCBlockImpl>(other.critic_->clone(device)))
		, actor_(std::dynamic_pointer_cast<ActorImpl>(other.actor_->clone(device)))
		, grucell_(use_gru_ ? std::dynamic_pointer_cast<torch::nn::GRUCellImpl>(other.grucell_->clone(device)) : nullptr)
{
	register_module("feature_extractor", feature_extractor_);

	if (config_.use_shared_extractor)
	{
		shared_ = std::dynamic_pointer_cast<FCBlockImpl>(other.shared_->clone(device));
		register_module("shared", shared_);
	}
	else
	{
		feature_extractor_critic_ =
			std::dynamic_pointer_cast<FeatureExtractorImpl>(other.feature_extractor_critic_->clone(device));
		register_module("feature_extractor_critic", feature_extractor_critic_);
	}
	if (use_gru_)
	{
		register_module("grucell", grucell_);
	}

	register_module("critic", critic_);
	register_module("actor", actor_);
}

ModelOutput ActorCriticModel::predict(const ModelInput& input)
{
	torch::Tensor features = flatten(feature_extractor_(input.observations));
	ModelOutput output;
	if (config_.use_shared_extractor || use_gru_)
	{
		features = shared_(features);
		// GRU can only be used with shared extractors
		if (use_gru_)
		{
			features = grucell_(features, input.prev_output.state[0]);
			output.state = {features};
		}
		if (predict_values_)
		{
			output.values = critic_(features);
		}
	}
	else
	{
		if (predict_values_)
		{
			output.values = critic_(flatten(feature_extractor_critic_(input.observations)));
		}
	}
	auto dist = actor_(features);
	if (input.deterministic)
	{
		output.action = dist->mode();
		return output;
	}
	else
	{
		output.action = dist->sample();
		output.action_log_probs = dist->log_prob(output.action);

		if (is_action_discrete(action_space_))
		{
			output.action.unsqueeze_(-1);
			output.action_log_probs.unsqueeze_(-1);
		}
		else
		{
			output.action_log_probs = output.action_log_probs.sum(-1, true);
		}

		return output;
	}
}

ModelOutput ActorCriticModel::initial() const
{
	ModelOutput output;
	auto device = parameters().front().device();
	if (is_action_discrete(action_space_))
	{
		output.action = torch::zeros(static_cast<int>(action_space_.shape.size()));
	}
	else
	{
		output.action = torch::zeros(action_space_.shape);
	}
	output.values = torch::zeros(critic_->get_output_size(), device);
	if (use_gru_)
	{
		output.state = {torch::zeros({1, config_.gru_hidden_size}, device)};
	}
	return output;
}

StateShapes ActorCriticModel::get_state_shape() const
{
	if (use_gru_)
		return {config_.gru_hidden_size};
	else
		return {};
}

ActionPolicyEvaluation ActorCriticModel::evaluate_actions(
	const Observations& observations, const torch::Tensor& actions, const HiddenStates& states)
{
	torch::Tensor features = flatten(feature_extractor_(observations));
	torch::Tensor values;
	if (config_.use_shared_extractor || use_gru_)
	{
		features = shared_(features);
		// GRU can only be used with shared extractors
		if (use_gru_)
		{
			features = grucell_(features, states[0]);
		}
		values = critic_(features);
	}
	else
	{
		values = critic_(flatten(feature_extractor_critic_(observations)));
	}
	auto dist = actor_(features);

	torch::Tensor action_log_probs;
	if (is_action_discrete(action_space_))
	{
		action_log_probs = dist->log_prob(actions.squeeze(-1)).view({actions.size(0), -1}).sum(-1).unsqueeze(-1);
	}
	else
	{
		action_log_probs = dist->log_prob(actions).sum(-1, true);
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

std::shared_ptr<torch::nn::Module> ActorCriticModel::clone(const c10::optional<torch::Device>& device) const
{
	torch::NoGradGuard no_grad;
	return std::make_shared<ActorCriticModel>(static_cast<const ActorCriticModel&>(*this), device);
}
