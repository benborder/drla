#include "qnet_model.h"

#include "model/utils.h"

#include <spdlog/spdlog.h>

#include <filesystem>

using namespace drla;
using namespace torch;

QNetModel::QNetModel(const Config::ModelConfig& config, const EnvironmentConfiguration& env_config)
		: config_(std::get<Config::QNetModelConfig>(config))
		, action_space_(env_config.action_space)
		, use_gru_(config_.gru_hidden_size > 0)
		, feature_extractor_(config_.feature_extractor, env_config.observation_shapes)
		, feature_extractor_target_(config_.feature_extractor, env_config.observation_shapes)
		, grucell_(nullptr)
		, grucell_target_(nullptr)
		, q_net_(nullptr)
		, q_net_target_(nullptr)
{
	if (!is_action_discrete(action_space_))
	{
		throw std::runtime_error("QNetwork model is only compatible with discrete action spaces");
	}

	register_module("feature_extractor", feature_extractor_);
	register_module("feature_extractor_target", feature_extractor_target_);

	int input_size = feature_extractor_->get_output_size();

	if (use_gru_)
	{
		grucell_ =
			register_module("grucell", torch::nn::GRUCell(torch::nn::GRUCellOptions(input_size, config_.gru_hidden_size)));
		grucell_target_ = register_module(
			"grucell_target", torch::nn::GRUCell(torch::nn::GRUCellOptions(input_size, config_.gru_hidden_size)));
		input_size = config_.gru_hidden_size;
	}

	// DQN can not handle multi actions unless they are permuted
	int action_shape = action_space_.shape[0];
	q_net_ = register_module("q_net", FCBlock(config_.q_net, "q_net", input_size, action_shape));
	q_net_target_ = register_module("q_net_target", FCBlock(config_.q_net, "q_net", input_size, action_shape));

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

QNetModel::QNetModel(const QNetModel& other, const c10::optional<torch::Device>& device)
		: config_(other.config_)
		, action_space_(other.action_space_)
		, use_gru_(other.use_gru_)
		, feature_extractor_(std::dynamic_pointer_cast<FeatureExtractorImpl>(other.feature_extractor_->clone(device)))
		, feature_extractor_target_(
				std::dynamic_pointer_cast<FeatureExtractorImpl>(other.feature_extractor_target_->clone(device)))
		, grucell_(use_gru_ ? std::dynamic_pointer_cast<torch::nn::GRUCellImpl>(other.grucell_->clone(device)) : nullptr)
		, grucell_target_(
				use_gru_ ? std::dynamic_pointer_cast<torch::nn::GRUCellImpl>(other.grucell_target_->clone(device)) : nullptr)
		, q_net_(std::dynamic_pointer_cast<FCBlockImpl>(other.q_net_->clone(device)))
		, q_net_target_(std::dynamic_pointer_cast<FCBlockImpl>(other.q_net_target_->clone(device)))
{
	register_module("feature_extractor", feature_extractor_);
	register_module("feature_extractor_target", feature_extractor_target_);
	if (use_gru_)
	{
		register_module("grucell", grucell_);
		register_module("grucell_target", grucell_target_);
	}
	register_module("q_net", q_net_);
	register_module("q_net_target", q_net_target_);
}

torch::Tensor QNetModel::forward(const Observations& observations, const HiddenStates& state)
{
	auto features = flatten(feature_extractor_(observations));
	if (use_gru_)
	{
		features = grucell_(features, state[0]);
	}
	return q_net_(features);
}

ModelOutput QNetModel::predict(const ModelInput& input)
{
	ModelOutput output;
	auto features = flatten(feature_extractor_(input.observations));
	if (use_gru_)
	{
		features = grucell_(features, input.prev_output.state[0]);
		output.state = {features};
	}
	output.values = q_net_(features);

	if (!input.deterministic && torch::rand({1}).item<float>() < exploration_)
	{
		output.action = torch::randint(action_space_.shape.front(), {input.observations.front().size(0), 1});
	}
	else
	{
		output.action = output.values.argmax(1).view({-1, 1});
	}

	return output;
}

StateShapes QNetModel::get_state_shape() const
{
	if (use_gru_)
		return {config_.gru_hidden_size};
	else
		return {};
}

torch::Tensor QNetModel::forward_target(const Observations& observations, const HiddenStates& state)
{
	torch::Tensor features = flatten(feature_extractor_target_->forward(observations));
	if (use_gru_)
	{
		features = grucell_target_(features, state[0]);
	}
	return q_net_target_(features);
}

inline void
update_params(const std::vector<torch::Tensor>& current, const std::vector<torch::Tensor>& target, double tau)
{
	for (size_t i = 0; i < current.size(); i++) { target[i].mul_(1.0 - tau).add_(current[i], tau); }
}

void QNetModel::update(double tau)
{
	torch::NoGradGuard no_grad;
	update_params(feature_extractor_->parameters(), feature_extractor_target_->parameters(), tau);
	if (use_gru_)
	{
		update_params(grucell_->parameters(), grucell_target_->parameters(), tau);
	}
	update_params(q_net_->parameters(), q_net_target_->parameters(), tau);
}

std::vector<torch::Tensor> QNetModel::parameters(bool recursive) const
{
	std::vector<torch::Tensor> params;

	const auto feature_current_params = feature_extractor_->parameters(recursive);
	const auto current_params = q_net_->parameters(recursive);
	params.insert(params.end(), feature_current_params.begin(), feature_current_params.end());
	params.insert(params.end(), current_params.begin(), current_params.end());
	if (use_gru_)
	{
		const auto gru_current_params = grucell_->parameters(recursive);
		params.insert(params.end(), gru_current_params.begin(), gru_current_params.end());
	}

	return params;
}

void QNetModel::set_exploration(double exploration)
{
	exploration_ = exploration;
}

void QNetModel::save(const std::filesystem::path& path)
{
	torch::save(std::dynamic_pointer_cast<torch::nn::Module>(shared_from_this()), path / "model.pt");
}

void QNetModel::load(const std::filesystem::path& path)
{
	auto model_path = path / "model.pt";
	if (std::filesystem::exists(model_path))
	{
		auto model = std::dynamic_pointer_cast<torch::nn::Module>(shared_from_this());
		torch::load(model, model_path);
		spdlog::debug("DQN model loaded");
	}
}

std::shared_ptr<torch::nn::Module> QNetModel::clone(const c10::optional<torch::Device>& device) const
{
	torch::NoGradGuard no_grad;
	return std::make_shared<QNetModel>(static_cast<const QNetModel&>(*this), device);
}
