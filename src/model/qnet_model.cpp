#include "qnet_model.h"

#include <spdlog/spdlog.h>

#include <filesystem>

using namespace drla;
using namespace torch;

QNetModel::QNetModel(const Config::ModelConfig& config, const EnvironmentConfiguration& env_config, int value_shape)
		: config_(std::get<Config::QNetModelConfig>(config))
		, action_space_(env_config.action_space)
		, feature_extractor_(make_feature_extractor(config_.feature_extractor, env_config.observation_shapes))
		, feature_extractor_target_(make_feature_extractor(config_.feature_extractor, env_config.observation_shapes))
		, q_net_(nullptr)
		, q_net_target_(nullptr)
{
	if (!is_action_discrete(action_space_))
	{
		throw std::runtime_error("QNetwork model is only compatible with discrete action spaces");
	}

	register_module("feature_extractor", feature_extractor_);
	register_module("feature_extractor_target", feature_extractor_target_);

	// DQN can not handle multi actions unless they are permuted
	int action_shape = action_space_.shape[0];
	q_net_ =
		register_module(config_.q_net.name, FCBlock(config_.q_net, feature_extractor_->get_output_size(), action_shape));
	q_net_target_ = register_module(
		config_.q_net.name + "_target", FCBlock(config_.q_net, feature_extractor_->get_output_size(), action_shape));

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

torch::Tensor QNetModel::forward(const Observations& observations)
{
	auto features = feature_extractor_->forward(observations);
	return q_net_(features);
}

PredictOutput QNetModel::predict(const Observations& observations, bool deterministic)
{
	auto values = forward(observations);
	torch::Tensor action;

	if (!deterministic && torch::rand({1}).item<float>() < exploration_)
	{
		action = torch::randint(action_space_.shape.front(), {observations.front().size(0), 1});
	}
	else
	{
		action = values.argmax(1).view({-1, 1});
	}

	return {values, action, {}};
}

torch::Tensor QNetModel::forward_target(const Observations& observations)
{
	torch::Tensor features = feature_extractor_target_->forward(observations);
	return q_net_target_(features);
}

void QNetModel::update(double tau)
{
	torch::NoGradGuard no_grad;
	const auto current_params = q_net_->parameters();
	auto target_params = q_net_target_->parameters();

	for (size_t i = 0; i < current_params.size(); i++) { target_params[i].mul_(1.0 - tau).add_(current_params[i], tau); }

	const auto feature_current_params = feature_extractor_->parameters();
	auto feature_target_params = feature_extractor_target_->parameters();

	for (size_t i = 0; i < feature_current_params.size(); i++)
	{
		feature_target_params[i].mul_(1.0 - tau).add_(feature_current_params[i], tau);
	}
}

std::vector<torch::Tensor> QNetModel::parameters(bool recursive) const
{
	std::vector<torch::Tensor> params;

	const auto feature_current_params = feature_extractor_->parameters(recursive);
	const auto current_params = q_net_->parameters(recursive);
	params.insert(params.end(), feature_current_params.begin(), feature_current_params.end());
	params.insert(params.end(), current_params.begin(), current_params.end());

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
