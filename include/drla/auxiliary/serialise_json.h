#pragma once

#include "drla/configuration.h"
#include "drla/types.h"

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

namespace
{

struct optional_input
{
	const nlohmann::json& json;
	const char* name;
};

template <typename T>
static inline void operator<<(T& value, const optional_input&& input)
{
	auto item = input.json.find(input.name);
	if (item != input.json.end())
	{
		value = item->get<T>();
	}
}

struct required_input
{
	const nlohmann::json& json;
	const char* name;
	const bool throw_on_error = false;
};

template <typename T>
static inline void operator<<(T& value, const required_input&& input)
{
	auto item = input.json.find(input.name);
	if (item != input.json.end())
	{
		value = item->get<T>();
	}
	else if (!input.throw_on_error)
	{
		spdlog::warn("[Config] Missing parameter '{}'. Using default instead.", input.name);
	}
	else
	{
		std::string err = "[Config] Missing parameter: ";
		err += input.name;
		throw std::runtime_error(err.c_str());
	}
}

} // namespace

namespace drla
{

NLOHMANN_JSON_SERIALIZE_ENUM(
	TrainAlgorithmType,
	{
		{TrainAlgorithmType::kNone, "None"},
		{TrainAlgorithmType::kA2C, "A2C"},
		{TrainAlgorithmType::kPPO, "PPO"},
		{TrainAlgorithmType::kSAC, "SAC"},
		{TrainAlgorithmType::kDQN, "DQN"},
		{TrainAlgorithmType::kMuZero, "MuZero"},
	})

NLOHMANN_JSON_SERIALIZE_ENUM(
	AgentPolicyModelType,
	{
		{AgentPolicyModelType::kRandom, "Random"},
		{AgentPolicyModelType::kActorCritic, "ActorCritic"},
		{AgentPolicyModelType::kSoftActorCritic, "SoftActorCritic"},
		{AgentPolicyModelType::kQNet, "QNet"},
		{AgentPolicyModelType::kMuZero, "MuZero"},
	})

NLOHMANN_JSON_SERIALIZE_ENUM(
	FeatureExtractorType,
	{
		{FeatureExtractorType::kMLP, "MLP"},
		{FeatureExtractorType::kCNN, "CNN"},
	})

NLOHMANN_JSON_SERIALIZE_ENUM(
	LayerType,
	{
		{LayerType::kInvalid, "Invalid"},
		{LayerType::kConv2d, "Conv2d"},
		{LayerType::kConvTranspose2d, "ConvTranspose2d"},
		{LayerType::kBatchNorm2d, "BatchNorm2d"},
		{LayerType::kLayerNorm, "LayerNorm"},
		{LayerType::kMaxPool2d, "MaxPool2d"},
		{LayerType::kAvgPool2d, "AvgPool2d"},
		{LayerType::kAdaptiveAvgPool2d, "AdaptiveAvgPool2d"},
		{LayerType::kResBlock2d, "ResBlock2d"},
		{LayerType::kActivation, "Activation"},
	})

NLOHMANN_JSON_SERIALIZE_ENUM(
	LearningRateScheduleType,
	{
		{LearningRateScheduleType::kConstant, "Constant"},
		{LearningRateScheduleType::kLinear, "Linear"},
		{LearningRateScheduleType::kExponential, "Exponential"},
	})

NLOHMANN_JSON_SERIALIZE_ENUM(
	OptimiserType,
	{
		{OptimiserType::kAdam, "Adam"},
		{OptimiserType::kSGD, "SGD"},
		{OptimiserType::kRMSProp, "RMSProp"},
	})

NLOHMANN_JSON_SERIALIZE_ENUM(
	ActionSpaceType,
	{
		{ActionSpaceType::kDiscrete, "Discrete"},
		{ActionSpaceType::kBox, "Box"},
		{ActionSpaceType::kMultiBinary, "MultiBinary"},
		{ActionSpaceType::kMultiDiscrete, "MultiDiscrete"},
	})

namespace Config
{

NLOHMANN_JSON_SERIALIZE_ENUM(
	Activation,
	{
		{Activation::kNone, "None"},
		{Activation::kReLU, "ReLU"},
		{Activation::kLeakyReLU, "LeakyReLU"},
		{Activation::kSigmoid, "Sigmoid"},
		{Activation::kTanh, "Tanh"},
		{Activation::kELU, "ELU"},
		{Activation::kSiLU, "SiLU"},
		{Activation::kSoftplus, "Softplus"},
	})

NLOHMANN_JSON_SERIALIZE_ENUM(
	FCLayerType,
	{
		{FCLayerType::kLinear, "Linear"},
		{FCLayerType::kLayerNorm, "LayerNorm"},
		{FCLayerType::kLayerConnection, "LayerConnection"},
		{FCLayerType::kActivation, "Activation"},
	})

NLOHMANN_JSON_SERIALIZE_ENUM(
	InitType,
	{
		{InitType::kDefault, "Default"},
		{InitType::kOrthogonal, "Orthogonal"},
		{InitType::kKaimingUniform, "KaimingUniform"},
		{InitType::kKaimingNormal, "KaimingNormal"},
		{InitType::kXavierUniform, "XavierUniform"},
		{InitType::kXavierNormal, "XavierNormal"},
	})

template <typename T>
void load_init_weights_config(T& layer, const nlohmann::json& json)
{
	layer.init_weight_type << optional_input{json, "init_weight_type"};
	switch (layer.init_weight_type)
	{
		case InitType::kDefault:
		case InitType::kOrthogonal: break;
		case InitType::kConstant:
		case InitType::kXavierUniform:
		case InitType::kXavierNormal: layer.init_weight = 1.0; break;
		case InitType::kKaimingUniform:
		case InitType::kKaimingNormal: layer.init_weight = 0.0; break;
	}
	layer.init_weight << optional_input{json, "init_weight"};
}

template <typename T>
void load_init_bias_config(T& layer, const nlohmann::json& json)
{
	layer.init_bias_type << optional_input{json, "init_bias_type"};
	switch (layer.init_bias_type)
	{
		case InitType::kDefault: break;
		case InitType::kXavierUniform:
		case InitType::kXavierNormal: layer.init_weight = 1.0; break;
		case InitType::kConstant:
		case InitType::kOrthogonal:
		case InitType::kKaimingUniform:
		case InitType::kKaimingNormal: layer.init_weight = 0.0; break;
	}
	layer.init_bias << optional_input{json, "init_bias"};
}

void from_json(const nlohmann::json& json, Rewards& reward_config);

void to_json(nlohmann::json& json, const Rewards& reward_config);

void from_json(const nlohmann::json& json, OptimiserBase& optimiser);

void to_json(nlohmann::json& json, const OptimiserBase& optimiser);

void from_json(const nlohmann::json& json, OptimiserAdam& optimiser);

void to_json(nlohmann::json& json, const OptimiserAdam& optimiser);

void from_json(const nlohmann::json& json, OptimiserSGD& optimiser);

void to_json(nlohmann::json& json, const OptimiserSGD& optimiser);

void from_json(const nlohmann::json& json, OptimiserRMSProp& optimiser);

void to_json(nlohmann::json& json, const OptimiserRMSProp& optimiser);

void from_json(const nlohmann::json& json, Optimiser& optimiser);

void to_json(nlohmann::json& json, const Optimiser& optimiser);

void from_json(const nlohmann::json& json, TrainAlgorithm& train_algorithm);

void to_json(nlohmann::json& json, const TrainAlgorithm& train_algorithm);

void from_json(const nlohmann::json& json, OnPolicyAlgorithm& on_policy_algorithm);

void to_json(nlohmann::json& json, const OnPolicyAlgorithm& on_policy_algorithm);

void from_json(const nlohmann::json& json, OffPolicyAlgorithm& off_policy_algorithm);

void to_json(nlohmann::json& json, const OffPolicyAlgorithm& off_policy_algorithm);

void from_json(const nlohmann::json& json, MCTSAlgorithm& mcts_algorithm);

void to_json(nlohmann::json& json, const MCTSAlgorithm& mcts_algorithm);

void from_json(const nlohmann::json& json, A2C& alg_a2c);

void to_json(nlohmann::json& json, const A2C& alg_a2c);

void from_json(const nlohmann::json& json, PPO& alg_ppo);

void to_json(nlohmann::json& json, const PPO& alg_ppo);

void from_json(const nlohmann::json& json, DQN& alg_dqn);

void to_json(nlohmann::json& json, const DQN& alg_dqn);

void from_json(const nlohmann::json& json, SAC& alg_sac);

void to_json(nlohmann::json& json, const SAC& alg_sac);

void from_json(const nlohmann::json& json, AgentTrainAlgorithm& train_algorithm);

void to_json(nlohmann::json& json, const AgentTrainAlgorithm& train_algorithm);

void from_json(const nlohmann::json& json, LinearConfig& linear);

void to_json(nlohmann::json& json, const LinearConfig& linear);

void from_json(const nlohmann::json& json, LayerConnectionConfig& layer_connection);

void to_json(nlohmann::json& json, const LayerConnectionConfig& layer_connection);

void from_json(const nlohmann::json& json, FCConfig& fc);

void to_json(nlohmann::json& json, const FCConfig& fc);

void from_json(const nlohmann::json& json, MLPConfig& mlp);

void to_json(nlohmann::json& json, const MLPConfig& mlp);

void from_json(const nlohmann::json& json, Conv2dConfig& conv);

void to_json(nlohmann::json& json, const Conv2dConfig& conv);

void from_json(const nlohmann::json& json, BatchNorm2dConfig& batch_norm);

void to_json(nlohmann::json& json, const BatchNorm2dConfig& batch_norm);

void from_json(const nlohmann::json& json, LayerNormConfig& layer_norm);

void to_json(nlohmann::json& json, const LayerNormConfig& layer_norm);

void from_json(const nlohmann::json& json, MaxPool2dConfig& maxpool);

void to_json(nlohmann::json& json, const MaxPool2dConfig& maxpool);

void from_json(const nlohmann::json& json, AvgPool2dConfig& avgpool);

void to_json(nlohmann::json& json, const AvgPool2dConfig& avgpool);

void from_json(const nlohmann::json& json, AdaptiveAvgPool2dConfig& adaptavgpool);

void to_json(nlohmann::json& json, const AdaptiveAvgPool2dConfig& adaptavgpool);

void from_json(const nlohmann::json& json, ResBlock2dConfig& resblock);

void to_json(nlohmann::json& json, const ResBlock2dConfig& resblock);

void from_json(const nlohmann::json& json, CNNLayerConfig& cnn_layer_config);

void to_json(nlohmann::json& json, const CNNLayerConfig& cnn_layer_config);

void from_json(const nlohmann::json& json, CNNConfig& cnn);

void to_json(nlohmann::json& json, const CNNConfig& cnn);

void from_json(const nlohmann::json& json, FeatureExtractorGroup& feature_extractor);

void to_json(nlohmann::json& json, const FeatureExtractorGroup& feature_extractor);

void from_json(const nlohmann::json& json, FeatureExtractorConfig& feature_extractor);

void to_json(nlohmann::json& json, const FeatureExtractorConfig& feature_extractor);

void from_json(const nlohmann::json& json, ActorConfig& paoc);

void to_json(nlohmann::json& json, const ActorConfig& paoc);

void from_json(const nlohmann::json& json, ActorCriticConfig& actor_critic);

void to_json(nlohmann::json& json, const ActorCriticConfig& actor_critic);

void from_json(const nlohmann::json& json, SoftActorCriticConfig& sac);

void to_json(nlohmann::json& json, const SoftActorCriticConfig& sac);

void from_json(const nlohmann::json& json, QNetModelConfig& dqn_model);

void to_json(nlohmann::json& json, const QNetModelConfig& dqn_model);

void from_json(const nlohmann::json& json, RandomConfig& random);

void to_json(nlohmann::json& json, const RandomConfig& random);

void from_json(const nlohmann::json& json, ModelConfig& model);

void to_json(nlohmann::json& json, const ModelConfig& model);

void from_json(const nlohmann::json& json, Config::AgentBase& agent);

void to_json(nlohmann::json& json, const Config::AgentBase& agent);

void from_json(const nlohmann::json& json, Config::InteractiveAgent& agent);

void to_json(nlohmann::json& json, const Config::InteractiveAgent& agent);

void from_json(const nlohmann::json& json, Config::OnPolicyAgent& agent);

void to_json(nlohmann::json& json, const Config::OnPolicyAgent& agent);

void from_json(const nlohmann::json& json, Config::OffPolicyAgent& agent);

void to_json(nlohmann::json& json, const Config::OffPolicyAgent& agent);

void from_json(const nlohmann::json& json, Config::MCTSAgent& agent);

void to_json(nlohmann::json& json, const Config::MCTSAgent& agent);

void to_json(nlohmann::json& json, const Config::Agent& agent);

void from_json(const nlohmann::json& json, Config::Agent& agent);

namespace MuZero
{

void from_json(const nlohmann::json& json, DynamicsNetwork& dyn_net);

void to_json(nlohmann::json& json, const DynamicsNetwork& dyn_net);

void from_json(const nlohmann::json& json, PredictionNetwork& pred_net);

void to_json(nlohmann::json& json, const PredictionNetwork& pred_net);

void from_json(const nlohmann::json& json, ModelConfig& muzero_config);

void to_json(nlohmann::json& json, const ModelConfig& muzero_config);

void from_json(const nlohmann::json& json, TrainConfig& alg_muzero);

void to_json(nlohmann::json& json, const TrainConfig& alg_muzero);

} // namespace MuZero

} // namespace Config

} // namespace drla
