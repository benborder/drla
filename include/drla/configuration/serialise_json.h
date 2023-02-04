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
	})

NLOHMANN_JSON_SERIALIZE_ENUM(
	AgentPolicyModelType,
	{
		{AgentPolicyModelType::kRandom, "Random"},
		{AgentPolicyModelType::kActorCritic, "ActorCritic"},
		{AgentPolicyModelType::kSoftActorCritic, "SoftActorCritic"},
		{AgentPolicyModelType::kQNet, "QNet"},
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
		{LayerType::kConv2d, "Conv2d"},
		{LayerType::kMaxPool2d, "MaxPool2d"},
		{LayerType::kAvgPool2d, "AvgPool2d"},
		{LayerType::kAdaptiveAvgPool2d, "AdaptiveAvgPool2d"},
		{LayerType::kResBlock2d, "ResBlock2d"},
	})

NLOHMANN_JSON_SERIALIZE_ENUM(
	LearningRateScheduleType,
	{
		{LearningRateScheduleType::kConstant, "Constant"},
		{LearningRateScheduleType::kLinear, "Linear"},
		{LearningRateScheduleType::kExponential, "Exponential"},
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
		{Activation::kSoftplus, "Softplus"},
	})

NLOHMANN_JSON_SERIALIZE_ENUM(
	FCLayerType,
	{
		{FCLayerType::kLinear, "Linear"},
		{FCLayerType::kInputConnected, "InputConnected"},
		{FCLayerType::kMultiConnected, "MultiConnected"},
		{FCLayerType::kResidual, "Residual"},
		{FCLayerType::kForwardInput, "ForwardInput"},
		{FCLayerType::kForwardAll, "ForwardAll"},
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

static inline void from_json(const nlohmann::json& json, Rewards& reward_config)
{
	reward_config.reward_clamp_min << optional_input{json, "reward_clamp_min"};
	reward_config.reward_clamp_max << optional_input{json, "reward_clamp_max"};
	reward_config.combine_rewards << optional_input{json, "combine_rewards"};
}

static inline void to_json(nlohmann::json& json, const Rewards& reward_config)
{
	json["reward_clamp_min"] = reward_config.reward_clamp_min;
	json["reward_clamp_max"] = reward_config.reward_clamp_max;
	json["combine_rewards"] = reward_config.combine_rewards;
}

static inline void from_json(const nlohmann::json& json, TrainAlgorithm& train_algorithm)
{
	train_algorithm.total_timesteps << required_input{json, "total_timesteps"};
	train_algorithm.start_timestep << required_input{json, "start_timestep"};
	train_algorithm.learning_rate << required_input{json, "learning_rate"};
	train_algorithm.learning_rate_min << required_input{json, "learning_rate_min"};
	train_algorithm.lr_schedule_type << required_input{json, "lr_schedule_type"};
	train_algorithm.lr_decay_rate << optional_input{json, "lr_decay_rate"};
}

static inline void to_json(nlohmann::json& json, const TrainAlgorithm& train_algorithm)
{
	json["total_timesteps"] = train_algorithm.total_timesteps;
	json["start_timestep"] = train_algorithm.start_timestep;
	json["learning_rate"] = train_algorithm.learning_rate;
	json["learning_rate_min"] = train_algorithm.learning_rate_min;
	json["lr_schedule_type"] = train_algorithm.lr_schedule_type;
	json["lr_decay_rate"] = train_algorithm.lr_decay_rate;
}

static inline void from_json(const nlohmann::json& json, OnPolicyAlgorithm& on_policy_algorithm)
{
	from_json(json, static_cast<TrainAlgorithm&>(on_policy_algorithm));
	on_policy_algorithm.horizon_steps << required_input{json, "horizon_steps"};
	on_policy_algorithm.policy_loss_coef << required_input{json, "policy_loss_coef"};
	on_policy_algorithm.value_loss_coef << required_input{json, "value_loss_coef"};
	on_policy_algorithm.entropy_coef << required_input{json, "entropy_coef"};
	on_policy_algorithm.gae_lambda << required_input{json, "gae_lambda"};
	on_policy_algorithm.gamma << required_input{json, "gamma"};
}

static inline void to_json(nlohmann::json& json, const OnPolicyAlgorithm& on_policy_algorithm)
{
	to_json(json, static_cast<const TrainAlgorithm&>(on_policy_algorithm));
	json["horizon_steps"] = on_policy_algorithm.horizon_steps;
	json["policy_loss_coef"] = on_policy_algorithm.policy_loss_coef;
	json["value_loss_coef"] = on_policy_algorithm.value_loss_coef;
	json["entropy_coef"] = on_policy_algorithm.entropy_coef;
	json["gae_lambda"] = on_policy_algorithm.gae_lambda;
	json["gamma"] = on_policy_algorithm.gamma;
}

static inline void from_json(const nlohmann::json& json, OffPolicyAlgorithm& off_policy_algorithm)
{
	from_json(json, static_cast<TrainAlgorithm&>(off_policy_algorithm));
	off_policy_algorithm.horizon_steps << required_input{json, "horizon_steps"};
	off_policy_algorithm.buffer_size << optional_input{json, "buffer_size"};
	off_policy_algorithm.batch_size << optional_input{json, "batch_size"};
	off_policy_algorithm.learning_starts << optional_input{json, "learning_starts"};
	off_policy_algorithm.gamma << optional_input{json, "gamma"};
	off_policy_algorithm.tau << optional_input{json, "tau"};
	off_policy_algorithm.gradient_steps << optional_input{json, "gradient_steps"};
	off_policy_algorithm.target_update_interval << optional_input{json, "target_update_interval"};
}

static inline void to_json(nlohmann::json& json, const OffPolicyAlgorithm& off_policy_algorithm)
{
	to_json(json, static_cast<const TrainAlgorithm&>(off_policy_algorithm));
	json["horizon_steps"] = off_policy_algorithm.horizon_steps;
	json["buffer_size"] = off_policy_algorithm.buffer_size;
	json["batch_size"] = off_policy_algorithm.batch_size;
	json["learning_starts"] = off_policy_algorithm.learning_starts;
	json["gamma"] = off_policy_algorithm.gamma;
	json["tau"] = off_policy_algorithm.tau;
	json["gradient_steps"] = off_policy_algorithm.gradient_steps;
	json["target_update_interval"] = off_policy_algorithm.target_update_interval;
}

static inline void from_json(const nlohmann::json& json, A2C& alg_a2c)
{
	from_json(json, static_cast<OnPolicyAlgorithm&>(alg_a2c));
	alg_a2c.alpha << required_input{json, "alpha"};
	alg_a2c.epsilon << required_input{json, "epsilon"};
	alg_a2c.max_grad_norm << required_input{json, "max_grad_norm"};
}

static inline void to_json(nlohmann::json& json, const A2C& alg_a2c)
{
	json["train_algorithm_type"] = TrainAlgorithmType::kA2C;
	to_json(json, static_cast<const OnPolicyAlgorithm&>(alg_a2c));
	json["alpha"] = alg_a2c.alpha;
	json["epsilon"] = alg_a2c.epsilon;
	json["max_grad_norm"] = alg_a2c.max_grad_norm;
}

static inline void from_json(const nlohmann::json& json, PPO& alg_ppo)
{
	from_json(json, static_cast<OnPolicyAlgorithm&>(alg_ppo));
	alg_ppo.clip_range_policy << required_input{json, "clip_range_policy"};
	alg_ppo.clip_vf << required_input{json, "clip_vf"};
	alg_ppo.clip_range_vf << required_input{json, "clip_range_vf"};
	alg_ppo.num_epoch << required_input{json, "num_epoch"};
	alg_ppo.num_mini_batch << required_input{json, "num_mini_batch"};
	alg_ppo.max_grad_norm << required_input{json, "max_grad_norm"};
	alg_ppo.kl_target << required_input{json, "kl_target"};
}

static inline void to_json(nlohmann::json& json, const PPO& alg_ppo)
{
	json["train_algorithm_type"] = TrainAlgorithmType::kPPO;
	to_json(json, static_cast<const OnPolicyAlgorithm&>(alg_ppo));
	json["clip_range_policy"] = alg_ppo.clip_range_policy;
	json["clip_vf"] = alg_ppo.clip_vf;
	json["clip_range_vf"] = alg_ppo.clip_range_vf;
	json["num_epoch"] = alg_ppo.num_epoch;
	json["num_mini_batch"] = alg_ppo.num_mini_batch;
	json["max_grad_norm"] = alg_ppo.max_grad_norm;
	json["kl_target"] = alg_ppo.kl_target;
}

static inline void from_json(const nlohmann::json& json, DQN& alg_dqn)
{
	from_json(json, static_cast<OffPolicyAlgorithm&>(alg_dqn));
	alg_dqn.epsilon << optional_input{json, "epsilon"};
	alg_dqn.max_grad_norm << optional_input{json, "max_grad_norm"};
	alg_dqn.exploration_fraction << optional_input{json, "exploration_fraction"};
	alg_dqn.exploration_init << optional_input{json, "exploration_init"};
	alg_dqn.exploration_final << optional_input{json, "exploration_final"};
}

static inline void to_json(nlohmann::json& json, const DQN& alg_dqn)
{
	json["train_algorithm_type"] = TrainAlgorithmType::kDQN;
	to_json(json, static_cast<const OffPolicyAlgorithm&>(alg_dqn));
	json["epsilon"] = alg_dqn.epsilon;
	json["max_grad_norm"] = alg_dqn.max_grad_norm;
	json["exploration_fraction"] = alg_dqn.exploration_fraction;
	json["exploration_init"] = alg_dqn.exploration_init;
	json["exploration_final"] = alg_dqn.exploration_final;
}

static inline void from_json(const nlohmann::json& json, SAC& alg_sac)
{
	from_json(json, static_cast<OffPolicyAlgorithm&>(alg_sac));
	alg_sac.epsilon << optional_input{json, "epsilon"};
	alg_sac.actor_loss_coef << optional_input{json, "actor_loss_coef"};
	alg_sac.value_loss_coef << optional_input{json, "value_loss_coef"};
	alg_sac.target_entropy_scale << optional_input{json, "target_entropy_scale"};
}

static inline void to_json(nlohmann::json& json, const SAC& alg_sac)
{
	json["train_algorithm_type"] = TrainAlgorithmType::kSAC;
	to_json(json, static_cast<const OffPolicyAlgorithm&>(alg_sac));
	json["epsilon"] = alg_sac.epsilon;
	json["actor_loss_coef"] = alg_sac.actor_loss_coef;
	json["value_loss_coef"] = alg_sac.value_loss_coef;
	json["target_entropy_scale"] = alg_sac.target_entropy_scale;
}

static inline void from_json(const nlohmann::json& json, AgentTrainAlgorithm& train_algorithm)
{
	TrainAlgorithmType train_algorithm_type = TrainAlgorithmType::kNone;
	train_algorithm_type << optional_input{json, "train_algorithm_type"};
	switch (train_algorithm_type)
	{
		case TrainAlgorithmType::kA2C: train_algorithm = json.get<A2C>(); break;
		case TrainAlgorithmType::kPPO: train_algorithm = json.get<PPO>(); break;
		case TrainAlgorithmType::kDQN: train_algorithm = json.get<DQN>(); break;
		case TrainAlgorithmType::kSAC: train_algorithm = json.get<SAC>(); break;
		case TrainAlgorithmType::kNone: break;
	}
}

static inline void to_json(nlohmann::json& json, const AgentTrainAlgorithm& train_algorithm)
{
	std::visit([&](auto& alg) { to_json(json, alg); }, train_algorithm);
}

static inline void from_json(const nlohmann::json& json, FCConfig::fc_layer& layer)
{
	layer.type << optional_input{json, "type"};
	if (
		layer.type == Config::FCLayerType::kResidual || layer.type == Config::FCLayerType::kForwardAll ||
		layer.type == Config::FCLayerType::kForwardInput)
	{
		layer.size << optional_input{json, "size"};
	}
	else
	{
		layer.size << required_input{json, "size"};
	}
	layer.activation << optional_input{json, "activation"};
	load_init_weights_config(layer, json);
	load_init_bias_config(layer, json);
}

static inline void to_json(nlohmann::json& json, const FCConfig::fc_layer& layer)
{
	json["size"] = layer.size;
	json["activation"] = layer.activation;
	json["init_bias_type"] = layer.init_bias_type;
	json["init_bias"] = layer.init_bias;
	json["init_weight_type"] = layer.init_weight_type;
	json["init_weight"] = layer.init_weight;
	json["type"] = layer.type;
}

static inline void from_json(const nlohmann::json& json, FCConfig& fc)
{
	fc.layers << required_input{json, "layers"};
}

static inline void to_json(nlohmann::json& json, const FCConfig& fc)
{
	json["layers"] = fc.layers;
}

static inline void from_json(const nlohmann::json& json, MLPConfig& mlp)
{
	from_json(json, static_cast<FCConfig&>(mlp));
	mlp.name << optional_input{json, "name"};
}

static inline void to_json(nlohmann::json& json, const MLPConfig& mlp)
{
	json["type"] = FeatureExtractorType::kMLP;
	to_json(json, static_cast<const FCConfig&>(mlp));
	json["name"] = mlp.name;
}

static inline void from_json(const nlohmann::json& json, Conv2dConfig& conv)
{
	conv.in_channels << optional_input{json, "in_channels"};
	conv.out_channels << required_input{json, "out_channels"};
	conv.kernel_size << required_input{json, "kernel_size"};
	conv.stride << required_input{json, "stride"};
	conv.padding << optional_input{json, "padding"};
	load_init_weights_config(conv, json);
	load_init_bias_config(conv, json);
	conv.use_bias << optional_input{json, "use_bias"};
	conv.activation << optional_input{json, "activation"};
}

static inline void to_json(nlohmann::json& json, const Conv2dConfig& conv)
{
	json["type"] = LayerType::kConv2d;
	json["in_channels"] = conv.in_channels;
	json["out_channels"] = conv.out_channels;
	json["kernel_size"] = conv.kernel_size;
	json["stride"] = conv.stride;
	json["padding"] = conv.padding;
	json["init_weight_type"] = conv.init_weight_type;
	json["init_weight"] = conv.init_weight;
	json["init_bias_type"] = conv.init_bias_type;
	json["init_bias"] = conv.init_bias;
	json["use_bias"] = conv.use_bias;
	json["activation"] = conv.activation;
}

static inline void from_json(const nlohmann::json& json, MaxPool2dConfig& maxpool)
{
	maxpool.kernel_size << required_input{json, "kernel_size"};
	maxpool.stride << required_input{json, "stride"};
	maxpool.padding << optional_input{json, "padding"};
}

static inline void to_json(nlohmann::json& json, const MaxPool2dConfig& maxpool)
{
	json["type"] = LayerType::kMaxPool2d;
	json["kernel_size"] = maxpool.kernel_size;
	json["stride"] = maxpool.stride;
	json["padding"] = maxpool.padding;
}

static inline void from_json(const nlohmann::json& json, AvgPool2dConfig& avgpool)
{
	avgpool.kernel_size << required_input{json, "kernel_size"};
	avgpool.stride << required_input{json, "stride"};
	avgpool.padding << optional_input{json, "padding"};
}

static inline void to_json(nlohmann::json& json, const AvgPool2dConfig& avgpool)
{
	json["type"] = LayerType::kAvgPool2d;
	json["kernel_size"] = avgpool.kernel_size;
	json["stride"] = avgpool.stride;
	json["padding"] = avgpool.padding;
}

static inline void from_json(const nlohmann::json& json, AdaptiveAvgPool2dConfig& adaptavgpool)
{
	adaptavgpool.size << required_input{json, "size"};
}

static inline void to_json(nlohmann::json& json, const AdaptiveAvgPool2dConfig& adaptavgpool)
{
	json["type"] = LayerType::kAdaptiveAvgPool2d;
	json["size"] = adaptavgpool.size;
}

static inline void from_json(const nlohmann::json& json, ResBlock2dConfig& resblock)
{
	resblock.layers << optional_input{json, "layers"};
	resblock.kernel_size << optional_input{json, "kernel_size"};
	resblock.stride << optional_input{json, "stride"};
	resblock.normalise << optional_input{json, "normalise"};
	resblock.init_weight_type << optional_input{json, "init_weight_type"};
	load_init_weights_config(resblock, json);
}

static inline void to_json(nlohmann::json& json, const ResBlock2dConfig& resblock)
{
	json["type"] = LayerType::kResBlock2d;
	json["layers"] = resblock.layers;
	json["kernel_size"] = resblock.kernel_size;
	json["stride"] = resblock.stride;
	json["normalise"] = resblock.normalise;
	json["init_weight_type"] = resblock.init_weight_type;
	json["init_weight"] = resblock.init_weight;
}

static inline void from_json(const nlohmann::json& json, CNNLayerConfig& cnn_layer_config)
{
	LayerType type;
	type << required_input{json, "type"};
	switch (type)
	{
		case LayerType::kConv2d: cnn_layer_config = json.get<Conv2dConfig>(); break;
		case LayerType::kMaxPool2d: cnn_layer_config = json.get<MaxPool2dConfig>(); break;
		case LayerType::kAvgPool2d: cnn_layer_config = json.get<AvgPool2dConfig>(); break;
		case LayerType::kAdaptiveAvgPool2d: cnn_layer_config = json.get<AdaptiveAvgPool2dConfig>(); break;
		case LayerType::kResBlock2d: cnn_layer_config = json.get<ResBlock2dConfig>(); break;
		default:
		{
			spdlog::error("[Config] The CNN layer type is not defined.");
			throw std::runtime_error("The CNN layer type is not defined");
			break;
		}
	}
}

static inline void to_json(nlohmann::json& json, const CNNLayerConfig& cnn_layer_config)
{
	std::visit([&json](const auto& cnn_layer) { to_json(json, cnn_layer); }, cnn_layer_config);
}

static inline void from_json(const nlohmann::json& json, CNNConfig& cnn)
{
	cnn.layers << optional_input{json, "layers"};
}

static inline void to_json(nlohmann::json& json, const CNNConfig& cnn)
{
	json["type"] = FeatureExtractorType::kCNN;
	json["layers"] = cnn.layers;
}

static inline void from_json(const nlohmann::json& json, FeatureExtractorGroup& feature_extractor)
{
	FeatureExtractorType type;
	type << required_input{json, "type"};
	switch (type)
	{
		case FeatureExtractorType::kMLP: feature_extractor = json.get<MLPConfig>(); break;
		case FeatureExtractorType::kCNN: feature_extractor = json.get<CNNConfig>(); break;
	}
}

static inline void to_json(nlohmann::json& json, const FeatureExtractorGroup& feature_extractor)
{
	std::visit([&](auto& feature_extractor) { to_json(json, feature_extractor); }, feature_extractor);
}

static inline void from_json(const nlohmann::json& json, FeatureExtractorConfig& feature_extractor)
{
	feature_extractor.feature_groups = json.get<decltype(feature_extractor.feature_groups)>();
}

static inline void to_json(nlohmann::json& json, const FeatureExtractorConfig& feature_extractor)
{
	json = feature_extractor.feature_groups;
}

static inline void from_json(const nlohmann::json& json, PolicyActionOutputConfig& paoc)
{
	paoc.activation << optional_input{json, "activation"};
	load_init_weights_config(paoc, json);
	load_init_bias_config(paoc, json);
}

static inline void to_json(nlohmann::json& json, const PolicyActionOutputConfig& paoc)
{
	json["activation"] = paoc.activation;
	json["init_bias_type"] = paoc.init_bias_type;
	json["init_bias"] = paoc.init_bias;
	json["init_weight_type"] = paoc.init_weight_type;
	json["init_weight"] = paoc.init_weight;
}

static inline void from_json(const nlohmann::json& json, CommonModelConfig& model_config)
{
}

static inline void to_json(nlohmann::json& json, const CommonModelConfig& model_config)
{
}

static inline void from_json(const nlohmann::json& json, ActorCriticConfig& actor_critic)
{
	from_json(json, static_cast<CommonModelConfig&>(actor_critic));
	actor_critic.use_shared_extractor << optional_input{json, "use_shared_extractor"};
	actor_critic.feature_extractor << required_input{json, "feature_extractor"};
	actor_critic.shared << optional_input{json, "shared"};
	actor_critic.actor << optional_input{json, "actor"};
	actor_critic.critic << optional_input{json, "critic"};
	actor_critic.policy_action_output << optional_input{json, "policy_action_output"};
	actor_critic.predict_values << optional_input{json, "predict_values"};
}

static inline void to_json(nlohmann::json& json, const ActorCriticConfig& actor_critic)
{
	json["model_type"] = AgentPolicyModelType::kActorCritic;
	to_json(json, static_cast<const CommonModelConfig&>(actor_critic));
	json["use_shared_extractor"] = actor_critic.use_shared_extractor;
	json["feature_extractor"] = actor_critic.feature_extractor;
	json["shared"] = actor_critic.shared;
	json["actor"] = actor_critic.actor;
	json["critic"] = actor_critic.critic;
	json["policy_action_output"] = actor_critic.policy_action_output;
	json["predict_values"] = actor_critic.predict_values;
}

static inline void from_json(const nlohmann::json& json, SoftActorCriticConfig& sac)
{
	from_json(json, static_cast<CommonModelConfig&>(sac));
	sac.feature_extractor << required_input{json, "feature_extractor"};
	sac.actor << optional_input{json, "actor"};
	sac.critic << optional_input{json, "critic"};
	sac.shared_feature_extractor << optional_input{json, "shared_feature_extractor"};
	sac.n_critics << optional_input{json, "n_critics"};
	sac.policy_action_output << optional_input{json, "policy_action_output"};
	sac.predict_values << optional_input{json, "predict_values"};
}

static inline void to_json(nlohmann::json& json, const SoftActorCriticConfig& sac)
{
	json["model_type"] = AgentPolicyModelType::kSoftActorCritic;
	to_json(json, static_cast<const CommonModelConfig&>(sac));
	json["feature_extractor"] = sac.feature_extractor;
	json["actor"] = sac.actor;
	json["critic"] = sac.critic;
	json["shared_feature_extractor"] = sac.shared_feature_extractor;
	json["n_critics"] = sac.n_critics;
	json["policy_action_output"] = sac.policy_action_output;
	json["predict_values"] = sac.predict_values;
}

static inline void from_json(const nlohmann::json& json, QNetModelConfig& dqn_model)
{
	from_json(json, static_cast<CommonModelConfig&>(dqn_model));
	dqn_model.feature_extractor << required_input{json, "feature_extractor"};
	dqn_model.q_net << required_input{json, "q_net"};
}

static inline void to_json(nlohmann::json& json, const QNetModelConfig& dqn_model)
{
	json["model_type"] = AgentPolicyModelType::kQNet;
	to_json(json, static_cast<const CommonModelConfig&>(dqn_model));
	json["feature_extractor"] = dqn_model.feature_extractor;
	json["q_net"] = dqn_model.q_net;
}

static inline void from_json(const nlohmann::json& json, RandomConfig& random)
{
}

static inline void to_json(nlohmann::json& json, const RandomConfig& random)
{
	json["model_type"] = AgentPolicyModelType::kRandom;
}

static inline void from_json(const nlohmann::json& json, ModelConfig& model)
{
	AgentPolicyModelType model_type;
	model_type << required_input{json, "model_type"};
	switch (model_type)
	{
		case AgentPolicyModelType::kRandom: model = json.get<RandomConfig>(); break;
		case AgentPolicyModelType::kActorCritic: model = json.get<ActorCriticConfig>(); break;
		case AgentPolicyModelType::kSoftActorCritic: model = json.get<SoftActorCriticConfig>(); break;
		case AgentPolicyModelType::kQNet: model = json.get<QNetModelConfig>(); break;
	}
}

static inline void to_json(nlohmann::json& json, const ModelConfig& model)
{
	std::visit([&](auto& model) { to_json(json, model); }, model);
}

static inline void from_json(const nlohmann::json& json, Config::AgentBase& agent)
{
	agent.env_count << optional_input{json, "env_count"};
	agent.use_cuda << optional_input{json, "use_cuda"};

	agent.model << required_input{json, "model"};
	agent.model_type << required_input{json.at("model"), "model_type"};

	agent.train_algorithm << optional_input{json, "train_algorithm"};
	agent.train_algorithm_type << optional_input{json.at("train_algorithm"), "train_algorithm_type"};

	agent.rewards << optional_input{json, "rewards"};

	agent.timestep_save_period << optional_input{json, "timestep_save_period"};
	agent.checkpoint_save_period << optional_input{json, "checkpoint_save_period"};
}

static inline void to_json(nlohmann::json& json, const Config::AgentBase& agent)
{
	json["env_count"] = agent.env_count;
	json["use_cuda"] = agent.use_cuda;

	json["model"] = agent.model;

	json["train_algorithm"] = agent.train_algorithm;

	json["rewards"] = agent.rewards;

	json["timestep_save_period"] = agent.timestep_save_period;
	json["checkpoint_save_period"] = agent.checkpoint_save_period;
}

static inline void from_json(const nlohmann::json& json, Config::InteractiveAgent& agent)
{
}

static inline void to_json(nlohmann::json& json, const Config::InteractiveAgent& agent)
{
}

static inline void from_json(const nlohmann::json& json, Config::OnPolicyAgent& agent)
{
	from_json(json, static_cast<Config::AgentBase&>(agent));
	agent.asynchronous_env << optional_input{json, "asynchronous_env"};
}

static inline void to_json(nlohmann::json& json, const Config::OnPolicyAgent& agent)
{
	to_json(json, static_cast<const Config::AgentBase&>(agent));
	json["asynchronous_env"] = agent.asynchronous_env;
}

static inline void from_json(const nlohmann::json& json, Config::OffPolicyAgent& agent)
{
	from_json(json, static_cast<Config::AgentBase&>(agent));
}

static inline void to_json(nlohmann::json& json, const Config::OffPolicyAgent& agent)
{
	to_json(json, static_cast<const Config::AgentBase&>(agent));
}

static inline void to_json(nlohmann::json& json, const Config::Agent& agent)
{
	std::visit([&](auto& agent) { to_json(json, agent); }, agent);
}

static inline void from_json(const nlohmann::json& json, Config::Agent& agent)
{
	auto model_json = json.find("model");
	if (model_json != json.end())
	{
		AgentPolicyModelType model_type;
		model_type << required_input{*model_json, "model_type"};
		switch (model_type)
		{
			case AgentPolicyModelType::kRandom: agent = json.get<Config::AgentBase>(); break;
			case AgentPolicyModelType::kActorCritic: agent = json.get<Config::OnPolicyAgent>(); break;
			case AgentPolicyModelType::kSoftActorCritic: agent = json.get<Config::OffPolicyAgent>(); break;
			case AgentPolicyModelType::kQNet: agent = json.get<Config::OffPolicyAgent>(); break;
		}
	}
	else
	{
		agent = json.get<Config::AgentBase>();
	}
}

} // namespace Config

} // namespace drla
