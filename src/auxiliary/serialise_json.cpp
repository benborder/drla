#include "serialise_json.h"

namespace drla::Config
{

void from_json(const nlohmann::json& json, Rewards& reward_config)
{
	reward_config.reward_clamp_min << optional_input{json, "reward_clamp_min"};
	reward_config.reward_clamp_max << optional_input{json, "reward_clamp_max"};
	reward_config.combine_rewards << optional_input{json, "combine_rewards"};
}

void to_json(nlohmann::json& json, const Rewards& reward_config)
{
	json["reward_clamp_min"] = reward_config.reward_clamp_min;
	json["reward_clamp_max"] = reward_config.reward_clamp_max;
	json["combine_rewards"] = reward_config.combine_rewards;
}

void from_json(const nlohmann::json& json, TrainAlgorithm& train_algorithm)
{
	train_algorithm.total_timesteps << required_input{json, "total_timesteps"};
	train_algorithm.start_timestep << required_input{json, "start_timestep"};
	train_algorithm.learning_rate << required_input{json, "learning_rate"};
	train_algorithm.learning_rate_min << required_input{json, "learning_rate_min"};
	train_algorithm.lr_schedule_type << required_input{json, "lr_schedule_type"};
	train_algorithm.lr_decay_rate << optional_input{json, "lr_decay_rate"};
	train_algorithm.eval_max_steps << optional_input{json, "eval_max_steps"};
	train_algorithm.eval_determinisic << optional_input{json, "eval_determinisic"};
}

void to_json(nlohmann::json& json, const TrainAlgorithm& train_algorithm)
{
	json["total_timesteps"] = train_algorithm.total_timesteps;
	json["start_timestep"] = train_algorithm.start_timestep;
	json["learning_rate"] = train_algorithm.learning_rate;
	json["learning_rate_min"] = train_algorithm.learning_rate_min;
	json["lr_schedule_type"] = train_algorithm.lr_schedule_type;
	json["lr_decay_rate"] = train_algorithm.lr_decay_rate;
	json["eval_max_steps"] = train_algorithm.eval_max_steps;
	json["eval_determinisic"] = train_algorithm.eval_determinisic;
}

void from_json(const nlohmann::json& json, OnPolicyAlgorithm& on_policy_algorithm)
{
	from_json(json, static_cast<TrainAlgorithm&>(on_policy_algorithm));
	on_policy_algorithm.horizon_steps << required_input{json, "horizon_steps"};
	on_policy_algorithm.policy_loss_coef << required_input{json, "policy_loss_coef"};
	on_policy_algorithm.value_loss_coef << required_input{json, "value_loss_coef"};
	on_policy_algorithm.entropy_coef << required_input{json, "entropy_coef"};
	on_policy_algorithm.gae_lambda << required_input{json, "gae_lambda"};
	on_policy_algorithm.gamma << required_input{json, "gamma"};
}

void to_json(nlohmann::json& json, const OnPolicyAlgorithm& on_policy_algorithm)
{
	to_json(json, static_cast<const TrainAlgorithm&>(on_policy_algorithm));
	json["horizon_steps"] = on_policy_algorithm.horizon_steps;
	json["policy_loss_coef"] = on_policy_algorithm.policy_loss_coef;
	json["value_loss_coef"] = on_policy_algorithm.value_loss_coef;
	json["entropy_coef"] = on_policy_algorithm.entropy_coef;
	json["gae_lambda"] = on_policy_algorithm.gae_lambda;
	json["gamma"] = on_policy_algorithm.gamma;
}

void from_json(const nlohmann::json& json, OffPolicyAlgorithm& off_policy_algorithm)
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
	off_policy_algorithm.per_alpha << optional_input{json, "per_alpha"};
}

void to_json(nlohmann::json& json, const OffPolicyAlgorithm& off_policy_algorithm)
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
	json["per_alpha"] = off_policy_algorithm.per_alpha;
}

void from_json(const nlohmann::json& json, MCTSAlgorithm& mcts_algorithm)
{
	from_json(json, static_cast<TrainAlgorithm&>(mcts_algorithm));
	mcts_algorithm.buffer_size << optional_input{json, "buffer_size"};
	mcts_algorithm.start_buffer_size << optional_input{json, "start_buffer_size"};
	mcts_algorithm.batch_size << optional_input{json, "batch_size"};
	mcts_algorithm.td_steps << optional_input{json, "td_steps"};
	mcts_algorithm.unroll_steps << optional_input{json, "unroll_steps"};
	mcts_algorithm.min_reanalyse_train_steps << optional_input{json, "min_reanalyse_train_steps"};
	mcts_algorithm.min_reanalyse_buffer_size << optional_input{json, "min_reanalyse_buffer_size"};
	mcts_algorithm.per_alpha << optional_input{json, "per_alpha"};
	mcts_algorithm.temperature_step << optional_input{json, "temperature_step"};
	mcts_algorithm.train_ratio << optional_input{json, "train_ratio"};
	mcts_algorithm.self_play_gpus << optional_input{json, "self_play_gpus"};
	mcts_algorithm.train_gpus << optional_input{json, "train_gpus"};
}

void to_json(nlohmann::json& json, const MCTSAlgorithm& mcts_algorithm)
{
	to_json(json, static_cast<const TrainAlgorithm&>(mcts_algorithm));
	json["buffer_size"] = mcts_algorithm.buffer_size;
	json["start_buffer_size"] = mcts_algorithm.start_buffer_size;
	json["batch_size"] = mcts_algorithm.batch_size;
	json["td_steps"] = mcts_algorithm.td_steps;
	json["unroll_steps"] = mcts_algorithm.unroll_steps;
	json["min_reanalyse_train_steps"] = mcts_algorithm.min_reanalyse_train_steps;
	json["min_reanalyse_buffer_size"] = mcts_algorithm.min_reanalyse_buffer_size;
	json["per_alpha"] = mcts_algorithm.per_alpha;
	json["temperature_step"] = mcts_algorithm.temperature_step;
	json["train_ratio"] = mcts_algorithm.train_ratio;
	json["self_play_gpus"] = mcts_algorithm.self_play_gpus;
	json["train_gpus"] = mcts_algorithm.train_gpus;
}

void from_json(const nlohmann::json& json, A2C& alg_a2c)
{
	from_json(json, static_cast<OnPolicyAlgorithm&>(alg_a2c));
	alg_a2c.alpha << required_input{json, "alpha"};
	alg_a2c.epsilon << required_input{json, "epsilon"};
	alg_a2c.max_grad_norm << required_input{json, "max_grad_norm"};
}

void to_json(nlohmann::json& json, const A2C& alg_a2c)
{
	json["train_algorithm_type"] = TrainAlgorithmType::kA2C;
	to_json(json, static_cast<const OnPolicyAlgorithm&>(alg_a2c));
	json["alpha"] = alg_a2c.alpha;
	json["epsilon"] = alg_a2c.epsilon;
	json["max_grad_norm"] = alg_a2c.max_grad_norm;
}

void from_json(const nlohmann::json& json, PPO& alg_ppo)
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

void to_json(nlohmann::json& json, const PPO& alg_ppo)
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

void from_json(const nlohmann::json& json, DQN& alg_dqn)
{
	from_json(json, static_cast<OffPolicyAlgorithm&>(alg_dqn));
	alg_dqn.epsilon << optional_input{json, "epsilon"};
	alg_dqn.max_grad_norm << optional_input{json, "max_grad_norm"};
	alg_dqn.exploration_fraction << optional_input{json, "exploration_fraction"};
	alg_dqn.exploration_init << optional_input{json, "exploration_init"};
	alg_dqn.exploration_final << optional_input{json, "exploration_final"};
}

void to_json(nlohmann::json& json, const DQN& alg_dqn)
{
	json["train_algorithm_type"] = TrainAlgorithmType::kDQN;
	to_json(json, static_cast<const OffPolicyAlgorithm&>(alg_dqn));
	json["epsilon"] = alg_dqn.epsilon;
	json["max_grad_norm"] = alg_dqn.max_grad_norm;
	json["exploration_fraction"] = alg_dqn.exploration_fraction;
	json["exploration_init"] = alg_dqn.exploration_init;
	json["exploration_final"] = alg_dqn.exploration_final;
}

void from_json(const nlohmann::json& json, SAC& alg_sac)
{
	from_json(json, static_cast<OffPolicyAlgorithm&>(alg_sac));
	alg_sac.epsilon << optional_input{json, "epsilon"};
	alg_sac.actor_loss_coef << optional_input{json, "actor_loss_coef"};
	alg_sac.value_loss_coef << optional_input{json, "value_loss_coef"};
	alg_sac.target_entropy_scale << optional_input{json, "target_entropy_scale"};
}

void to_json(nlohmann::json& json, const SAC& alg_sac)
{
	json["train_algorithm_type"] = TrainAlgorithmType::kSAC;
	to_json(json, static_cast<const OffPolicyAlgorithm&>(alg_sac));
	json["epsilon"] = alg_sac.epsilon;
	json["actor_loss_coef"] = alg_sac.actor_loss_coef;
	json["value_loss_coef"] = alg_sac.value_loss_coef;
	json["target_entropy_scale"] = alg_sac.target_entropy_scale;
}

void from_json(const nlohmann::json& json, AgentTrainAlgorithm& train_algorithm)
{
	TrainAlgorithmType train_algorithm_type = TrainAlgorithmType::kNone;
	train_algorithm_type << optional_input{json, "train_algorithm_type"};
	switch (train_algorithm_type)
	{
		case TrainAlgorithmType::kA2C: train_algorithm = json.get<A2C>(); break;
		case TrainAlgorithmType::kPPO: train_algorithm = json.get<PPO>(); break;
		case TrainAlgorithmType::kDQN: train_algorithm = json.get<DQN>(); break;
		case TrainAlgorithmType::kSAC: train_algorithm = json.get<SAC>(); break;
		case TrainAlgorithmType::kMuZero: train_algorithm = json.get<MuZero::TrainConfig>(); break;
		case TrainAlgorithmType::kNone: break;
	}
}

void to_json(nlohmann::json& json, const AgentTrainAlgorithm& train_algorithm)
{
	std::visit([&](auto& alg) { to_json(json, alg); }, train_algorithm);
}

void from_json(const nlohmann::json& json, LinearConfig& linear)
{
	linear.size << optional_input{json, "size"};
	load_init_weights_config(linear, json);
	linear.use_bias << optional_input{json, "use_bias"};
	if (linear.use_bias)
	{
		load_init_bias_config(linear, json);
	}
}

void to_json(nlohmann::json& json, const LinearConfig& linear)
{
	json["size"] = linear.size;
	json["use_bias"] = linear.use_bias;
	json["init_bias_type"] = linear.init_bias_type;
	json["init_bias"] = linear.init_bias;
	json["init_weight_type"] = linear.init_weight_type;
	json["init_weight"] = linear.init_weight;
}

void from_json(const nlohmann::json& json, FCLayerConfig& fc_layer_config)
{
	auto json_activation = json.find("activation");
	if (json_activation != json.end())
	{
		fc_layer_config = json_activation->get<Activation>();
		return;
	}

	FCLayerType type;
	type << required_input{json, "type"};
	switch (type)
	{
		case FCLayerType::kLinear: fc_layer_config = json.get<LinearConfig>(); break;
		case FCLayerType::kLayerNorm: fc_layer_config = json.get<LayerNormConfig>(); break;
		case FCLayerType::kLayerConnection: fc_layer_config = json.get<LayerConnectionConfig>(); break;
		default:
		{
			spdlog::error("[Config] The MLP layer type is not defined.");
			throw std::invalid_argument("The MLP layer type is not defined");
			break;
		}
	}
}

void to_json(nlohmann::json& json, const FCLayerConfig& fc_layer_config)
{
	std::visit(
		[&json](const auto& fc_layer) {
			using T = std::decay_t<decltype(fc_layer)>;
			if constexpr (std::is_same_v<Config::Activation, T>)
			{
				json["activation"] = fc_layer;
			}
			else
			{
				to_json(json, fc_layer);
			}
		},
		fc_layer_config);
}

void from_json(const nlohmann::json& json, LayerConnectionConfig& layer_connection)
{
	layer_connection.connection << required_input{json, "connection"};
	layer_connection.residual << optional_input{json, "residual"};
}

void to_json(nlohmann::json& json, const LayerConnectionConfig& layer_connection)
{
	json["layer_connection"] = layer_connection.connection;
	json["residual"] = layer_connection.residual;
}

void from_json(const nlohmann::json& json, FCConfig& fc)
{
	fc.layers << required_input{json, "layers"};
}

void to_json(nlohmann::json& json, const FCConfig& fc)
{
	json["layers"] = fc.layers;
}

void from_json(const nlohmann::json& json, MLPConfig& mlp)
{
	from_json(json, static_cast<FCConfig&>(mlp));
	mlp.name << optional_input{json, "name"};
}

void to_json(nlohmann::json& json, const MLPConfig& mlp)
{
	json["type"] = FeatureExtractorType::kMLP;
	to_json(json, static_cast<const FCConfig&>(mlp));
	json["name"] = mlp.name;
}

void from_json(const nlohmann::json& json, Conv2dConfig& conv)
{
	conv.in_channels << optional_input{json, "in_channels"};
	conv.out_channels << required_input{json, "out_channels"};
	conv.kernel_size << required_input{json, "kernel_size"};
	conv.stride << required_input{json, "stride"};
	conv.padding << optional_input{json, "padding"};
	load_init_weights_config(conv, json);
	load_init_bias_config(conv, json);
	conv.use_bias << optional_input{json, "use_bias"};
}

void to_json(nlohmann::json& json, const Conv2dConfig& conv)
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
}

void from_json(const nlohmann::json& json, ConvTranspose2dConfig& conv)
{
	from_json(json, *static_cast<Conv2dConfig*>(&conv));
	conv.output_padding << optional_input{json, "output_padding"};
}

void to_json(nlohmann::json& json, const ConvTranspose2dConfig& conv)
{
	to_json(json, *static_cast<const Conv2dConfig*>(&conv));
	json["type"] = LayerType::kConvTranspose2d;
	json["output_padding"] = conv.output_padding;
}

void from_json(const nlohmann::json& json, BatchNorm2dConfig& batch_norm)
{
	batch_norm.affine << optional_input{json, "affine"};
	batch_norm.eps << optional_input{json, "eps"};
	batch_norm.momentum << optional_input{json, "momentum"};
	batch_norm.track_running_stats << optional_input{json, "track_running_stats"};
}

void to_json(nlohmann::json& json, const BatchNorm2dConfig& batch_norm)
{
	json["type"] = LayerType::kBatchNorm2d;
	json["affine"] = batch_norm.affine;
	json["eps"] = batch_norm.eps;
	json["momentum"] = batch_norm.momentum;
	json["track_running_stats"] = batch_norm.track_running_stats;
}

void from_json(const nlohmann::json& json, LayerNormConfig& layer_norm)
{
	layer_norm.eps << optional_input{json, "eps"};
}

void to_json(nlohmann::json& json, const LayerNormConfig& layer_norm)
{
	json["type"] = LayerType::kLayerNorm;
	json["eps"] = layer_norm.eps;
}

void from_json(const nlohmann::json& json, MaxPool2dConfig& maxpool)
{
	maxpool.kernel_size << required_input{json, "kernel_size"};
	maxpool.stride << required_input{json, "stride"};
	maxpool.padding << optional_input{json, "padding"};
}

void to_json(nlohmann::json& json, const MaxPool2dConfig& maxpool)
{
	json["type"] = LayerType::kMaxPool2d;
	json["kernel_size"] = maxpool.kernel_size;
	json["stride"] = maxpool.stride;
	json["padding"] = maxpool.padding;
}

void from_json(const nlohmann::json& json, AvgPool2dConfig& avgpool)
{
	avgpool.kernel_size << required_input{json, "kernel_size"};
	avgpool.stride << required_input{json, "stride"};
	avgpool.padding << optional_input{json, "padding"};
}

void to_json(nlohmann::json& json, const AvgPool2dConfig& avgpool)
{
	json["type"] = LayerType::kAvgPool2d;
	json["kernel_size"] = avgpool.kernel_size;
	json["stride"] = avgpool.stride;
	json["padding"] = avgpool.padding;
}

void from_json(const nlohmann::json& json, AdaptiveAvgPool2dConfig& adaptavgpool)
{
	adaptavgpool.size << required_input{json, "size"};
}

void to_json(nlohmann::json& json, const AdaptiveAvgPool2dConfig& adaptavgpool)
{
	json["type"] = LayerType::kAdaptiveAvgPool2d;
	json["size"] = adaptavgpool.size;
}

void from_json(const nlohmann::json& json, ResBlock2dConfig& resblock)
{
	resblock.layers << optional_input{json, "layers"};
	resblock.kernel_size << optional_input{json, "kernel_size"};
	resblock.stride << optional_input{json, "stride"};
	resblock.normalise << optional_input{json, "normalise"};
	resblock.init_weight_type << optional_input{json, "init_weight_type"};
	load_init_weights_config(resblock, json);
}

void to_json(nlohmann::json& json, const ResBlock2dConfig& resblock)
{
	json["type"] = LayerType::kResBlock2d;
	json["layers"] = resblock.layers;
	json["kernel_size"] = resblock.kernel_size;
	json["stride"] = resblock.stride;
	json["normalise"] = resblock.normalise;
	json["init_weight_type"] = resblock.init_weight_type;
	json["init_weight"] = resblock.init_weight;
}

void from_json(const nlohmann::json& json, CNNLayerConfig& cnn_layer_config)
{
	auto json_activation = json.find("activation");
	if (json_activation != json.end())
	{
		cnn_layer_config = json_activation->get<Activation>();
		return;
	}

	LayerType type;
	type << required_input{json, "type"};
	switch (type)
	{
		case LayerType::kConv2d: cnn_layer_config = json.get<Conv2dConfig>(); break;
		case LayerType::kConvTranspose2d: cnn_layer_config = json.get<ConvTranspose2dConfig>(); break;
		case LayerType::kBatchNorm2d: cnn_layer_config = json.get<BatchNorm2dConfig>(); break;
		case LayerType::kLayerNorm: cnn_layer_config = json.get<LayerNormConfig>(); break;
		case LayerType::kMaxPool2d: cnn_layer_config = json.get<MaxPool2dConfig>(); break;
		case LayerType::kAvgPool2d: cnn_layer_config = json.get<AvgPool2dConfig>(); break;
		case LayerType::kAdaptiveAvgPool2d: cnn_layer_config = json.get<AdaptiveAvgPool2dConfig>(); break;
		case LayerType::kResBlock2d: cnn_layer_config = json.get<ResBlock2dConfig>(); break;
		default:
		{
			spdlog::error("[Config] The CNN layer type is not defined.");
			throw std::invalid_argument("The CNN layer type is not defined");
			break;
		}
	}
}

void to_json(nlohmann::json& json, const CNNLayerConfig& cnn_layer_config)
{
	std::visit(
		[&json](const auto& cnn_layer) {
			using T = std::decay_t<decltype(cnn_layer)>;
			if constexpr (std::is_same_v<Config::Activation, T>)
			{
				json["activation"] = cnn_layer;
			}
			else
			{
				to_json(json, cnn_layer);
			}
		},
		cnn_layer_config);
}

void from_json(const nlohmann::json& json, CNNConfig& cnn)
{
	cnn.layers << optional_input{json, "layers"};
}

void to_json(nlohmann::json& json, const CNNConfig& cnn)
{
	json["type"] = FeatureExtractorType::kCNN;
	json["layers"] = cnn.layers;
}

void from_json(const nlohmann::json& json, FeatureExtractorGroup& feature_extractor)
{
	FeatureExtractorType type;
	type << required_input{json, "type"};
	switch (type)
	{
		case FeatureExtractorType::kMLP: feature_extractor = json.get<MLPConfig>(); break;
		case FeatureExtractorType::kCNN: feature_extractor = json.get<CNNConfig>(); break;
	}
}

void to_json(nlohmann::json& json, const FeatureExtractorGroup& feature_extractor)
{
	std::visit([&](auto& feature_extractor) { to_json(json, feature_extractor); }, feature_extractor);
}

void from_json(const nlohmann::json& json, FeatureExtractorConfig& feature_extractor)
{
	feature_extractor.feature_groups = json.get<decltype(feature_extractor.feature_groups)>();
}

void to_json(nlohmann::json& json, const FeatureExtractorConfig& feature_extractor)
{
	json = feature_extractor.feature_groups;
}

void from_json(const nlohmann::json& json, ActorConfig& actor)
{
	from_json(json, static_cast<FCConfig&>(actor));
	actor.use_bias << optional_input{json, "use_bias"};
	load_init_weights_config(actor, json);
	if (actor.use_bias)
	{
		load_init_bias_config(actor, json);
	}
}

void to_json(nlohmann::json& json, const ActorConfig& actor)
{
	to_json(json, static_cast<const FCConfig&>(actor));
	json["use_bias"] = actor.use_bias;
	json["init_bias_type"] = actor.init_bias_type;
	json["init_bias"] = actor.init_bias;
	json["init_weight_type"] = actor.init_weight_type;
	json["init_weight"] = actor.init_weight;
}

void from_json(const nlohmann::json& json, ActorCriticConfig& actor_critic)
{
	actor_critic.use_shared_extractor << optional_input{json, "use_shared_extractor"};
	actor_critic.feature_extractor << required_input{json, "feature_extractor"};
	actor_critic.shared << optional_input{json, "shared"};
	actor_critic.actor << optional_input{json, "actor"};
	actor_critic.critic << optional_input{json, "critic"};
	actor_critic.predict_values << optional_input{json, "predict_values"};
	actor_critic.gru_hidden_size << optional_input{json, "gru_hidden_size"};
}

void to_json(nlohmann::json& json, const ActorCriticConfig& actor_critic)
{
	json["model_type"] = AgentPolicyModelType::kActorCritic;
	json["use_shared_extractor"] = actor_critic.use_shared_extractor;
	json["feature_extractor"] = actor_critic.feature_extractor;
	json["shared"] = actor_critic.shared;
	json["actor"] = actor_critic.actor;
	json["critic"] = actor_critic.critic;
	json["predict_values"] = actor_critic.predict_values;
	json["gru_hidden_size"] = actor_critic.gru_hidden_size;
}

void from_json(const nlohmann::json& json, SoftActorCriticConfig& sac)
{
	sac.feature_extractor << required_input{json, "feature_extractor"};
	sac.actor << optional_input{json, "actor"};
	sac.critic << optional_input{json, "critic"};
	sac.shared_feature_extractor << optional_input{json, "shared_feature_extractor"};
	sac.n_critics << optional_input{json, "n_critics"};
	sac.predict_values << optional_input{json, "predict_values"};
	sac.gru_hidden_size << optional_input{json, "gru_hidden_size"};
}

void to_json(nlohmann::json& json, const SoftActorCriticConfig& sac)
{
	json["model_type"] = AgentPolicyModelType::kSoftActorCritic;
	json["feature_extractor"] = sac.feature_extractor;
	json["actor"] = sac.actor;
	json["critic"] = sac.critic;
	json["shared_feature_extractor"] = sac.shared_feature_extractor;
	json["n_critics"] = sac.n_critics;
	json["predict_values"] = sac.predict_values;
	json["gru_hidden_size"] = sac.gru_hidden_size;
}

void from_json(const nlohmann::json& json, QNetModelConfig& dqn_model)
{
	dqn_model.feature_extractor << required_input{json, "feature_extractor"};
	dqn_model.q_net << required_input{json, "q_net"};
	dqn_model.gru_hidden_size << optional_input{json, "gru_hidden_size"};
}

void to_json(nlohmann::json& json, const QNetModelConfig& dqn_model)
{
	json["model_type"] = AgentPolicyModelType::kQNet;
	json["feature_extractor"] = dqn_model.feature_extractor;
	json["q_net"] = dqn_model.q_net;
	json["gru_hidden_size"] = dqn_model.gru_hidden_size;
}

void from_json([[maybe_unused]] const nlohmann::json& json, [[maybe_unused]] RandomConfig& random)
{
}

void to_json([[maybe_unused]] nlohmann::json& json, [[maybe_unused]] const RandomConfig& random)
{
	json["model_type"] = AgentPolicyModelType::kRandom;
}

void from_json(const nlohmann::json& json, ModelConfig& model)
{
	AgentPolicyModelType model_type;
	model_type << required_input{json, "model_type"};
	switch (model_type)
	{
		case AgentPolicyModelType::kRandom: model = json.get<RandomConfig>(); break;
		case AgentPolicyModelType::kActorCritic: model = json.get<ActorCriticConfig>(); break;
		case AgentPolicyModelType::kSoftActorCritic: model = json.get<SoftActorCriticConfig>(); break;
		case AgentPolicyModelType::kQNet: model = json.get<QNetModelConfig>(); break;
		case AgentPolicyModelType::kMuZero: model = json.get<MuZero::ModelConfig>(); break;
	}
}

void to_json(nlohmann::json& json, const ModelConfig& model)
{
	std::visit([&](auto& model) { to_json(json, model); }, model);
}

void from_json(const nlohmann::json& json, Config::AgentBase& agent)
{
	agent.env_count << optional_input{json, "env_count"};
	agent.cuda_devices << optional_input{json, "cuda_devices"};

	agent.model << required_input{json, "model"};
	agent.model_type << required_input{json.at("model"), "model_type"};

	agent.train_algorithm << optional_input{json, "train_algorithm"};
	agent.train_algorithm_type << optional_input{json.at("train_algorithm"), "train_algorithm_type"};

	agent.rewards << optional_input{json, "rewards"};

	agent.timestep_save_period << optional_input{json, "timestep_save_period"};
	agent.checkpoint_save_period << optional_input{json, "checkpoint_save_period"};
	agent.eval_period << optional_input{json, "eval_period"};
}

void to_json(nlohmann::json& json, const Config::AgentBase& agent)
{
	json["env_count"] = agent.env_count;
	json["cuda_devices"] = agent.cuda_devices;

	json["model"] = agent.model;

	json["train_algorithm"] = agent.train_algorithm;

	json["rewards"] = agent.rewards;

	json["timestep_save_period"] = agent.timestep_save_period;
	json["checkpoint_save_period"] = agent.checkpoint_save_period;
	json["eval_period"] = agent.eval_period;
}

void from_json([[maybe_unused]] const nlohmann::json& json, [[maybe_unused]] Config::InteractiveAgent& agent)
{
}

void to_json([[maybe_unused]] nlohmann::json& json, [[maybe_unused]] const Config::InteractiveAgent& agent)
{
}

void from_json(const nlohmann::json& json, Config::OnPolicyAgent& agent)
{
	from_json(json, static_cast<Config::AgentBase&>(agent));
	agent.asynchronous_env << optional_input{json, "asynchronous_env"};
}

void to_json(nlohmann::json& json, const Config::OnPolicyAgent& agent)
{
	to_json(json, static_cast<const Config::AgentBase&>(agent));
	json["asynchronous_env"] = agent.asynchronous_env;
}

void from_json(const nlohmann::json& json, Config::OffPolicyAgent& agent)
{
	from_json(json, static_cast<Config::AgentBase&>(agent));
}

void to_json(nlohmann::json& json, const Config::OffPolicyAgent& agent)
{
	to_json(json, static_cast<const Config::AgentBase&>(agent));
}

void from_json(const nlohmann::json& json, Config::MCTSAgent& agent)
{
	from_json(json, static_cast<Config::AgentBase&>(agent));
	agent.root_dirichlet_alpha << optional_input{json, "root_dirichlet_alpha"};
	agent.root_exploration_fraction << optional_input{json, "root_exploration_fraction"};
	agent.num_simulations << optional_input{json, "num_simulations"};
	agent.pb_c_base << optional_input{json, "pb_c_base"};
	agent.pb_c_init << optional_input{json, "pb_c_init"};
	agent.gamma << optional_input{json, "gamma"};
	agent.temperature << optional_input{json, "temperature"};
}

void to_json(nlohmann::json& json, const Config::MCTSAgent& agent)
{
	to_json(json, static_cast<const Config::AgentBase&>(agent));
	json["root_dirichlet_alpha"] = agent.root_dirichlet_alpha;
	json["root_exploration_fraction"] = agent.root_exploration_fraction;
	json["num_simulations"] = agent.num_simulations;
	json["pb_c_base"] = agent.pb_c_base;
	json["pb_c_init"] = agent.pb_c_init;
	json["gamma"] = agent.gamma;
	json["temperature"] = agent.temperature;
}

void to_json(nlohmann::json& json, const Config::Agent& agent)
{
	std::visit([&](auto& agent) { to_json(json, agent); }, agent);
}

void from_json(const nlohmann::json& json, Config::Agent& agent)
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
			case AgentPolicyModelType::kMuZero: agent = json.get<Config::MCTSAgent>(); break;
		}
	}
	else
	{
		agent = json.get<Config::AgentBase>();
	}
}

namespace MuZero
{

void from_json(const nlohmann::json& json, DynamicsNetwork& dyn_net)
{
	dyn_net.num_blocks << optional_input{json, "num_blocks"};
	dyn_net.num_channels << optional_input{json, "num_channels"};
	dyn_net.reduced_channels_reward << optional_input{json, "reduced_channels_reward"};
	dyn_net.resblock << optional_input{json, "resblock"};
	dyn_net.fc_reward << required_input{json, "fc_reward"};
	dyn_net.fc_dynamics << optional_input{json, "fc_dynamics"};
}

void to_json(nlohmann::json& json, const DynamicsNetwork& dyn_net)
{
	json["num_blocks"] = dyn_net.num_blocks;
	json["num_channels"] = dyn_net.num_channels;
	json["reduced_channels_reward"] = dyn_net.reduced_channels_reward;
	json["resblock"] = dyn_net.resblock;
	json["fc_reward"] = dyn_net.fc_reward;
	json["fc_dynamics"] = dyn_net.fc_dynamics;
}

void from_json(const nlohmann::json& json, PredictionNetwork& pred_net)
{
	pred_net.num_blocks << optional_input{json, "num_blocks"};
	pred_net.num_channels << optional_input{json, "num_channels"};
	pred_net.reduced_channels_value << optional_input{json, "reduced_channels_value"};
	pred_net.reduced_channels_policy << optional_input{json, "reduced_channels_policy"};
	pred_net.resblock << optional_input{json, "resblock"};
	pred_net.fc_value << required_input{json, "fc_value"};
	pred_net.fc_policy << required_input{json, "fc_policy"};
}

void to_json(nlohmann::json& json, const PredictionNetwork& pred_net)
{
	json["num_blocks"] = pred_net.num_blocks;
	json["num_channels"] = pred_net.num_channels;
	json["reduced_channels_value"] = pred_net.reduced_channels_value;
	json["reduced_channels_policy"] = pred_net.reduced_channels_policy;
	json["resblock"] = pred_net.resblock;
	json["fc_value"] = pred_net.fc_value;
	json["fc_policy"] = pred_net.fc_policy;
}

void from_json(const nlohmann::json& json, ModelConfig& muzero_config)
{
	muzero_config.representation_network << required_input{json, "representation_network"};
	muzero_config.dynamics_network << required_input{json, "dynamics_network"};
	muzero_config.prediction_network << required_input{json, "prediction_network"};
	muzero_config.support_size << optional_input{json, "support_size"};
	muzero_config.stacked_observations << optional_input{json, "stacked_observations"};
}

void to_json(nlohmann::json& json, const ModelConfig& muzero_config)
{
	json["model_type"] = AgentPolicyModelType::kMuZero;
	json["representation_network"] = muzero_config.representation_network;
	json["dynamics_network"] = muzero_config.dynamics_network;
	json["prediction_network"] = muzero_config.prediction_network;
	json["support_size"] = muzero_config.support_size;
	json["stacked_observations"] = muzero_config.stacked_observations;
}

void from_json(const nlohmann::json& json, TrainConfig& alg_muzero)
{
	from_json(json, static_cast<MCTSAlgorithm&>(alg_muzero));
	alg_muzero.value_loss_weight << optional_input{json, "value_loss_weight"};
	alg_muzero.weight_decay << optional_input{json, "weight_decay"};
	alg_muzero.epsilon << optional_input{json, "epsilon"};
	alg_muzero.momentum << optional_input{json, "momentum"};
	alg_muzero.optimiser << optional_input{json, "optimiser"};
}

void to_json(nlohmann::json& json, const TrainConfig& alg_muzero)
{
	json["train_algorithm_type"] = TrainAlgorithmType::kMuZero;
	to_json(json, static_cast<const MCTSAlgorithm&>(alg_muzero));
	json["value_loss_weight"] = alg_muzero.value_loss_weight;
	json["weight_decay"] = alg_muzero.weight_decay;
	json["epsilon"] = alg_muzero.epsilon;
	json["momentum"] = alg_muzero.momentum;
	json["optimiser"] = alg_muzero.optimiser;
}

} // namespace MuZero

} // namespace drla::Config
