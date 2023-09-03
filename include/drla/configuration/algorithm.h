#pragma once

#include <cstddef>
#include <string>
#include <variant>
#include <vector>

namespace drla
{

/// @brief The training algorithm type
enum class TrainAlgorithmType
{
	kNone,
	kA2C,
	kPPO,
	kDQN,
	kSAC,
	kMuZero,
};

/// @brief The schedule type for learning rate decay configuration
enum class LearningRateScheduleType
{
	kConstant,
	kLinear,
	kExponential,
};

/// @brief The optimiser type to use for training
enum class OptimiserType
{
	kAdam,
	kSGD,
	kRMSProp,
};

namespace Config
{

/// @brief Common optimiser config
struct OptimiserBase
{
	// The learning rate to use for training
	double learning_rate = 1e-4;
	// The minimum learning rate to use for training. Only applicable for non constant
	double learning_rate_min = 0;
	// Decay the learning rate based on a specific schedule type
	LearningRateScheduleType lr_schedule_type = LearningRateScheduleType::kConstant;
	// The rate at which the lr is decayed.
	// For linear lr_scale = 1 - lr_decay_rate * progress
	// For expenenatial lr_scale = exp(-0.5 * PI * lr_decay_rate * progress)
	double lr_decay_rate = 1.0;
	// Gradient value clipping. Clips gradients to a maximum specified value. 0 disables clipping.
	double grad_clip = 0.0;
	// Gradient normalised clipping. Clips gradients to the specified maximum normalised value. 0 disables clipping.
	double grad_norm_clip = 0.0;
};

/// @brief Adam optimiser specific config
struct OptimiserAdam : public OptimiserBase
{
	// Controls the smoothness of gradient updates. Can help escape local minima and better converge towards global
	// optima.
	double epsilon = 1e-8;
	// L2 weights regularization. Can help prevent overfitting by adding a penalty term to the loss function that
	// encourages smaller weight values.
	double weight_decay = 0;
};

/// @brief SGD optimiser specific config
struct OptimiserSGD : public OptimiserBase
{
	// Improve convergence speed and accuracy by incorporating past gradients into the update step
	double momentum = 0;
	// L2 weights regularization. Can help prevent overfitting by adding a penalty term to the loss function that
	// encourages smaller weight values.
	double weight_decay = 0;
	// Reduces overshooting and oscillations in the learning process by adding a small resistance term to the update rule
	double dampening = 0;
};

/// @brief RMSProp optimiser specific config
struct OptimiserRMSProp : public OptimiserBase
{
	// Controls how much each parameter update relies on past gradients relative to current ones
	double alpha = 0.99;
	// Controls the smoothness of gradient updates. Can help escape local minima and better converge towards global
	// optima.
	double epsilon = 1e-8;
	// L2 weights regularization. Can help prevent overfitting by adding a penalty term to the loss function that
	// encourages smaller weight values.
	double weight_decay = 0;
	// Improve convergence speed and accuracy by incorporating past gradients into the update step
	double momentum = 0;
};

/// @brief The optimiser to use for training
using Optimiser = std::variant<OptimiserAdam, OptimiserSGD, OptimiserRMSProp>;

/// @brief Config common to all algorithm types
struct TrainAlgorithm
{
	// The total number of training timesteps to run
	int total_timesteps = 100'000;
	// The step number to start training at. This is used to resume training
	int start_timestep = 0;
	// The max steps to run the agent when performing evaluation
	int eval_max_steps = 0;
	// Options to run evaluation model deterministically
	bool eval_determinisic = true;
};

/// @brief On policy algorithm configuration. This assumes an actor critic based model.
struct OnPolicyAlgorithm : public TrainAlgorithm
{
	// The size of the rollout buffer for each training step.
	int horizon_steps = 128;
	// The ratio the policy loss contributes to the total loss
	double policy_loss_coef = 1.0;
	// The ratio the value loss contributes to the total loss
	double value_loss_coef = 0.5;
	// The ratio the entropy loss contributes to the total loss
	double entropy_coef = 0.01;

	// The discount factor
	std::vector<float> gamma = {0.99F};
	// Factor for trade-off of bias vs variance for Generalized Advantage Estimator. Equivalent to classic advantage when
	// set to 1.
	double gae_lambda = 0.95;
};

/// @brief Off policy algorithm configuration. This assumes a Q network based model.
struct OffPolicyAlgorithm : public TrainAlgorithm
{
	// The number of environment steps to use in a train step.
	int horizon_steps = 128;
	// The size of the replay buffer
	int buffer_size = 1'000'000;
	// The number of samples to use in a train update step
	int batch_size = 256;
	// How many steps of the model to collect transitions for before learning starts
	int learning_starts = 100;
	// The discount factor
	std::vector<float> gamma = {0.99F};
	// The ratio to move the target network to the current network
	double tau = 1.0;
	// The number of gradient steps to perform in an update step
	int gradient_steps = 1;
	// Number of gradient steps to delay updating the target network
	int target_update_interval = 1;
	// The ammount of prioritised experience replay to use
	float per_alpha = 1.0F;
};

/// @brief DQN training algorithm specific configuration
struct DQN : public OffPolicyAlgorithm
{
	// The fraction of entire training period over which the exploration rate is reduced
	double exploration_fraction = 0.1;
	// The initial exploration rate to start training with
	double exploration_init = 1.0;
	// The minimum exploration rate to decay to during training
	double exploration_final = 0.05;
	// The training optimiser. Defaults to Adam
	Optimiser optimiser = OptimiserAdam{{2e-4, 0, LearningRateScheduleType::kConstant, 1.0, 0, 0.5}};
};

/// @brief SAC training algorithm specific configuration
struct SAC : public OffPolicyAlgorithm
{
	// The ratio the actor loss contributes to the total loss
	double actor_loss_coef = 1.0;
	// The ratio the value loss contributes to the total loss
	double value_loss_coef = 0.5;
	// The coefficient for scaling the entropy target
	double target_entropy_scale = 0.89;
	// The training actor optimiser. Defaults to Adam
	Optimiser actor_optimiser = OptimiserAdam{{3e-4, 1e-6, LearningRateScheduleType::kLinear, 1.0, 0, 0.5}};
	// The training critic optimiser. Defaults to Adam
	Optimiser critic_optimiser = OptimiserAdam{{3e-4, 1e-6, LearningRateScheduleType::kLinear, 1.0, 0, 0.5}};
	// The training entropy coefitient optimiser. Defaults to Adam
	Optimiser ent_coef_optimiser = OptimiserAdam{{3e-4, 1e-6, LearningRateScheduleType::kLinear, 1.0, 0, 0.5}};
};

/// @brief A2C training algorithm specific configuration
struct A2C : public OnPolicyAlgorithm
{
	// Normalise the advantages. Use this if the reward amounts vary significantly over an episode.
	bool normalise_advantage = true;
	// The training optimiser. Defaults to RMSProp
	Optimiser optimiser = OptimiserRMSProp{{2.5e-4, 1e-6, LearningRateScheduleType::kLinear, 1.0, 0, 0.5}};
};

/// @brief PPO training algorithm specific configuration
struct PPO : public OnPolicyAlgorithm
{
	// Normalise the advantages. Use this if the reward amounts vary significantly over an episode.
	bool normalise_advantage = true;
	// Clip the difference between the old and new policy
	double clip_range_policy = 0.2;
	// Enable clipping the value function
	bool clip_vf = false;
	// Clip the difference between the old and new value function
	double clip_range_vf = 0.2;
	// The number of iterations over the rollout buffer
	int num_epoch = 4;
	// Divide the samples from the rollout buffer into mini batches
	int num_mini_batch = 4;
	// Limit the Kulback Liebler divergence to a target range to avoid large updates
	double kl_target = 0.01;
	// The training optimiser. Defaults to Adam
	Optimiser optimiser = OptimiserAdam{{2.5e-4, 1e-6, LearningRateScheduleType::kLinear, 1.0, 0, 0.5}};
};

/// @brief MCTS based training algorithms. This assumes episodic prioritised replay buffer and MCTS algorithm is used
/// for action generation.
struct MCTSAlgorithm : public TrainAlgorithm
{
	// The number of episodes to keep in the PER buffer
	int buffer_size = 1'000'000;
	// The minimum number of episodes before training starts
	int start_buffer_size = 10;
	// The number of samples to use in a train update step
	int batch_size = 256;
	// Don't start reanalysing until the min number of training steps are performed
	int min_reanalyse_train_steps = 10;
	// Don't start reanalysing until the min size of the buffer is reached
	int min_reanalyse_buffer_size = 100;
	// The ammount of prioritised experience replay to use
	float per_alpha = 1.0F;

	// The td steps for value target calculation
	int td_steps = 5;
	// The unroll steps for training
	int unroll_steps = 10;

	// The temperature to use for action selection for a training step range. i.e. {{100e3,1.0},{200e3,0.5}} means between
	// 0 to 100e3 use 1.0 and between 100e3 and 200e3 use 0.5, otherwise use default
	std::vector<std::pair<int, float>> temperature_step;
	// The ratio of self play episodes to train steps. The agent will delay a training while the self play episodes are
	// less than train_ratio * train steps
	double train_ratio = 0;

	// The GPU indexs to use for self play. A single -1 entry will use all available GPUs. Leave empty to force CPU use.
	std::vector<int> self_play_gpus = {-1};
	// The GPU indexs to use for training. A single -1 entry will use all available GPUs. Leave empty to force CPU use.
	std::vector<int> train_gpus = {-1};
};

namespace MuZero
{
/// @brief MuZero train algorithm configuration.
struct TrainConfig : public MCTSAlgorithm
{
	// Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix
	// Reanalyze)
	float value_loss_weight = 0.25F;
	// The training optimiser. Defaults to SGD
	Optimiser optimiser = OptimiserSGD{{3e-3, 1e-4, LearningRateScheduleType::kExponential, 1.0}, 0.9, 1e-4};
};
} // namespace MuZero

/// @brief The agent training configuration
using AgentTrainAlgorithm = std::variant<A2C, PPO, DQN, SAC, MuZero::TrainConfig>;

} // namespace Config

} // namespace drla
