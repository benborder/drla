#pragma once

#include <torch/torch.h>

#include <any>
#include <string>
#include <vector>

namespace drla
{

/// @brief The type of action space of an environment
enum class ActionSpaceType
{
	// Individual actions, i.e. move
	kDiscrete,
	// Continuous action space rescaled and bound to a range
	kBox,
	// Multiple on/off actions which can form any combination
	kMultiBinary,
	// The same as discrete, but multiple actions can be performed simultaneously
	kMultiDiscrete,
};

/// @brief Describes the action type and shape of the environment
struct ActionSpace
{
	// The type of action. This falls in 2 primary categories: discrete and continuos
	ActionSpaceType type;
	// The number of actions in the action space. For discrete action types, each dimenion is a simultaneous action. For
	// continous action types, only a single dimension is used.
	std::vector<int64_t> shape;
};

/// @brief Indicates if the action space is of a discrete nature
/// @return True if the action space is discrete, false otherwise
inline bool is_action_discrete(const ActionSpace& action_space)
{
	return action_space.type == ActionSpaceType::kDiscrete || action_space.type == ActionSpaceType::kMultiDiscrete;
}

/// @brief A vector of observation tensors. Each vector item group can have a unique shape and data type.
using Observations = std::vector<torch::Tensor>;
/// @brief The observation shape of the environment
using ObservationShapes = std::vector<std::vector<int64_t>>;
/// @brief The data type of each. For example, float, int, bool etc.
using ObservationDataTypes = std::vector<torch::ScalarType>;
/// @brief The hidden states for recurrent models
using HiddenStates = std::vector<torch::Tensor>;
/// @brief The shape of the models internal state
using StateShapes = std::vector<int64_t>;
/// @brief The type of rewards of the environment
using RewardTypes = std::vector<std::string>;
/// @brief The list of discrete actions available to perform in the environment.
using ActionSet = std::vector<int>;
/// @brief Activation function alias
using ActivationFunction = std::function<torch::Tensor(const torch::Tensor&)>;

/// @brief Describes the configuration of the environment. Shapes and data types for observations, actions space and
/// reward types.
struct EnvironmentConfiguration
{
	// The name/type of the environment
	std::string name;
	// The shape of each observation group
	ObservationShapes observation_shapes;
	// The data types of the environment (int, float, etc) for each observation group
	ObservationDataTypes observation_dtypes;
	// The action type and shape the environment uses
	ActionSpace action_space;
	// The types of rewards the environment returns
	RewardTypes reward_types;
	// The action set for the environment
	ActionSet action_set;
	// The number of actors this environment is configured for
	int num_actors;
};

/// @brief The result type from a training step
enum class TrainResultType
{
	kLoss,
	kValueLoss,
	kPolicyLoss,
	kRewardLoss,
	kEntropyLoss,
	kClipFraction,
	kKLDivergence,
	kLearningRate,
	kExplainedVariance,
	kExploration,
	kEntropyCoeficients,
	kReanalyseCount,
};

/// @brief Converts a TrainResultType to a string
/// @return A string of the TrainResultType
inline std::string get_result_type_name(TrainResultType type)
{
	switch (type)
	{
		case TrainResultType::kLoss: return "loss";
		case TrainResultType::kValueLoss: return "value_loss";
		case TrainResultType::kPolicyLoss: return "policy_loss";
		case TrainResultType::kRewardLoss: return "reward_loss";
		case TrainResultType::kEntropyLoss: return "entropy_loss";
		case TrainResultType::kClipFraction: return "clip_fraction";
		case TrainResultType::kKLDivergence: return "kl_divergence";
		case TrainResultType::kLearningRate: return "learning_rate";
		case TrainResultType::kExplainedVariance: return "explained_variance";
		case TrainResultType::kExploration: return "exploration";
		case TrainResultType::kEntropyCoeficients: return "entropy_coeficients";
		case TrainResultType::kReanalyseCount: return "reanalyse_count";
		default: return "";
	}
}

/// @brief The output from the agent's model prediction
struct PredictOutput
{
	// The actions the model determined should be taken
	torch::Tensor action;
	// The values output from a model forward pass (for supported models)
	torch::Tensor values = {};
	// The log of the probalitity of taking each action, required for importance sampling in a train update (for supported
	// models)
	torch::Tensor action_log_probs = {};
	// The policy logits of taking each action (for supported models)
	torch::Tensor policy = {};
	// The reward from model prediction (for supported models)
	torch::Tensor reward = {};
	// The predicted state (from supported models)
	HiddenStates state = {};
};

/// @brief The input for the agent's model prediction
struct ModelInput
{
	// The input observations to pass to the model. The observations must be on the same device as the model.
	Observations observations;
	// Previous model output
	PredictOutput prev_output = {};
	// Use a deterministic forward pass through the model to determine the action if true. Otherwise a stochastic policy
	// gradient is used to determine the action. This option is only relevant for policy gradient based models.
	bool deterministic = true;
};

/// @brief A batch sampled from a buffer
struct Batch
{
	// episode id and episode step index
	std::vector<std::pair<int, int>> indicies;
	Observations observation;
	torch::Tensor reward;
	torch::Tensor values;
	torch::Tensor policy;
	torch::Tensor action;
	torch::Tensor non_terminal;
	torch::Tensor weight;
	torch::Tensor gradient_scale;
};

/// @brief Training update result data
struct UpdateResult
{
	// The result type
	TrainResultType type;
	// The result value
	double value;
};

/// @brief The state of the environment
struct State
{
	// Environment specific state
	std::any env_state;
	// The current number of steps in the environment
	int step = 0;
	// Indicates an episode has ended
	bool episode_end = false;
	// The maximum number of steps the episode can be run for
	int max_episode_steps = 0;
};

/// @brief The result of a step or reset in an environment
struct EnvStepData
{
	// The observation from the environment after a step or reset
	Observations observation;
	// The raw reward recieved from an environment step or reset
	torch::Tensor reward;
	// The state of the environment after a step or reset
	State state;
	// The list of discrete actions that are possible to perform given the current environment state. Only applicable for
	// environments with discrete actions.
	ActionSet legal_actions = {};
	// The index of the agent to take the next turn given the current state. Only relevant for multi agent environments
	// turn based environments.
	int turn_index = 0;
};

/// @brief The input for model prediction
struct PredictInput
{
	// The input data from the environment
	EnvStepData env_data;
	// The action taken in the previous step. Not required for model free agents.
	torch::Tensor action = {};
};

/// @brief Output agent/model data from a single step in an environment
struct StepData
{
	// The index of environment the step is executed in
	unsigned int env = 0;
	// The step number
	int step = 0;
	// The agent is running in evaluation mode (only relevant in training)
	bool eval_mode = false;

	// The result of a step or reset in an environment
	EnvStepData env_data;

	// Action prediction result
	PredictOutput predict_result;

	// The clamped/scaled reward gained when executing this step
	torch::Tensor reward;
	// The raw output observation from the step. Typically used for display/debug purposes
	Observations raw_observation;
};

} // namespace drla
