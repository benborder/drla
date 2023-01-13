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
/// @brief The type of rewards of the environment
using RewardTypes = std::vector<std::string>;

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
};

/// @brief The result type from a training step
enum class TrainResultType
{
	kLoss,
	kValueLoss,
	kPolicyLoss,
	kEntropyLoss,
	kClipFraction,
	kKLDivergence,
	kLearningRate,
	kExplainedVariance,
	kExploration,
};

/// @brief Converts a TrainResultType to a string
/// @return A string of the TrainResultType
inline std::string get_result_type_name(TrainResultType type)
{
	switch (type)
	{
		case TrainResultType::kValueLoss: return "value_loss";
		case TrainResultType::kPolicyLoss: return "policy_loss";
		case TrainResultType::kEntropyLoss: return "entropy_loss";
		case TrainResultType::kClipFraction: return "clip_fraction";
		case TrainResultType::kKLDivergence: return "kl_divergence";
		case TrainResultType::kLoss: return "loss";
		case TrainResultType::kExplainedVariance: return "explained_variance";
		case TrainResultType::kLearningRate: return "learning_rate";
		case TrainResultType::kExploration: return "exploration";
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
};

/// @brief Output data from a single step in an environment
struct StepData
{
	// The index of environment the step is executed in
	unsigned int env = 0;
	// The step number
	int step = 0;

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
