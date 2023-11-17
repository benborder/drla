#pragma once

#include "configuration/algorithm.h"
#include "configuration/model.h"

#include <cstddef>
#include <string>
#include <variant>
#include <vector>

namespace drla
{

/// @brief The opponent agent to use when running the environment
enum class OpponentType
{
	kSelf,	 // The opponent agent uses the same agent
	kExpert, // The opponent agent uses the environments expert agent
};

namespace Config
{

/// @brief Post environment reward modifications
struct Rewards
{
	// Combine all rewards into a single value for each step
	bool combine_rewards = false;

	// Clamp rewards to the minimum specified. 0 implies no clamping
	double reward_clamp_min = 0;
	// Clamp rewards to the maximum specified. 0 implies no clamping
	double reward_clamp_max = 0;
};

/// @brief Agent configuration common to all agent types
struct AgentBase
{
	// The number of environments to run concurrently
	int env_count = 8;
	// The cuda devices to use. Leaving empty uses none, a single -1 uses all available cuda GPUs
	std::vector<int> cuda_devices = {-1};

	// The training algorithm type. Is only required if training and agent.
	TrainAlgorithmType train_algorithm_type = TrainAlgorithmType::kNone;
	// The training algorithm configuration. Can be empty if not training.
	AgentTrainAlgorithm train_algorithm;

	// The model type to use, must match
	AgentPolicyModelType model_type;
	// Model configuration. Defaults to random.
	ModelConfig model = RandomConfig();

	// Configuration for rewards input into the agent.
	Rewards rewards;

	// Save the agent every n timesteps
	int timestep_save_period = 100;
	// Save a checkpoint of state every n timesteps, 0 disables checkpoint saving
	int checkpoint_save_period = 0;
	// Evaluate the agent every n timesteps during training. 0 disables evaluation.
	int eval_period = 0;
};

/// @brief Interactive agent specific configuration
struct InteractiveAgent : public AgentBase
{
};

/// @brief On Policy agent specific configuration
struct OnPolicyAgent : public AgentBase
{
	// When false each environment thread steps at the same time and combined model inference is performed. Use this when
	// environments have a consistent step time (low variance).
	// When true each environment thread steps independently and individual model inference is performed. Use this when
	// environments step time is highly inconsistent (high variance).
	bool asynchronous_env = false;
	// When enabled, the maximum number of envs that can run concurrently is clamped to the number of available hardware
	// threads
	bool clamp_concurrent_envs = true;
};

/// @brief Off Policy agent specific configuration
struct OffPolicyAgent : public AgentBase
{
	// When false each environment thread steps at the same time and combined model inference is performed. Use this when
	// environments have a consistent step time (low variance).
	// When true each environment thread steps independently and individual model inference is performed. Use this when
	// environments step time is highly inconsistent (high variance).
	bool asynchronous_env = false;
	// When enabled, the maximum number of envs that can run concurrently is clamped to the number of available hardware
	// threads
	bool clamp_concurrent_envs = true;
};

/// @brief MCTS agent specific configuration
struct MCTSAgent : public AgentBase
{
	// Root prior exploration noise
	double root_dirichlet_alpha = 0.25;
	// Root prior exploration noise
	double root_exploration_fraction = 0.25;
	// Number of future moves self-simulated
	int num_simulations = 50;
	// UCB formula c_base constant
	double pb_c_base = 19652;
	// UCB formula c_init constant
	double pb_c_init = 1.25;
	// The discount factor for value calculation
	std::vector<float> gamma = {0.997F};
	// The number of moves before dropping the temperature to 0 (ie selecting the best action). If 0, the provided
	// temperature is used every time.
	int temperature_threshold = 0;
	// Determines how greedy the action selection is for MCTS based agents. 0 is maximally greedy and the larger tha value
	// the more random.
	float temperature = 0.0F;
	// The index for the agent to use in the environment (only relevent for multi actor environments). When < 0 a random
	// index is assigned.
	int actor_index = 0;
	// The opponent agent to use for multi actor environments.
	OpponentType opponent_type = OpponentType::kExpert;
};

/// @brief Hybrid agent specific configuration
struct HybridAgent : public AgentBase
{
	// The discount factor for value calculation
	std::vector<float> gamma = {0.997F};
};

/// @brief The agent configuration
using Agent = std::variant<AgentBase, InteractiveAgent, OnPolicyAgent, OffPolicyAgent, MCTSAgent, HybridAgent>;

} // namespace Config

} // namespace drla
