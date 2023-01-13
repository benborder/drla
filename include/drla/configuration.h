#pragma once

#include "configuration/algorithm.h"
#include "configuration/model.h"

#include <cstddef>
#include <string>
#include <variant>
#include <vector>

namespace drla
{

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
	// Enable cuda if available
	bool use_cuda = true;

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
};

/// @brief Off Policy agent specific configuration
struct OffPolicyAgent : public AgentBase
{
	// When false each environment thread steps at the same time and combined model inference is performed. Use this when
	// environments have a consistent step time (low variance).
	// When true each environment thread steps independently and individual model inference is performed. Use this when
	// environments step time is highly inconsistent (high variance).
	bool asynchronous_env = false;
};

/// @brief The agent configuration
using Agent = std::variant<AgentBase, InteractiveAgent, OnPolicyAgent, OffPolicyAgent>;

} // namespace Config

} // namespace drla
