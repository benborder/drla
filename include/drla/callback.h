#pragma once
#include "drla/configuration.h"
#include "drla/types.h"

#include <torch/torch.h>

#include <chrono>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace drla
{

/// @brief Initialisation data from the agent. See `train_init` callback.
struct InitData
{
	// The configuration of the environment
	EnvironmentConfiguration env_config;

	// The reward shape used for the model
	int reward_shape;

	// The initial output of each environment
	std::vector<EnvStepData> env_output;
};

/// @brief Data from the current train timestep update. See `train_update` callback.
struct TrainUpdateData
{
	// The total fps from all envs
	double fps = 0.0;

	// The fps of the slowest env
	double fps_env = 0.0;

	// The number of training timesteps
	int timestep = 0;

	// The total number of env steps taken
	int global_steps = 0;

	// Stats from a single training update step
	std::vector<UpdateResult> update_data;

	// The time it took to perform a single training update step
	std::chrono::duration<double> update_duration;

	// The time taken to run the envs
	std::chrono::duration<double> env_duration;
};

/// @brief Config for the agent for an environment when reset
struct AgentResetConfig
{
	// When true the agent should stop.
	bool stop = false;
	// When true the agent captures raw observations for the episode.
	bool raw_capture = false;
};

/// @brief An interface defining callbacks the agent provides
class AgentCallbackInterface
{
public:
	/// @brief Callback prior to training start when training is initialised
	/// @param data The relevant initialised data used for training
	virtual void train_init(const InitData& data) = 0;

	/// @brief The callback after each train update timestep has been completed
	/// @param data Contains statistics and information of the train update timestep
	virtual void train_update(const TrainUpdateData& data) = 0;

	/// @brief The callback after an environment is reset
	/// @param data The data after a reset including the result, state and observations
	/// @return configuration options for the agent for the current episode
	virtual AgentResetConfig env_reset(const StepData& data) = 0;

	/// @brief The callback after a step is enacted in the environment
	/// @param data The step data including the result, state and observations
	/// @return true if the agent should stop, false if the agent should continue
	virtual bool env_step(const StepData& data) = 0;

	/// @brief The callback to determine the action to perform in the environment for interactive mode. This should block
	/// until input is recieved.
	/// @return The action for the environment to perform. The output shape should be [num_envs, <action_dims>]
	virtual torch::Tensor interactive_step() = 0;

	/// @brief Is called when saving the config and model
	/// @param steps The number of train steps performed
	/// @param path The path the model is saved to
	virtual void save(int steps, const std::filesystem::path& path) = 0;
};

} // namespace drla
