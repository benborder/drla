#pragma once

#include "drla/configuration.h"
#include "drla/types.h"

#include <torch/torch.h>

namespace drla
{

/// @brief An interface for an environment, enabling it to be used by the agent
class Environment
{
public:
	/// @brief Gets the environment configuration which describes the observation shape and data types, action space and
	/// reward types
	/// @return The environment configuration
	virtual EnvironmentConfiguration get_configuration() const = 0;

	/// @brief Steps the environment and performs the specified action
	/// @param action The action to perform in the environment
	/// @return The result of the step
	virtual EnvStepData step(torch::Tensor action) = 0;

	/// @brief Resets the environment and sets the initial state
	/// @param initial_state The initial state of the environment to reset to
	/// @return The initial state after reset
	virtual EnvStepData reset(const State& initial_state) = 0;

	/// @brief Returns the unprocessed observations of the environment. This is typically used to return the raw
	/// unfiltered state observations. For example an rgb or grayscale image.
	/// @return The raw state ofthe environment. The dimensions can differ from observations obtained from a step.
	virtual Observations get_raw_observations() const = 0;

	/// @brief Generates an action via the environments expert agent
	/// @return The action determined by the environments expert agent
	virtual torch::Tensor expert_agent() = 0;
};

/// @brief An interface to abstract the construction of environments and creating an initial state during training
class EnvironmentManager
{
public:
	/// @brief Creates an environment and returns a pointer to the interface. The manager does not own the environment.
	/// @return A pointer to the environment interface
	virtual std::unique_ptr<Environment> make_environment() = 0;

	/// @brief Adds an environment (via calling `make_environment`)and returns a pointer to the interface. The environment
	/// is owned by the manager.
	/// @return A non owning pointer to the environment interface
	virtual Environment* add_environment() = 0;

	/// @brief Gets the environment
	/// @param i The index of the environment to return
	/// @return A non owning raw pointer to the environment interface
	virtual Environment* get_environment(int i) = 0;

	/// @brief The callback when an episode ends and the environment is reset. This allows the environments to be
	/// configured to have a specific state
	/// @return A struct defining the initial state of the environment.
	virtual State get_initial_state() = 0;

	/// @brief Gets the environment configuration which describes the observation shape and data types, action space and
	/// reward types for all the environments.
	/// @return The environment configuration used for all environments
	virtual EnvironmentConfiguration get_configuration() = 0;

	/// @brief Returns the number of enviroments that have been created
	/// @return The number of envs
	virtual int num_envs() const = 0;

	/// @brief Removes all environments, running their destructors
	virtual void reset() = 0;
};

} // namespace drla
