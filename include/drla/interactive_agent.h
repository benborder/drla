#pragma once

#include "drla/agent.h"
#include "drla/callback.h"
#include "drla/configuration.h"
#include "drla/environment.h"
#include "drla/types.h"

#include <torch/torch.h>

#include <string>

namespace drla
{

/// @brief An agent for running in an interactive mode
class InteractiveAgent final : public Agent
{
public:
	/// @brief Creates and configures the agent.
	/// @param config The configuration for the agent.
	/// @param environment_manager A non owning pointer to the environment manager interface, which is responsible for
	/// creating environments.
	/// @param callback The non owning pointer to the agent callback interface, which is used for training and environment
	/// step updates.
	/// @param data_path The save/load path for config, models and training optimiser state.
	InteractiveAgent(
			const Config::Agent& config,
			EnvironmentManager* environment_manager,
			AgentCallbackInterface* callback,
			std::string data_path);
	/// @brief Creates and configures the agent.
	/// @param config The configuration for the agent.
	/// @param environment_manager A non owning pointer to the environment manager interface, which is responsible for
	/// creating environments.
	/// @param callback The non owning pointer to the agent callback interface, which is used for training and environment
	/// step updates.
	/// @param data_path The save/load path for config, models and training optimiser state.
	InteractiveAgent(
			const Config::InteractiveAgent& config,
			EnvironmentManager* environment_manager,
			AgentCallbackInterface* callback,
			std::string data_path);

	/// @brief Runs the agent deterministically in the environment specified for the max number of steps. The first time
	/// this is run, a model is loaded based on the configuration if no model has been loaded yet. Subsequent calls to
	/// this will use the model already laoded.
	/// @param initial_state The initial state for each environment. The number of initial state elements determins the
	/// number of environments to run.
	/// @param options Change the behaviour of the agent based on these options.
	void run(const std::vector<State>& initial_state, RunOptions options = {}) override;

	/// @brief Clears and resets any loaded models and state
	void reset() override;

	/// @brief Initiate training the agent, running for the number of epochs defined in the configuration file
	void train() override;

protected:
	// Interactive agent specific configuration
	const Config::InteractiveAgent config_;
};

} // namespace drla
