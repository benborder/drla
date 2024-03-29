#pragma once

#include "drla/agent.h"
#include "drla/callback.h"
#include "drla/configuration.h"
#include "drla/environment.h"
#include "drla/types.h"

#include <memory>
#include <string>

namespace drla
{

class Model;

/// @brief An agent for training via off policy based algorithms
class OffPolicyAgent final : public Agent
{
public:
	/// @brief Creates and configures the agent.
	/// @param config The configuration for the agent.
	/// @param environment_manager A non owning pointer to the environment manager interface, which is responsible for
	/// creating environments.
	/// @param callback The non owning pointer to the agent callback interface, which is used for training and environment
	/// step updates.
	/// @param data_path The save/load path for config, models and training optimiser state.
	OffPolicyAgent(
		const Config::Agent& config,
		EnvironmentManager* environment_manager,
		AgentCallbackInterface* callback,
		std::filesystem::path data_path);
	/// @brief Creates and configures the agent.
	/// @param config The configuration for the agent.
	/// @param environment_manager A non owning pointer to the environment manager interface, which is responsible for
	/// creating environments.
	/// @param callback The non owning pointer to the agent callback interface, which is used for training and environment
	/// step updates.
	/// @param data_path The save/load path for config, models and training optimiser state.
	OffPolicyAgent(
		const Config::OffPolicyAgent& config,
		EnvironmentManager* environment_manager,
		AgentCallbackInterface* callback,
		std::filesystem::path data_path);
	~OffPolicyAgent();

	/// @brief Initiate training the agent, running for the number of epochs defined in the configuration file
	void train() override;

protected:
	// Off policy specific agent configuration
	const Config::OffPolicyAgent config_;
};

} // namespace drla
