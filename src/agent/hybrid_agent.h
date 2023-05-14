#pragma once

#include "drla/agent.h"
#include "drla/callback.h"
#include "drla/configuration.h"
#include "drla/environment.h"
#include "drla/model.h"
#include "drla/stats.h"
#include "drla/types.h"

#include <torch/torch.h>

#include <atomic>
#include <memory>
#include <random>
#include <string>

namespace drla
{

/// @brief An agent based on hybrid model and actor critic architecture
class HybridAgent final : public Agent
{
public:
	/// @brief Creates and configures the agent.
	/// @param config The configuration for the agent.
	/// @param environment_manager A non owning pointer to the environment manager interface, which is responsible for
	/// creating environments.
	/// @param callback The non owning pointer to the agent callback interface, which is used for training and environment
	/// step updates.
	/// @param data_path The save/load path for config, models and training optimiser state.
	HybridAgent(
		const Config::Agent& config,
		EnvironmentManager* environment_manager,
		AgentCallbackInterface* callback,
		std::filesystem::path data_path);
	~HybridAgent();

	/// @brief Initiate training the agent, running for the number of epochs defined in the configuration file
	void train() override;

protected:
	// Hybrid agent configuration
	const Config::HybridAgent config_;

	std::mutex m_env_stats_;
	Stats<> env_samples_stats_;
	Stats<> env_duration_stats_;
	int64_t total_samples_ = 0;
};

} // namespace drla
