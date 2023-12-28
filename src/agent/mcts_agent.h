#pragma once

#include "drla/agent.h"
#include "drla/callback.h"
#include "drla/configuration.h"
#include "drla/environment.h"
#include "drla/stats.h"
#include "drla/types.h"

#include <torch/torch.h>

#include <atomic>
#include <memory>
#include <random>
#include <string>

namespace drla
{

class Model;
class MCTSEpisode;

/// @brief A Monte Carlo Tree Search based agent
class MCTSAgent final : public Agent
{
public:
	/// @brief Creates and configures the agent.
	/// @param config The configuration for the agent.
	/// @param environment_manager A non owning pointer to the environment manager interface, which is responsible for
	/// creating environments.
	/// @param callback The non owning pointer to the agent callback interface, which is used for training and environment
	/// step updates.
	/// @param data_path The save/load path for config, models and training optimiser state.
	MCTSAgent(
		const Config::Agent& config,
		EnvironmentManager* environment_manager,
		AgentCallbackInterface* callback,
		std::filesystem::path data_path);
	~MCTSAgent();

	/// @brief Creates/Resets environments and runs the agent for a max number of steps or until the environment
	/// terminates with an episode end. By default the first time 'run()' is called a model is loaded if no model has been
	/// loaded yet. Subsequent calls to `run()` will use the model already loaded.
	/// @param initial_states The initial state for each environment. The number of initial state elements determins the
	/// number of environments to run the agent in.
	/// @param options Options which change various behaviours of the agent. See RunOptions for more detail on available
	/// options.
	void run(const std::vector<State>& initial_states, RunOptions options = {}) override;

	/// @brief Predicts the next action to perform given the input observations. Effectively performs a forward pass
	/// through the model. **NOTE** 'load_model()' must be called at least once before calling this method.
	/// @param step_history The agent and environment data to pass to the model. This method handles converting to the
	/// correct device.
	/// @param deterministic Use a deterministic forward pass through the model to determine the action if true. Otherwise
	/// a stochastic policy gradient is used to determine the action. This option is only relevant for policy gradient
	/// based models.
	/// @return The predicted action and/or value from the forward pass through the model.
	ModelOutput predict_action(const std::vector<StepData>& step_history, bool deterministic = true) override;

	/// @brief Initiate training the agent, running for the number of epochs defined in the configuration file
	void train() override;

protected:
	std::unique_ptr<MCTSEpisode>
	run_episode(MCTSModelInterface* model, const State& initial_state, int env, bool eval_mode, RunOptions options);
	ModelOutput run_step(
		MCTSModelInterface* model,
		const std::vector<StepData>& step_history,
		bool deterministic = true,
		float temperature = 0.0f);
	Observations get_stacked_observations(const std::vector<StepData>& step_history, int stack_size, int num_actions);

protected:
	// MCTS model based specific agent configuration
	const Config::MCTSAgent config_;

	std::mt19937 gen_;

	std::mutex m_env_stats_;
	Stats<> env_samples_stats_;
	Stats<> env_duration_stats_;
	Stats<> env_steps_stats_;
	int64_t total_samples_ = 0;
};

} // namespace drla
