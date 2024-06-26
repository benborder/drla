#pragma once

#include "drla/callback.h"
#include "drla/configuration.h"
#include "drla/environment.h"
#include "drla/model.h"
#include "drla/types.h"

#include <filesystem>
#include <string>
#include <vector>

namespace drla
{

class ThreadPool;

/// @brief Options for running an agent.
struct RunOptions
{
	// The maximum number of steps to run the environment. <= 0 implies to run until the environment terminates normally.
	int max_steps = 0;
	// Use deterministic actions when true, otherwise a stochastic policy gradient to determine actions
	bool deterministic = true;
	// Capture visualisation data and pass to callbacks which use StepData
	bool enable_visualisations = true;
	// If a model is already loaded, force reload the model again from the data_path
	bool force_model_reload = false;
	// Determines how greedy the action selection is for MCTS based agents. 0 is maximally greedy and the larger the value
	// the more random.
	float temperature = 0.0F;
};

/// @brief The base agent functionality.
class Agent
{
public:
	/// @brief Creates and configures the agent.
	/// @param config The configuration for the agent.
	/// @param environment_manager A non owning pointer to the environment manager interface, which is responsible for
	/// creating environments.
	/// @param callback The non owning pointer to the agent callback interface, which is used for training and environment
	/// step updates.
	/// @param data_path The save/load path for config, models and training optimiser state.
	Agent(
		const Config::Agent& config,
		EnvironmentManager* environment_manager,
		AgentCallbackInterface* callback,
		std::filesystem::path data_path);
	virtual ~Agent();

	/// @brief Creates/Resets environments and runs the agent for a max number of steps or until the environment
	/// terminates with an episode end. By default the first time 'run()' is called a model is loaded if no model has been
	/// loaded yet. Subsequent calls to will use the model already loaded.
	/// @param initial_state The initial state for each environment. The number of initial state elements determins the
	/// number of environments to run the agent in.
	/// @param options Options which change various behaviours of the agent. See RunOptions for more detail on available
	/// options.
	virtual void run(const std::vector<State>& initial_state, RunOptions options = {});

	/// @brief Initialises the agent, loading the model and returning the initial model output. **NOTE** This is only
	/// necessary to be used in conjunction with the 'predict()' method where the environment is handled externally.
	/// @param env_config The environment configuration which the model may use to correctly construct its
	/// inputs/outputs
	/// @param reload Reloads the model if another model has already been loaded
	/// @return The initial model output
	virtual ModelOutput initialise(const EnvironmentConfiguration& env_config, bool reload = false);

	/// @brief Predicts the next action and state to perform given the input observations. Effectively performs a forward
	/// pass through the model. **NOTE** 'initialise()' must be called at least once before calling this method.
	/// @param step_history The agent and environment data to pass to the model. This method handles converting to the
	/// correct device. **NOTE* There must be at least one item.
	/// @param deterministic Use a deterministic forward pass through the model to determine the action if true. Otherwise
	/// a stochastic policy gradient is used to determine the action. This option is only relevant for policy gradient
	/// based models.
	/// @return The predicted action and/or value from the forward pass through the model.
	virtual ModelOutput predict(const std::vector<StepData>& step_history, bool deterministic = true);

	/// @brief Clears and resets any loaded models, environments and state
	virtual void reset();

	/// @brief Initiate training the agent, running for the number of epochs defined in the configuration file
	virtual void train();

	/// @brief Load a model from the data path
	/// @param force_reload Forces loading the model even if a model has already been loaded
	virtual void load_model(bool force_reload = false);

	/// @brief Stop training after the current epoch is completed
	void stop_train();

	/// @brief Sets the save/load path for config, models and training optimizer state
	void set_data_path(const std::filesystem::path& path);

protected:
	std::filesystem::path get_save_path(const std::string& postfix = "") const;
	void make_environments(ThreadPool& threadpool, int env_count);
	void run_episode(Model* model, const State& initial_state, int env, RunOptions options);
	void load_model(const EnvironmentConfiguration& env_config);

	// Common configuration for all agents
	const Config::AgentBase base_config_;

	// The compute devices to use
	std::vector<torch::Device> devices_;

	// Indicates if the agent is training
	bool training_ = false;
	// Indicates if rewards from the environment should be combined/summed
	bool combine_rewards_ = false;

	// The model loaded when the agent is run
	std::shared_ptr<Model> model_;

	// Interface to use for creating and initialising environments
	EnvironmentManager* environment_manager_;
	// Interface to use for external callbacks
	AgentCallbackInterface* agent_callback_;

	// The path where configuration and model are saved/loaded
	std::filesystem::path data_path_;
};

/// @brief Creates an agent based on the model type specified in the configuration. The training algorithms available
/// are dependent on the model type.
/// @param config The agent specific configuration
/// @param environment_manager The non owning interface pointer to the environment manager
/// @param callback The non owning interface pointer to the agent's callback methods
/// @param data_path The path to use to load/save configuration and model data
/// @return An owning interface pointer to the created agent
std::unique_ptr<Agent> make_agent(
	const Config::Agent& config,
	EnvironmentManager* environment_manager,
	AgentCallbackInterface* callback,
	const std::filesystem::path& data_path);

} // namespace drla
