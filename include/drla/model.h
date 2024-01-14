#pragma once

#include "drla/types.h"

#include <torch/torch.h>

#include <filesystem>

namespace drla
{

/// @brief Common model interface
class Model : public torch::nn::Module
{
public:
	/// @brief Virtual destructor
	virtual ~Model() = default;

	/// @brief Model prediction. Predicts the action and/or value for the given observations
	/// @param input The input for the agent's model prediction.
	/// @return The predicted model output, typically including action and/or value from the forward pass through the
	/// model.
	virtual ModelOutput predict(const ModelInput& input) = 0;

	/// @brief This initialises the model output for the first step in an environment (i.e. when env reset is called)
	/// @return The initial model output, typically zerod action and reward/value
	virtual ModelOutput initial() = 0;

	/// @brief Gets the shape of internal hidden state of the model for recurrent based models
	/// @return The shape of internal hidden state.
	virtual StateShapes get_state_shape() const = 0;

	/// @brief Copies a models params from the supplied model to this one
	/// @param model The model to copy params from
	virtual void copy(const Model* model) = 0;

	/// @brief Save the model to file at the specified directory path
	/// @param path The full directory path to save the model to
	virtual void save(const std::filesystem::path& path) = 0;

	/// @brief Load the model from file at the specified directory path
	/// @param path The full directory path to save the model to
	virtual void load(const std::filesystem::path& path) = 0;
};

/// @brief The evaluation results of an actor critic model
struct ActionPolicyEvaluation
{
	// The predicted values
	torch::Tensor values;
	// The log of the probalitity of taking each action, required for importance sampling in a train update
	torch::Tensor action_log_probs;
	// The distribution entropy
	torch::Tensor dist_entropy;
};

/// @brief Common actor critic model interface
class ActorCriticModelInterface : public Model
{
public:
	/// @brief Virtual destructor
	virtual ~ActorCriticModelInterface() = default;

	/// @brief Evaluate the action policy given the observations and actions taken
	/// @param observations The observations used to determine the actions
	/// @param actions The actions taken given the observations
	/// @param states The previous step hidden states output from recurrent models
	/// @return Evaluation of the actor critic model
	virtual ActionPolicyEvaluation evaluate_actions(
		const Observations& observations, const torch::Tensor& actions, const std::vector<torch::Tensor>& states) = 0;
};

/// @brief Common QNetwork model interface
class QNetModelInterface : public Model
{
public:
	/// @brief Virtual destructor
	virtual ~QNetModelInterface() = default;

	/// @brief Performs a forward pass through the feature extractor and QNet model to obtain the q values from
	/// observations. If using recurrent uses the hidden latent state as part of the input.
	/// @param observations The observations to use as input
	/// @param state The hidden recurrent states to use as input. Only for recurrent models.
	/// @return The q values
	virtual torch::Tensor forward(const Observations& observations, const HiddenStates& state) = 0;

	/// @brief Performs a forward pass through the feature extractor and target QNet model in the same manner as `forward`
	/// above.
	/// @param observations The observations to use as input
	/// @param state The hidden recurrent states to use as input. Only for recurrent models.
	/// @return The q values
	virtual torch::Tensor forward_target(const Observations& observations, const HiddenStates& state) = 0;

	/// @brief Updates the target critic params towards the current critic params.
	/// @param tau Ratio of current critic to update the target critic with.
	virtual void update(double tau) = 0;

	/// @brief Returns the model params (both the QNetwork and feature extractor)
	virtual std::vector<torch::Tensor> parameters(bool recursive = true) const = 0;

	/// @brief Sets the exploration ratio. If the model is operating deterministically, this is ignored.
	/// @param exploration The exploration ratio. A value of 1 is full random exploration, a value of 0 is no exploration.
	virtual void set_exploration(double exploration) = 0;
};

/// @brief Common mcts model interface
class MCTSModelInterface : public Model
{
public:
	virtual ~MCTSModelInterface() = default;

	/// @brief Recurrent model prediction. Predicts the action and/or value from the hidden state and actions.
	/// @param previous_output The the output from a previous prediction pass.
	/// @return The predicted action, value and state from the forward pass through the model.
	virtual ModelOutput predict_recurrent(const ModelOutput& previous_output) = 0;

	/// @brief Transform a categorical representation to a scalar
	/// @param logits The transformed categorical support representation
	/// @return Converted scalar representation
	virtual torch::Tensor support_to_scalar(torch::Tensor logits) = 0;

	/// @brief Transform a scalar to a categorical representation
	/// @param x The scalar representation
	/// @return The transformed categorical support representation
	virtual torch::Tensor scalar_to_support(torch::Tensor x) = 0;

	/// @brief Gets the observation stack size of the model
	/// @return The observation stack size. value >= 0.
	virtual int get_stacked_observation_size() const = 0;
};

/// @brief The output generated from passing observations and actions through a world model
struct WorldModelOutput
{
	// The predicted observations
	Observations observation = {};
	// The reward from model prediction
	torch::Tensor reward = {};
	// The prediction of continuation (non terminal) of the episode
	torch::Tensor non_terminal = {};
	// The predicted world model latent states
	std::vector<torch::Tensor> latents = {};
	// The values output from a model forward pass, used only for updating buffer priorities
	torch::Tensor values = {};
};

/// @brief The imagined trajectory. The shape should be [step, batch, ...]
struct ImaginedTrajectory
{
	// The actions the model determined should be taken
	torch::Tensor action = {};
	// The predicted latent states
	torch::Tensor latents = {};
	// The reward from model prediction
	torch::Tensor reward = {};
	// The prediction of continuation (non terminal) of the episode
	torch::Tensor non_terminal = {};
	// The weight for each step in the trajectory
	torch::Tensor weight = {};
};

/// @brief The output generate from passing the latents through the behavioural model
struct BehaviouralModelOutput
{
	// The log of the probalitity of taking each action, required for importance sampling in a train update
	torch::Tensor log_probs = {};
	// The action distribution entropy
	torch::Tensor entropy = {};
	// The values output from a model forward pass
	torch::Tensor values = {};
	// The target values output from a model forward pass
	torch::Tensor target_values = {};
};

/// @brief Common hybrid model interface which contains a world model for the dynamics representation and a seperate
/// behavioural model (which uses the world model) for policy generation.
class HybridModelInterface : public Model
{
public:
	virtual ~HybridModelInterface() = default;

	/// @brief Evaluates the world model using the output for training
	/// @param observations The input observations to evaluate
	/// @param actions The input actions to evaluate
	/// @return The output from the world model
	virtual WorldModelOutput evaluate_world_model(
		const Observations& observations,
		const torch::Tensor& actions,
		const HiddenStates& states,
		const torch::Tensor& is_first) = 0;

	/// @brief Imagines a trajectory of state action pairs via the behavioural policy model using latent hidden states
	/// generated by the world model
	/// @param horizon The number of steps to imagine
	/// @param initial_states The initial starting representation states to start imagining from
	/// @return The imagined trajectory of action and latent states
	virtual ImaginedTrajectory imagine_trajectory(int horizon, const WorldModelOutput& initial_states) = 0;

	/// @brief Evaluates the behavioural model using the output for training
	/// @param latents The input latent states to generate values from
	/// @param actions The actions taken given the latents
	/// @return The behavioural model output, i.e. values
	virtual BehaviouralModelOutput
	evaluate_behavioural_model(const torch::Tensor& latents, const torch::Tensor& actions) = 0;

	/// @brief Returns the parameters for the world model
	/// @return The params
	virtual std::vector<torch::Tensor> world_model_parameters() const = 0;

	/// @brief Returns the parameters for the actor
	/// @return The params
	virtual std::vector<torch::Tensor> actor_parameters() const = 0;

	/// @brief Returns the parameters for the critic
	/// @return The params
	virtual std::vector<torch::Tensor> critic_parameters() const = 0;

	/// @brief Updates slow/target critic networks with a smoothing factor of tau
	/// @param tau The ratio of the latest critic to use to update target/slow networks
	virtual void update(double tau) = 0;
};

} // namespace drla
