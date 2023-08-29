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

	/// @brief Save the model to file at the specified directory path
	/// @param path The full directory path to save the model to
	virtual void save(const std::filesystem::path& path) = 0;

	/// @brief Load the model from file at the specified directory path
	/// @param path The full directory path to save the model to
	virtual void load(const std::filesystem::path& path) = 0;
};

struct ActionPolicyEvaluation
{
	torch::Tensor values;
	// The log of the probalitity of taking each action, required for importance sampling in a train update
	torch::Tensor action_log_probs;
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

/// @brief Common actor critic model interface
class QNetModelInterface : public Model
{
public:
	/// @brief Virtual destructor
	virtual ~QNetModelInterface() = default;

	virtual torch::Tensor forward(const Observations& observations, const HiddenStates& state) = 0;

	virtual torch::Tensor forward_target(const Observations& observations, const HiddenStates& state) = 0;
	virtual void update(double tau) = 0;

	virtual std::vector<torch::Tensor> parameters(bool recursive = true) const = 0;

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

} // namespace drla
