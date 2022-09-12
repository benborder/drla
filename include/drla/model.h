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
	virtual ~Model();

	/// @brief Model prediction. Predicts the action and/or value for the given observations
	/// @param observations The input observations to pass to the model. The observations must be on the same device as
	/// the model.
	/// @param deterministic Use a deterministic forward pass through the model to determine the action if true. Otherwise
	/// a stochastic policy gradient is used to determine the action. This option is only relevant for policy gradient
	/// based models.
	/// @return The predicted action and/or value from the forward pass through the model.
	virtual PredictOutput predict(const Observations& observations, bool deterministic = true) = 0;

	/// @brief Save the model to file at the specified directory path
	/// @param path The full directory path to save the model to
	virtual void save(const std::filesystem::path& path) = 0;

	/// @brief Load the model from file at the specified directory path
	/// @param path The full directory path to save the model to
	virtual void load(const std::filesystem::path& path) = 0;
};

inline Model::~Model()
{
}

struct ActionPolicyEvaluation
{
	torch::Tensor values;
	torch::Tensor action_log_probs;
	torch::Tensor dist_entropy;
};

/// @brief Common actor critic model interface
class ActorCriticModelInterface : public Model
{
public:
	/// @brief Virtual destructor
	virtual ~ActorCriticModelInterface();

	virtual ActionPolicyEvaluation evaluate_actions(const Observations& observations, const torch::Tensor& actions) = 0;
};

inline ActorCriticModelInterface::~ActorCriticModelInterface()
{
}

/// @brief Common actor critic model interface
class QNetModelInterface : public Model
{
public:
	/// @brief Virtual destructor
	virtual ~QNetModelInterface();

	virtual torch::Tensor forward(const Observations& observations) = 0;

	virtual torch::Tensor forward_target(const Observations& observations) = 0;
	virtual void update(double tau) = 0;

	virtual std::vector<torch::Tensor> parameters(bool recursive = true) const = 0;

	virtual void set_exploration(double exploration) = 0;
};

inline QNetModelInterface::~QNetModelInterface()
{
}

} // namespace drla
