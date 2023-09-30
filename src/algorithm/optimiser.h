#pragma once

#include "configuration/algorithm.h"
#include "model/utils.h"

#include <torch/torch.h>

#include <memory>

namespace drla
{

/// @brief Wraps optimiser functionality in a convenience class
class Optimiser
{
public:
	explicit Optimiser(
		const Config::Optimiser& optimiser_config, const std::vector<torch::Tensor>& params, int max_timesteps);

	/// @brief Updates the learning rate and alpha
	/// @param timestep
	void update(int timestep);

	/// @brief Performance an optimisation step, zeroing the grad, backpropagating the loss applying clipping based on
	/// config and then stepping the optimiser.
	/// @param loss The loss to use backpropagate
	void step(torch::Tensor& loss);

	/// @brief Returns the optimiser
	/// @return A reference to the optimiser
	torch::optim::Optimizer& get_optimiser() const;

	/// @brief Returns the current learning rate
	/// @return The learning rate
	double get_lr() const;

	/// @brief Returns the current alpha, which is the scaling ratio applied to the max learning rate. This can be used to
	/// scale other hyperparameters.
	/// @return The current alpha
	double get_alpha() const;

protected:
	const Config::Optimiser config_;
	const int max_timesteps_;
	double lr_ = 0.0;
	double alpha_ = 1.0;
	std::unique_ptr<torch::optim::Optimizer> optimiser_;
};

} // namespace drla
