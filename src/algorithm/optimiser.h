#pragma once

#include "configuration/algorithm.h"
#include "model/utils.h"

#include <torch/torch.h>

#include <memory>

namespace drla
{

class Optimiser
{
public:
	explicit Optimiser(
		const Config::Optimiser& optimiser_config, const std::vector<torch::Tensor>& params, int max_timesteps);

	std::tuple<double, double> update(int timestep);

	void step(torch::Tensor& loss);

	torch::optim::Optimizer& get_optimiser() const;

protected:
	const Config::Optimiser config_;
	const int max_timesteps_;
	std::unique_ptr<torch::optim::Optimizer> optimiser_;
};

} // namespace drla
