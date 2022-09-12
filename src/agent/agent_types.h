#pragma once

#include "types.h"

#include <torch/torch.h>

#include <vector>

namespace drla
{

struct TimeStepData
{
	// the step number
	int step = 0;

	// Action prediction result
	PredictOutput predict_results;

	// The reward recieved from the step
	torch::Tensor rewards;
	// The output observation from the step
	Observations observations;
	// The state of the environment at the end of the step
	std::vector<State> states;
};

} // namespace drla
