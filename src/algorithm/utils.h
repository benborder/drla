#pragma once

#include "configuration/algorithm.h"

#include <torch/torch.h>

namespace drla
{

inline double explained_variance(const torch::Tensor& predicted, const torch::Tensor& actual)
{
	auto pred = predicted.flatten();
	double var_pred = pred.var().item<double>();
	return 1.0 - (actual.flatten() - pred).var().item<double>() / var_pred;
}

} // namespace drla
