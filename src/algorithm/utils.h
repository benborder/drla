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

inline double learning_rate_decay(const Config::TrainAlgorithm* config, int step, int max_steps)
{
	double progress = static_cast<double>(step) / static_cast<double>(max_steps);
	switch (config->lr_schedule_type)
	{
		case LearningRateScheduleType::kLinear:
		{
			return std::max<double>(0.0, 1.0 - config->lr_decay_rate * progress);
		}
		case LearningRateScheduleType::kExponential:
		{
			return std::max<double>(0.0, std::exp(-0.5 * M_PI * config->lr_decay_rate * progress));
		}
		case LearningRateScheduleType::kConstant:
		default:
		{
			return 1.0;
		}
	}
}

} // namespace drla
