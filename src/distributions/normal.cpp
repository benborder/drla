#include "normal.h"

#include <c10/util/ArrayRef.h>
#include <torch/torch.h>

#define _USE_MATH_DEFINES
#include <cmath>
#include <limits>

using namespace drla;

Normal::Normal(torch::Tensor loc, torch::Tensor scale)
{
	auto broadcasted_tensors = torch::broadcast_tensors({loc, scale});
	loc_ = broadcasted_tensors[0];
	scale_ = broadcasted_tensors[1];
	batch_shape_ = loc_.sizes().vec();
	event_shape_ = {};
}

torch::Tensor Normal::entropy()
{
	return (0.5 + 0.5 * std::log(2 * M_PI) + torch::log(scale_)).sum(-1);
}

torch::Tensor Normal::action_log_prob(torch::Tensor action)
{
	auto variance = scale_.pow(2);
	auto log_scale = scale_.log();
	return (-(action - loc_).pow(2) / (2 * variance) - log_scale - std::log(std::sqrt(2 * M_PI)));
}

torch::Tensor Normal::sample(bool deterministic, c10::ArrayRef<int64_t> sample_shape)
{
	if (deterministic)
	{
		return loc_;
	}
	auto shape = extended_shape(sample_shape);
	torch::NoGradGuard no_grad;
	return at::normal(loc_.expand(shape), scale_.expand(shape));
}

const torch::Tensor Normal::get_action_output() const
{
	return loc_;
}
