#include "diagonal_gaussian.h"

#include "normal.h"

#include <c10/util/ArrayRef.h>
#include <torch/torch.h>

#define _USE_MATH_DEFINES
#include <cmath>
#include <limits>

using namespace drla;

DiagonalGaussian::DiagonalGaussian(const torch::Tensor& mu, const torch::Tensor& log_std, bool squash, float epsilon)
		: dist_(mu, log_std), squash_(squash), epsilon_(epsilon)
{
}

torch::Tensor DiagonalGaussian::entropy()
{
	return dist_.entropy().sum(-1);
}

torch::Tensor DiagonalGaussian::action_log_prob(torch::Tensor action)
{
	if (squash_)
	{
		action = action.clamp(-1 + std::numeric_limits<float>::epsilon(), 1 - std::numeric_limits<float>::epsilon());
		// According to stable baselines atanh has numerical stability issues, so use a custom one
		action = 0.5 * (action.log1p() - (-action).log1p());

		auto log_prob = dist_.action_log_prob(action);
		log_prob -= (1 - action.square() + epsilon_).log().sum(-1);
		return log_prob;
	}
	else
	{
		return dist_.action_log_prob(action).sum(-1);
	}
}

torch::Tensor DiagonalGaussian::sample(bool deterministic, c10::ArrayRef<int64_t> sample_shape)
{
	auto actions = dist_.sample(deterministic, sample_shape);
	if (squash_)
	{
		actions = torch::tanh(actions);
	}
	return actions;
}

const torch::Tensor DiagonalGaussian::get_action_output() const
{
	return dist_.get_action_output();
}
