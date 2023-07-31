#include "bernoulli.h"

#include <ATen/core/Reduction.h>

#include <limits>

using namespace drla;

Bernoulli::Bernoulli(const torch::Tensor probs, const torch::Tensor logits)
{
	if (probs.defined() == logits.defined())
	{
		throw std::invalid_argument("Either `probs` or `logits` must be specified, but not both.");
	}
	if (probs.defined())
	{
		if (probs.dim() < 1)
		{
			throw std::invalid_argument("Probs dimension must be non zero!");
		}
		param_ = probs;
		probs_ = param_ / param_.sum(-1, true);
		auto pclamped = probs_.clamp(1e-8, 1 - 1e-8);
		logits_ = pclamped.log() - pclamped.log1p();
	}
	else
	{
		if (logits.dim() < 1)
		{
			throw std::invalid_argument("Logits dimension must be non zero!");
		}
		param_ = logits;
		logits_ = param_;
		probs_ = torch::sigmoid(logits_);
	}

	batch_shape_ = logits_.sizes().vec();
}

torch::Tensor Bernoulli::entropy()
{
	return torch::binary_cross_entropy_with_logits(logits_, probs_, {}, {}, torch::Reduction::None);
}

torch::Tensor Bernoulli::log_prob(torch::Tensor value)
{
	auto broadcasted_tensors = torch::broadcast_tensors({logits_, value});
	return -torch::binary_cross_entropy_with_logits(
		broadcasted_tensors[0], broadcasted_tensors[1], {}, {}, torch::Reduction::None);
}

torch::Tensor Bernoulli::sample(c10::ArrayRef<int64_t> sample_shape)
{
	auto ext_sample_shape = extended_shape(sample_shape);
	torch::NoGradGuard no_grad;
	return torch::bernoulli(probs_.expand(ext_sample_shape));
}

torch::Tensor Bernoulli::mean() const
{
	return probs_;
}

torch::Tensor Bernoulli::mode() const
{
	return (probs_ > 0.5F).to(probs_);
}

const torch::Tensor Bernoulli::get_action_output() const
{
	return param_;
}
