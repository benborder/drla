#pragma once

#include "categorical.h"
#include "configuration/algorithm.h"
#include "distribution.h"
#include "functions.h"
#include "model/utils.h"

#include <ATen/core/Tensor.h>

namespace drla
{

inline double explained_variance(const torch::Tensor& predicted, const torch::Tensor& actual)
{
	auto pred = predicted.flatten();
	double var_pred = pred.var().item<double>();
	return 1.0 - (actual.flatten() - pred).var().item<double>() / var_pred;
}

/// @brief Calculate KL divergence via Categorical independent distribution
/// @param p dist to compare
/// @param q dist to compare
/// @param reinterpreted_batch_ndims the number of batch dims to reinterpret as event dims
/// @return difference between p and q
inline torch::Tensor
kl_divergence_independent(const Categorical& p, const Categorical& q, int reinterpreted_batch_ndims = 1)
{
	auto t = p.probs() * (p.logits() - q.logits());
	t.masked_fill_(q.probs() == 0, std::numeric_limits<float>::infinity());
	t.masked_fill_(p.probs() == 0, 0);
	return sum_rightmost(t, reinterpreted_batch_ndims + 1);
}

inline torch::Tensor mse_logprob_loss(const torch::Tensor& mode, torch::Tensor value, int ndims)
{
	return -sum_rightmost((mode - value).square(), ndims);
}

class Moments : public torch::nn::Module
{
public:
	Moments(float decay = 0.99F, float low = 0.05F, float high = 0.95F, float epsilon = 1e-8)
			: decay_(decay)
			, epsilon_(make_tensor(epsilon))
			, plow_(make_tensor(low))
			, phigh_(make_tensor(high))
			, low_(torch::zeros(1))
			, high_(torch::zeros(1))
	{
		register_buffer("epsilon", epsilon_);
		register_buffer("plow", plow_);
		register_buffer("phigh", phigh_);
		register_buffer("low", low_);
		register_buffer("high", high_);
	}

	std::tuple<torch::Tensor, torch::Tensor> forward(const torch::Tensor& x)
	{
		update(x);
		return stats();
	}

	void update(const torch::Tensor& x)
	{
		torch::NoGradGuard no_grad;
		low_ = decay_ * low_ + (1.0F - decay_) * torch::quantile(x.detach(), plow_);
		high_ = decay_ * high_ + (1.0F - decay_) * torch::quantile(x.detach(), phigh_);
	}

	std::tuple<torch::Tensor, torch::Tensor> stats() const
	{
		return {low_.detach(), torch::max(epsilon_, high_ - low_).detach()};
	}

private:
	const float decay_;
	torch::Tensor epsilon_;
	torch::Tensor plow_;
	torch::Tensor phigh_;
	torch::Tensor low_;
	torch::Tensor high_;
};

} // namespace drla
