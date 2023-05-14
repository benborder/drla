#pragma once

#include "model/utils.h"

#include <torch/torch.h>

#include <vector>

namespace drla
{

class Discrete
{
public:
	Discrete(const torch::Tensor& logits, int low = -20, int high = 20)
			: logits_(logits)
			, probs_(torch::softmax(logits_, -1))
			, nbins_(logits.size(-1))
			, bins_(torch::linspace(low, high, nbins_, logits_.device()))
	{
	}

	torch::Tensor mode() const { return (probs_ * bins_).sum(-1); }

	torch::Tensor mean() const { return mode(); }

	torch::Tensor log_prob(torch::Tensor x)
	{
		auto below = (bins_.unsqueeze(0) <= x.unsqueeze(-1)).to(torch::kLong).sum(-1) - 1;
		auto above = nbins_ - (bins_.unsqueeze(0) > x.unsqueeze(-1)).to(torch::kLong).sum(-1);
		below.clip_(0, nbins_ - 1);
		above.clip_(0, nbins_ - 1);
		auto equal = (below == above);
		auto dist_to_below = torch::where(equal, 1, torch::abs(bins_.index({below}) - x));
		auto dist_to_above = torch::where(equal, 1, torch::abs(bins_.index({above}) - x));
		auto total = dist_to_below + dist_to_above;
		auto weight_below = dist_to_above / total;
		auto weight_above = dist_to_below / total;
		auto target = torch::one_hot(below, nbins_) * weight_below.unsqueeze(-1) +
									torch::one_hot(above, nbins_) * weight_above.unsqueeze(-1);
		auto log_pred = logits_ - torch::logsumexp(logits_, -1, true);
		return sum_rightmost(target * log_pred, 2);
	}

private:
	torch::Tensor logits_;
	torch::Tensor probs_;
	int nbins_;
	torch::Tensor bins_;
};

} // namespace drla
