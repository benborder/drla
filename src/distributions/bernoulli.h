#pragma once

#include "distribution.h"

#include <c10/util/ArrayRef.h>
#include <torch/torch.h>

#include <optional>

namespace drla
{

class Bernoulli : public Distribution
{
public:
	Bernoulli(const torch::Tensor probs = {}, const torch::Tensor logits = {});

	torch::Tensor entropy() override;
	torch::Tensor log_prob(torch::Tensor value) override;
	torch::Tensor sample(c10::ArrayRef<int64_t> sample_shape = {}) override;
	torch::Tensor mean() const override;
	torch::Tensor mode() const override;
	const torch::Tensor get_action_output() const override;

private:
	torch::Tensor param_;
	torch::Tensor probs_;
	torch::Tensor logits_;
};

} // namespace drla
