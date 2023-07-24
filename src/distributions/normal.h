#pragma once

#include "distribution.h"

#include <c10/util/ArrayRef.h>
#include <torch/torch.h>

namespace drla
{

class Normal : public Distribution
{
public:
	Normal(torch::Tensor loc, torch::Tensor scale);

	torch::Tensor entropy() override;
	torch::Tensor log_prob(torch::Tensor value) override;
	torch::Tensor sample(bool deterministic, c10::ArrayRef<int64_t> sample_shape = {}) override;
	const torch::Tensor get_action_output() const override;

private:
	torch::Tensor loc_;
	torch::Tensor scale_;
};

} // namespace drla
