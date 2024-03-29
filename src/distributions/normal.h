#pragma once

#include "distribution.h"

#include <ATen/core/Tensor.h>
#include <c10/util/ArrayRef.h>

namespace drla
{

class Normal : public Distribution
{
public:
	Normal(torch::Tensor loc, torch::Tensor scale);

	torch::Tensor entropy() override;
	torch::Tensor log_prob(torch::Tensor value) override;
	torch::Tensor sample(c10::ArrayRef<int64_t> sample_shape = {}) override;
	torch::Tensor mean() const override;
	torch::Tensor mode() const override;
	const torch::Tensor get_action_output() const override;

private:
	torch::Tensor loc_;
	torch::Tensor scale_;
};

} // namespace drla
