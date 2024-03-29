#pragma once

#include "distribution.h"
#include "normal.h"

#include <ATen/core/Tensor.h>
#include <c10/util/ArrayRef.h>

namespace drla
{

class DiagonalGaussian : public Distribution
{
public:
	DiagonalGaussian(const torch::Tensor& mu, const torch::Tensor& log_std, bool squash = false, float epsilon = 1e-8);

	torch::Tensor entropy() override;
	torch::Tensor log_prob(torch::Tensor value) override;
	torch::Tensor sample(c10::ArrayRef<int64_t> sample_shape = {}) override;
	torch::Tensor mean() const override;
	torch::Tensor mode() const override;
	const torch::Tensor get_action_output() const override;

private:
	Normal dist_;
	bool squash_;
	float epsilon_;
};

} // namespace drla
