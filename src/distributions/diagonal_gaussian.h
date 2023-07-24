#pragma once

#include "distribution.h"
#include "normal.h"

#include <c10/util/ArrayRef.h>
#include <torch/torch.h>

namespace drla
{

class DiagonalGaussian : public Distribution
{
public:
	DiagonalGaussian(const torch::Tensor& mu, const torch::Tensor& log_std, bool squash = false, float epsilon = 1e-8);

	torch::Tensor entropy() override;
	torch::Tensor log_prob(torch::Tensor value) override;
	torch::Tensor sample(bool deterministic, c10::ArrayRef<int64_t> sample_shape = {}) override;
	const torch::Tensor get_action_output() const override;

private:
	Normal dist_;
	bool squash_;
	float epsilon_;
};

} // namespace drla
