#pragma once

#include <torch/torch.h>

#include <vector>

namespace drla
{

class Distribution
{
public:
	virtual ~Distribution();

	virtual torch::Tensor entropy() = 0;
	virtual torch::Tensor action_log_prob(torch::Tensor action) = 0;
	virtual torch::Tensor sample(bool deterministic = false, c10::ArrayRef<int64_t> sample_shape = {}) = 0;
	virtual const torch::Tensor get_action_output() const = 0;

protected:
	std::vector<int64_t> extended_shape(c10::ArrayRef<int64_t> sample_shape);

	std::vector<int64_t> batch_shape_;
	std::vector<int64_t> event_shape_;
};

inline Distribution::~Distribution()
{
}

} // namespace drla
