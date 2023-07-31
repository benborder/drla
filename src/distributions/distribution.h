#pragma once

#include <torch/torch.h>

#include <vector>

namespace drla
{

class Distribution
{
public:
	virtual ~Distribution() = default;

	virtual torch::Tensor entropy() = 0;
	virtual torch::Tensor log_prob(torch::Tensor value) = 0;
	virtual torch::Tensor sample(c10::ArrayRef<int64_t> sample_shape = {}) = 0;
	virtual torch::Tensor mean() const = 0;
	virtual torch::Tensor mode() const = 0;
	virtual const torch::Tensor get_action_output() const = 0;

	const std::vector<int64_t>& get_batch_shape() const;
	const std::vector<int64_t>& get_event_shape() const;

protected:
	std::vector<int64_t> extended_shape(c10::ArrayRef<int64_t> sample_shape = {}) const;

	std::vector<int64_t> batch_shape_;
	std::vector<int64_t> event_shape_;
};

torch::Tensor sum_rightmost(const torch::Tensor& value, int64_t dim);

} // namespace drla
