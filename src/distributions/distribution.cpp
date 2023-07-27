#include "distribution.h"

using namespace drla;

const std::vector<int64_t>& Distribution::get_batch_shape() const
{
	return batch_shape_;
}

const std::vector<int64_t>& Distribution::get_event_shape() const
{
	return event_shape_;
}

std::vector<int64_t> Distribution::extended_shape(c10::ArrayRef<int64_t> sample_shape)
{
	std::vector<int64_t> output_shape;
	output_shape.insert(output_shape.end(), sample_shape.begin(), sample_shape.end());
	output_shape.insert(output_shape.end(), batch_shape_.begin(), batch_shape_.end());
	output_shape.insert(output_shape.end(), event_shape_.begin(), event_shape_.end());
	return output_shape;
}

torch::Tensor drla::sum_rightmost(const torch::Tensor& value, int64_t dim)
{
	if (dim <= 0)
	{
		return value;
	}
	auto required_shape = value.sizes().vec();
	required_shape.resize(required_shape.size() - (dim - 1));
	required_shape.back() = -1;
	return value.reshape(required_shape).sum(-1);
}
