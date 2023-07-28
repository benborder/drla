#include "drla/model/utils.h"

namespace drla
{

torch::Tensor flatten(const std::vector<torch::Tensor>& x, int dims)
{
	std::vector<int64_t> shape;
	shape.resize(dims + 2, -1);
	for (int i = 0, ilen = dims + 1; i < ilen; ++i) { shape[i] = x.front().size(i); }
	std::vector<torch::Tensor> reshaped;
	reshaped.reserve(x.size());
	for (auto& feature : x) { reshaped.emplace_back(feature.view(shape)); }
	return torch::cat(reshaped, dims + 1);
}

torch::Tensor normalise(const torch::Tensor& x)
{
	torch::Tensor min;
	torch::Tensor max;
	if (x.dim() == 4)
	{
		min = std::get<0>(x.view({x.size(0), x.size(1), -1}).min(2)).view({x.size(0), x.size(1), 1, 1});
		max = std::get<0>(x.view({x.size(0), x.size(1), -1}).max(2)).view({x.size(0), x.size(1), 1, 1});
	}
	else
	{
		std::vector<int64_t> shape(x.dim(), 1);
		shape[0] = x.size(0);
		min = std::get<0>(x.view({x.size(0), -1}).min(1)).view(shape);
		max = std::get<0>(x.view({x.size(0), -1}).max(1)).view(shape);
	}
	return (x - min) / (max - min).clamp_min(1e-5);
}

std::vector<std::vector<int64_t>> condense_shape(const std::vector<std::vector<int64_t>>& input_shape)
{
	std::vector<std::vector<int64_t>> output_shapes;

	for (auto& shape : input_shape)
	{
		auto size = shape.size();

		bool matched = false;
		for (auto& output_shape : output_shapes)
		{
			if (output_shape == shape)
			{
				matched = true;
				// Image based tensor dims are assumed to be of the format [channels, height, width] and appended in the
				// channels dim
				if (size == 3)
				{
					output_shape[0] += shape[0];
				}
				// data based tensor dims are assumed to be of the format [..., data] and appended in the data dim
				else
				{
					output_shape[size - 1] += shape[size - 1];
				}
			}
		}
		if (!matched)
		{
			output_shapes.push_back(shape);
		}
	}

	return output_shapes;
}

std::vector<torch::Tensor> condense(const std::vector<torch::Tensor>& input)
{
	std::vector<torch::Tensor> output;

	for (auto& input_tensor : input)
	{
		bool matched = false;
		for (auto& output_tensor : output)
		{
			if (output_tensor.sizes() == input_tensor.sizes())
			{
				matched = true;
				// Image based tensor dims are assumed to be of the format [batch, channels, height, width] and appended in the
				// channels dim. Other tensors are assumed to be of the format [batch, data] and appended in the second dim
				output_tensor = torch::cat({output_tensor, input_tensor}, 1);
			}
		}
		if (!matched)
		{
			output.push_back(input_tensor);
		}
	}

	return output;
}

ObservationShapes stacked_observation_shape(const ObservationShapes& shape, int stack_size)
{
	stack_size = std::max(stack_size, 0);
	ObservationShapes stacked_shape;
	for (const auto& dims : shape)
	{
		auto stacked_dims = dims;
		if (dims.size() < 3)
		{
			stacked_dims.back() *= stack_size + 1;
			stacked_dims.back() += stack_size;
		}
		else
		{
			stacked_dims.front() *= stack_size + 1;
			stacked_dims.front() += stack_size;
		}
		stacked_shape.push_back(stacked_dims);
	}
	return stacked_shape;
}

} // namespace drla
