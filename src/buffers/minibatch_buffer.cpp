#include "minibatch_buffer.h"

#include <torch/torch.h>

#include <algorithm>
#include <vector>

using namespace drla;

MiniBatchBuffer::MiniBatchBuffer(const RolloutBuffer& buffer, int mini_batch_size) : buffer_(buffer)
{
	indices_ =
		torch::randperm(buffer_.get_buffer_sample_size(), torch::TensorOptions(torch::kLong)).view({-1, mini_batch_size});
}

MiniBatchBuffer::Iterator MiniBatchBuffer::begin()
{
	return Iterator(buffer_, indices_, 0);
}

MiniBatchBuffer::Iterator MiniBatchBuffer::end()
{
	return Iterator(buffer_, indices_, indices_.size(0));
}

size_t MiniBatchBuffer::size() const
{
	return static_cast<size_t>(indices_.size(0));
}

void MiniBatchBuffer::Iterator::get_minibatch()
{
	auto index = indices_[index_];
	int steps = buffer_.get_actions().size(0);

	auto& observations = buffer_.get_observations();
	for (size_t i = 0; i < observations.size(); i++)
	{
		auto observations_shape = observations[i].sizes().vec();
		observations_shape.erase(observations_shape.begin());
		observations_shape[0] = -1;
		minibatch_.observations.push_back(
			observations[i].narrow(0, 0, steps).view(observations_shape).index({index}).to(buffer_.get_device()));
	}

	auto action_shape = buffer_.get_actions().sizes().vec();
	action_shape.erase(action_shape.begin());
	action_shape[0] = -1;
	minibatch_.actions = buffer_.get_actions().view({action_shape}).index({index});

	auto value_shape = buffer_.get_returns().size(2);
	minibatch_.old_log_probs = buffer_.get_action_log_probs().view({-1, 1}).index({index});
	minibatch_.old_values = buffer_.get_values().narrow(0, 0, steps).view({-1, value_shape}).index({index});
	minibatch_.returns = buffer_.get_returns().narrow(0, 0, steps).view({-1, value_shape}).index({index});
	minibatch_.advantages = buffer_.get_advantages().narrow(0, 0, steps).view({-1, value_shape}).index({index});
	auto states = buffer_.get_states();
	for (size_t i = 0; i < states.size(); i++)
	{
		auto state = states[i];
		minibatch_.states.push_back(state.narrow(0, 0, steps).view({-1, state.size(2)}).index({index}));
	}
}
