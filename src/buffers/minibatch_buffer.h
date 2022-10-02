#pragma once

#include "rollout_buffer.h"
#include "types.h"

#include <torch/torch.h>

#include <cstddef>

namespace drla
{

struct MiniBatch
{
	Observations observations;
	torch::Tensor actions;
	torch::Tensor old_values;
	torch::Tensor returns;
	torch::Tensor old_log_probs;
	torch::Tensor advantages;
};

class MiniBatchBuffer
{
public:
	struct Iterator
	{
		using iterator_category = std::forward_iterator_tag;
		using difference_type = std::ptrdiff_t;
		using value_type = MiniBatch;
		using pointer = MiniBatch*;
		using reference = MiniBatch&;

		Iterator(const RolloutBuffer& buffer, torch::Tensor& indices, int index)
				: buffer_(buffer), indices_(indices), index_(index)
		{
			if (index < indices_.size(0))
			{
				get_minibatch();
			}
		}

		const MiniBatch& operator*() const { return minibatch_; }

		pointer operator->() { return &minibatch_; }

		Iterator& operator++()
		{
			index_++;
			return *this;
		}

		friend bool operator==(const Iterator& a, const Iterator& b) { return a.index_ == b.index_; };

		friend bool operator!=(const Iterator& a, const Iterator& b) { return a.index_ != b.index_; };

	private:
		void get_minibatch();

	private:
		const RolloutBuffer& buffer_;
		torch::Tensor indices_;
		int index_;
		MiniBatch minibatch_;
	};

	MiniBatchBuffer(const RolloutBuffer& buffer, int mini_batch_size);

	Iterator begin();
	Iterator end();
	size_t size() const;

private:
	const RolloutBuffer& buffer_;
	torch::Tensor indices_;
};

} // namespace drla
