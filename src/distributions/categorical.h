#pragma once

#include "distribution.h"

#include <c10/util/ArrayRef.h>
#include <torch/torch.h>

namespace drla
{

class Categorical : public Distribution
{
public:
	Categorical(const torch::Tensor probs = {}, const torch::Tensor logits = {});

	torch::Tensor entropy() override;
	torch::Tensor log_prob(torch::Tensor value) override;
	torch::Tensor probs() const;
	torch::Tensor logits() const;
	torch::Tensor sample(c10::ArrayRef<int64_t> sample_shape = {}) override;
	torch::Tensor mean() const override;
	torch::Tensor mode() const override;
	const torch::Tensor get_action_output() const override;

protected:
	torch::Tensor probs_;
	torch::Tensor param_;
	torch::Tensor logits_;
	int num_events_;
};

class MultiCategorical : public Distribution
{
public:
	MultiCategorical(
		const std::vector<int64_t>& action_shape, const torch::Tensor probs = {}, const torch::Tensor logits = {});

	torch::Tensor entropy() override;
	torch::Tensor log_prob(torch::Tensor value) override;
	torch::Tensor sample(c10::ArrayRef<int64_t> sample_shape = {}) override;
	torch::Tensor mean() const override;
	torch::Tensor mode() const override;
	const torch::Tensor get_action_output() const override;

private:
	std::vector<Categorical> category_dim_;
};

class OneHotCategorical : public Categorical
{
public:
	OneHotCategorical(const torch::Tensor probs = {}, const torch::Tensor logits = {});

	torch::Tensor log_prob(torch::Tensor value) override;
	torch::Tensor sample(c10::ArrayRef<int64_t> sample_shape = {}) override;
	torch::Tensor mode() const override;
};

} // namespace drla
