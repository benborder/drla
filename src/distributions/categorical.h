#pragma once

#include "distribution.h"

#include <c10/util/ArrayRef.h>
#include <torch/torch.h>

#include <optional>

namespace drla
{

// TODO: add mask to ignore certain categories
class Categorical : public Distribution
{
public:
	Categorical(std::optional<torch::Tensor> probs, std::optional<torch::Tensor> logits);

	torch::Tensor entropy() override;
	torch::Tensor action_log_prob(torch::Tensor action) override;
	torch::Tensor sample(bool deterministic, c10::ArrayRef<int64_t> sample_shape = {}) override;
	const torch::Tensor get_action_output() const override;

private:
	torch::Tensor probs_;
	torch::Tensor param_;
	torch::Tensor logits_;
	int num_events_;
};

class MultiCategorical : public Distribution
{
public:
	MultiCategorical(
		const std::vector<int64_t>& action_shape, std::optional<torch::Tensor> probs, std::optional<torch::Tensor> logits);

	torch::Tensor entropy() override;
	torch::Tensor action_log_prob(torch::Tensor action) override;
	torch::Tensor sample(bool deterministic, c10::ArrayRef<int64_t> sample_shape = {}) override;
	const torch::Tensor get_action_output() const override;

private:
	std::vector<Categorical> category_dim_;
};

} // namespace drla
