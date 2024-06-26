#include "categorical.h"

#include <ATen/ATen.h>

#include <algorithm>

using namespace drla;

Categorical::Categorical(const torch::Tensor probs, const torch::Tensor logits)
{
	if (probs.defined() == logits.defined())
	{
		throw std::invalid_argument("Either `probs` or `logits` must be specified, but not both.");
	}
	if (probs.defined())
	{
		if (probs.dim() < 1)
		{
			throw std::invalid_argument("Probs dimension must be non zero!");
		}
		param_ = probs;
		auto p = param_.clamp(1e-8, 1 - 1e-8);
		probs_ = p / p.sum(-1, true);
		logits_ = probs_.log();
	}
	else
	{
		if (logits.dim() < 1)
		{
			throw std::invalid_argument("Logits dimension must be non zero!");
		}
		param_ = logits;
		// Normalise
		logits_ = param_ - param_.logsumexp(-1, true);
		probs_ = torch::softmax(logits_, -1);
	}

	num_events_ = param_.size(-1);
	batch_shape_ = param_.sizes().vec();
	batch_shape_.resize(batch_shape_.size() - 1);
}

torch::Tensor Categorical::entropy()
{
	auto p_log_p = logits_ * probs_;
	return -p_log_p.sum(-1);
}

torch::Tensor Categorical::log_prob(torch::Tensor value)
{
	value = value.to(torch::kLong).unsqueeze(-1);
	auto broadcasted_tensors = torch::broadcast_tensors({value, logits_});
	value = broadcasted_tensors[0];
	value = value.narrow(-1, 0, 1);
	return broadcasted_tensors[1].gather(-1, value).squeeze(-1);
}

torch::Tensor Categorical::probs() const
{
	return probs_;
}

torch::Tensor Categorical::logits() const
{
	return logits_;
}

torch::Tensor Categorical::sample(c10::ArrayRef<int64_t> sample_shape)
{
	auto probs_2d = probs_.reshape({-1, num_events_});
	int samples = std::accumulate(sample_shape.begin(), sample_shape.end(), 1, std::multiplies<>());
	auto sample_2d = torch::multinomial(probs_2d, samples, true).t();
	return sample_2d.reshape(extended_shape(sample_shape));
}

torch::Tensor Categorical::mean() const
{
	return torch::full(extended_shape(), std::numeric_limits<float>::quiet_NaN(), param_.device());
}

torch::Tensor Categorical::mode() const
{
	return probs_.argmax(-1, true).reshape(extended_shape());
}

const torch::Tensor Categorical::get_action_output() const
{
	return param_;
}

MultiCategorical::MultiCategorical(
	const std::vector<int64_t>& action_shape, const torch::Tensor probs, const torch::Tensor logits)
{
	if (action_shape.empty())
	{
		throw std::invalid_argument("The action dimensions must be non zero!");
	}
	if (probs.defined() == logits.defined())
	{
		throw std::invalid_argument("Either `probs` or `logits` must be specified, but not both.");
	}
	if (probs.defined())
	{
		auto split = torch::split_with_sizes(probs, action_shape, 1);
		for (auto& split_probs : split) { category_dim_.emplace_back(split_probs, torch::Tensor{}); }
	}
	else
	{
		auto split = torch::split_with_sizes(logits, action_shape, 1);
		for (auto& split_logits : split) { category_dim_.emplace_back(torch::Tensor{}, split_logits); }
	}
}

torch::Tensor MultiCategorical::entropy()
{
	std::vector<torch::Tensor> entropy;
	for (auto& category : category_dim_) { entropy.push_back(category.entropy()); }
	return torch::stack(entropy, 1).sum(1);
}

torch::Tensor MultiCategorical::log_prob(torch::Tensor value)
{
	std::vector<torch::Tensor> values;
	auto split_values = torch::unbind(value, 1);
	for (size_t i = 0; i < category_dim_.size(); i++) { values.push_back(category_dim_[i].log_prob(split_values[i])); }

	return torch::stack(values, 1).sum(1);
}

torch::Tensor MultiCategorical::sample(c10::ArrayRef<int64_t> sample_shape)
{
	std::vector<torch::Tensor> actions;
	for (auto& category : category_dim_) { actions.push_back(category.sample(sample_shape)); }
	return torch::stack(actions, 1);
}

torch::Tensor MultiCategorical::mean() const
{
	std::vector<torch::Tensor> mean;
	for (auto& category : category_dim_) { mean.push_back(category.mean()); }
	return torch::stack(mean, 1);
}

torch::Tensor MultiCategorical::mode() const
{
	std::vector<torch::Tensor> mode;
	for (auto& category : category_dim_) { mode.push_back(category.mode()); }
	return torch::stack(mode, 1);
}

const torch::Tensor MultiCategorical::get_action_output() const
{
	std::vector<torch::Tensor> logits;
	for (auto& category : category_dim_) { logits.push_back(category.get_action_output()); }
	return torch::stack(logits, 1);
}

OneHotCategorical::OneHotCategorical(const torch::Tensor probs, const torch::Tensor logits) : Categorical(probs, logits)
{
}

torch::Tensor OneHotCategorical::log_prob(torch::Tensor value)
{
	auto indices = value.argmax(-1);
	return Categorical::log_prob(indices);
}

torch::Tensor OneHotCategorical::sample(c10::ArrayRef<int64_t> sample_shape)
{
	auto indices = Categorical::sample(sample_shape);
	return torch::one_hot(indices.to(torch::kLong), num_events_).to(probs_);
}

torch::Tensor OneHotCategorical::mode() const
{
	auto indices = Categorical::mode();
	return torch::one_hot(indices.to(torch::kLong), num_events_).to(probs_);
}
