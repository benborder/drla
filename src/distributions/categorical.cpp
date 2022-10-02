#include "categorical.h"

#include <c10/util/ArrayRef.h>
#include <torch/torch.h>

using namespace drla;

Categorical::Categorical(std::optional<torch::Tensor> probs, std::optional<torch::Tensor> logits)
{
	if (probs.has_value() == logits.has_value())
	{
		throw std::runtime_error("Either `probs` or `logits` must be specified, but not both.");
	}
	if (probs.has_value())
	{
		if (probs->dim() < 1)
		{
			throw std::runtime_error("Probs dimension must be non zero!");
		}
		param_ = *probs;
		auto p = param_.clamp(1e-8, 1 - 1e-8);
		probs_ = p / p.sum(-1, true);
		logits_ = probs_.log();
	}
	else
	{
		if (logits->dim() < 1)
		{
			throw std::runtime_error("Logits dimension must be non zero!");
		}
		param_ = *logits;
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

torch::Tensor Categorical::action_log_prob(torch::Tensor action)
{
	action = action.to(torch::kLong).unsqueeze(-1);
	auto broadcasted_tensors = torch::broadcast_tensors({action, logits_});
	action = broadcasted_tensors[0];
	action = action.narrow(-1, 0, 1);
	return broadcasted_tensors[1].gather(-1, action).squeeze(-1);
}

torch::Tensor Categorical::sample(bool deterministic, c10::ArrayRef<int64_t> sample_shape)
{
	if (deterministic)
	{
		return probs_.argmax(-1, true);
	}
	auto ext_sample_shape = extended_shape(sample_shape);
	auto param_shape = ext_sample_shape;
	param_shape.insert(param_shape.end(), {num_events_});
	auto exp_probs = probs_.expand(param_shape);
	torch::Tensor probs_2d;
	if (probs_.dim() == 1 || probs_.size(0) == 1)
	{
		probs_2d = exp_probs.view({-1, num_events_});
	}
	else
	{
		probs_2d = exp_probs.contiguous().view({-1, num_events_});
	}
	auto sample_2d = torch::multinomial(probs_2d, 1, true);
	return sample_2d.contiguous().view(ext_sample_shape);
}

const torch::Tensor Categorical::get_action_output() const
{
	return param_;
}

MultiCategorical::MultiCategorical(
	const std::vector<int64_t>& action_shape, std::optional<torch::Tensor> probs, std::optional<torch::Tensor> logits)
{
	if (action_shape.empty())
	{
		throw std::runtime_error("The action dimensions must be non zero!");
	}
	if (probs.has_value() == logits.has_value())
	{
		throw std::runtime_error("Either `probs` or `logits` must be specified, but not both.");
	}
	if (probs.has_value())
	{
		auto split = torch::split_with_sizes(*probs, action_shape, 1);
		for (auto& split_probs : split) { category_dim_.emplace_back(split_probs, std::nullopt); }
	}
	else
	{
		auto split = torch::split_with_sizes(*logits, action_shape, 1);
		for (auto& split_logits : split) { category_dim_.emplace_back(std::nullopt, split_logits); }
	}
}

torch::Tensor MultiCategorical::entropy()
{
	std::vector<torch::Tensor> entropy;
	for (auto& category : category_dim_) { entropy.push_back(category.entropy()); }
	return torch::stack(entropy, 1).sum(1);
}

torch::Tensor MultiCategorical::action_log_prob(torch::Tensor action)
{
	std::vector<torch::Tensor> actions;
	auto split_actions = torch::unbind(action, 1);
	for (size_t i = 0; i < category_dim_.size(); i++)
	{
		actions.push_back(category_dim_[i].action_log_prob(split_actions[i]));
	}

	return torch::stack(actions, 1).sum(1);
}

torch::Tensor MultiCategorical::sample(bool deterministic, c10::ArrayRef<int64_t> sample_shape)
{
	std::vector<torch::Tensor> actions;
	for (auto& category : category_dim_) { actions.push_back(category.sample(deterministic, sample_shape)); }
	return torch::stack(actions, 1);
}

const torch::Tensor MultiCategorical::get_action_output() const
{
	std::vector<torch::Tensor> logits;
	for (auto& category : category_dim_) { logits.push_back(category.get_action_output()); }
	return torch::stack(logits, 1);
}
