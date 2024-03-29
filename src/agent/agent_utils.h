#pragma once

#include "configuration.h"

#include <ATen/core/Tensor.h>

namespace drla
{

inline torch::Tensor clamp_reward(const torch::Tensor& reward, const Config::Rewards& config)
{
	torch::Tensor clamped_reward = reward.clone();
	if (config.reward_clamp_min != 0)
	{
		clamped_reward.clamp_max_(-config.reward_clamp_min);
	}
	if (config.reward_clamp_max != 0)
	{
		clamped_reward.clamp_min_(-config.reward_clamp_max);
	}
	return clamped_reward;
}

} // namespace drla
