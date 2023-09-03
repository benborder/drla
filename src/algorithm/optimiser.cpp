#include "optimiser.h"

using namespace drla;

Optimiser::Optimiser(
	const Config::Optimiser& optimiser_config, const std::vector<torch::Tensor>& params, int max_timesteps)
		: config_(optimiser_config)
		, max_timesteps_(max_timesteps)
		, optimiser_(std::visit(
				[&params](const auto& config) -> std::unique_ptr<torch::optim::Optimizer> {
					using T = std::decay_t<decltype(config)>;
					if constexpr (std::is_same_v<Config::OptimiserAdam, T>)
					{
						return std::make_unique<torch::optim::Adam>(
							params,
							torch::optim::AdamOptions(config.learning_rate).eps(config.epsilon).weight_decay(config.weight_decay));
					}
					if constexpr (std::is_same_v<Config::OptimiserSGD, T>)
					{
						return std::make_unique<torch::optim::SGD>(
							params,
							torch::optim::SGDOptions(config.learning_rate)
								.momentum(config.momentum)
								.weight_decay(config.weight_decay)
								.dampening(config.dampening));
					}
					if constexpr (std::is_same_v<Config::OptimiserRMSProp, T>)
					{
						return std::make_unique<torch::optim::RMSprop>(
							params,
							torch::optim::RMSpropOptions(config.learning_rate)
								.alpha(config.alpha)
								.eps(config.epsilon)
								.weight_decay(config.weight_decay)
								.momentum(config.momentum));
					}
				},
				optimiser_config))
{
}

std::tuple<double, double> Optimiser::update(int timestep)
{
	const double progress = static_cast<double>(timestep) / static_cast<double>(max_timesteps_);
	double ratio = 1.0;
	double lr = 0.0;
	std::visit(
		[&](const auto& config) {
			switch (config.lr_schedule_type)
			{
				case LearningRateScheduleType::kLinear:
				{
					ratio = std::max<double>(0.0, 1.0 - config.lr_decay_rate * progress);
					break;
				}
				case LearningRateScheduleType::kExponential:
				{
					ratio = std::max<double>(0.0, std::exp(-0.5 * M_PI * config.lr_decay_rate * progress));
					break;
				}
				case LearningRateScheduleType::kConstant: break;
			}
			lr = ratio * config.learning_rate;
			if (config.learning_rate_min > 0)
			{
				lr = std::max(lr, config.learning_rate_min);
			}
		},
		config_);
	for (auto& group : optimiser_->param_groups())
	{
		if (group.has_options())
		{
			group.options().set_lr(lr);
		}
	}
	return {ratio, lr};
}

void Optimiser::step(torch::Tensor& loss)
{
	assert(!at::isnan(loss).item<bool>());
	optimiser_->zero_grad();
	loss.backward();
	auto& params = optimiser_->param_groups().front().params();
	assert(!params.empty());
	std::visit(
		[&](const auto& config) {
			if (config.grad_clip > 0.0)
			{
				torch::nn::utils::clip_grad_value_(params, config.grad_clip);
			}
			if (config.grad_norm_clip > 0.0)
			{
				torch::nn::utils::clip_grad_norm_(params, config.grad_norm_clip);
			}
		},
		config_);
	optimiser_->step();
}

torch::optim::Optimizer& Optimiser::get_optimiser() const
{
	return *optimiser_.get();
}
