#include "muzero.h"

#include "muzero_model.h"
#include "utils.h"

#include <spdlog/spdlog.h>
#include <torch/torch.h>

#include <chrono>
#include <filesystem>
#include <memory>

using namespace drla;

MuZero::MuZero(
	const Config::AgentTrainAlgorithm& config, std::shared_ptr<MCTSModelInterface> model, EpisodicPERBuffer& buffer)
		: config_(std::get<Config::MuZero>(config)), model_(model), buffer_(buffer)
{
	switch (config_.optimiser)
	{
		case OptimiserType::kAdam:
		{
			optimiser_ = std::make_unique<torch::optim::Adam>(
				model_->parameters(),
				torch::optim::AdamOptions(config_.learning_rate).eps(config_.epsilon).weight_decay(config_.weight_decay));
			break;
		}
		case OptimiserType::kSGD:
		{
			optimiser_ = std::make_unique<torch::optim::SGD>(
				model_->parameters(),
				torch::optim::SGDOptions(config_.learning_rate).momentum(config_.momentum).weight_decay(config_.weight_decay));
			break;
		}
	}

	model_->train();
}

std::string MuZero::name() const
{
	return "MuZero";
}

std::vector<UpdateResult> MuZero::update(int timestep)
{
	update_learning_rate(timestep);

	auto device = model_->parameters().front().device();
	auto batch = buffer_.sample(config_.batch_size, device);

	auto target_values_scalar = batch.values.clone().detach().to(torch::kCPU);
	batch.values = model_->scalar_to_support(batch.values);
	batch.reward = model_->scalar_to_support(batch.reward);

	auto prediction = std::dynamic_pointer_cast<Model>(model_)->predict(batch.observation);

	std::vector<PredictOutput> predictions{prediction};

	for (int i = 1, ilen = batch.action.size(1); i < ilen; ++i)
	{
		prediction.action = batch.action.narrow(1, i, 1).squeeze(1);
		prediction = model_->predict(prediction);
		for (auto& hidden_state : prediction.state)
		{
			hidden_state.register_hook([](torch::Tensor grad) { return grad * 0.5F; });
		}
		predictions.push_back(prediction);
		// The state and actions aren't required to be stored to calculate the loss, so save VRAM by removing it
		predictions.back().state = {};
	}

	torch::Tensor value_loss = torch::zeros({config_.batch_size}, device);
	torch::Tensor reward_loss = torch::zeros({config_.batch_size}, device);
	torch::Tensor policy_loss = torch::zeros({config_.batch_size}, device);
	torch::Tensor priorities = torch::zeros_like(target_values_scalar);
	priorities.set_requires_grad(false);

	auto calculate_loss = [&](int index) -> std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> {
		using namespace torch::nn::functional;
		auto& pred = predictions[index];
		auto value_loss = (-batch.values.narrow(1, index, 1).squeeze(1) * log_softmax(pred.values, 1)).sum(-1);
		auto reward_loss = (-batch.reward.narrow(1, index, 1).squeeze(1) * log_softmax(pred.reward, 1)).sum(-1);
		auto policy_loss = (-batch.policy.narrow(1, index, 1).squeeze(1) * log_softmax(pred.policy, 1)).sum(-1);
		return {value_loss, reward_loss, policy_loss};
	};

	for (int i = 0, ilen = static_cast<size_t>(predictions.size()); i < ilen; ++i)
	{
		auto [current_value_loss, current_reward_loss, current_policy_loss] = calculate_loss(i);

		if (i > 0)
		{
			auto gradient_scale = batch.gradient_scale.narrow(1, i, 1).squeeze(1);
			current_value_loss.register_hook([gradient_scale](torch::Tensor grad) { return grad / gradient_scale; });
			current_reward_loss.register_hook([gradient_scale](torch::Tensor grad) { return grad / gradient_scale; });
			current_policy_loss.register_hook([gradient_scale](torch::Tensor grad) { return grad / gradient_scale; });
			// Ignore the reward loss for the initial prediction
			reward_loss += current_reward_loss;
		}

		value_loss += current_value_loss;
		policy_loss += current_policy_loss;

		// Calculate priorities for PER buffer
		{
			auto pred_value = model_->support_to_scalar(predictions[i].values).detach().to(torch::kCPU).squeeze();
			priorities.narrow(1, i, 1).squeeze() =
				torch::pow((pred_value - target_values_scalar.narrow(1, i, 1).squeeze()).abs(), config_.per_alpha)
					// Clamping to 1e-8 to avoid 0 as this can lead to divide by zero issues in some circumstances
					.clamp_min(1e-8);
		}
	}

	auto loss = value_loss * config_.value_loss_weight + reward_loss + policy_loss;
	// Correct for PER bias by using importance sampling of weights
	loss *= batch.weight;
	loss = loss.mean();
	assert(!at::isnan(loss).item<bool>());

	// Optimise
	optimiser_->zero_grad();
	loss.backward();
	optimiser_->step();

	buffer_.update_priorities(priorities, batch.indicies);

	return {
		{TrainResultType::kLoss, loss.item<float>()},
		{TrainResultType::kValueLoss, value_loss.mean().item<float>()},
		{TrainResultType::kRewardLoss, reward_loss.mean().item<float>()},
		{TrainResultType::kPolicyLoss, policy_loss.mean().item<float>()},
		{TrainResultType::kLearningRate, lr_}};
}

void MuZero::save(const std::filesystem::path& path) const
{
	torch::save(*optimiser_, path / "optimiser.pt");
	model_->save(path);
}

void MuZero::load(const std::filesystem::path& path)
{
	auto opt_path = path / "optimiser.pt";
	if (std::filesystem::exists(opt_path))
	{
		torch::load(*optimiser_, opt_path);
		spdlog::info("Optimiser loaded");
	}
	model_->load(path);
	model_->train();
}

void MuZero::update_learning_rate(int timestep)
{
	double alpha = learning_rate_decay(&config_, timestep, config_.total_timesteps);
	lr_ = config_.learning_rate * alpha;
	for (auto& group : optimiser_->param_groups())
	{
		if (group.has_options())
		{
			switch (config_.optimiser)
			{
				case OptimiserType::kAdam: dynamic_cast<torch::optim::AdamOptions&>(group.options()).lr(lr_); break;
				case OptimiserType::kSGD: dynamic_cast<torch::optim::SGDOptions&>(group.options()).lr(lr_); break;
			}
		}
	}
}
