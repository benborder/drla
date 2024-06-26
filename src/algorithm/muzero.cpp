#include "muzero.h"

#include "muzero_model.h"
#include "utils.h"

#include <spdlog/spdlog.h>
#include <torch/nn/functional.h>
#include <torch/serialize.h>

#include <chrono>
#include <filesystem>
#include <memory>

using namespace drla;

MuZero::MuZero(
	const Config::AgentTrainAlgorithm& config, std::shared_ptr<MCTSModelInterface> model, MCTSReplayBuffer& buffer)
		: config_(std::get<Config::MuZero::TrainConfig>(config))
		, model_(model)
		, buffer_(buffer)
		, optimiser_(config_.optimiser, model_->parameters(), config_.total_timesteps)
{
	model_->train();
}

std::string MuZero::name() const
{
	return "MuZero";
}

Metrics MuZero::update(int timestep)
{
	auto device = model_->parameters().front().device();
	auto batch = buffer_.sample(config_.batch_size, device);

	auto target_values_scalar = batch.values.clone().detach().to(torch::kCPU);
	batch.values = model_->scalar_to_support(batch.values);
	batch.reward = model_->scalar_to_support(batch.reward);

	auto prediction = model_->predict({batch.observation});

	std::vector<ModelOutput> predictions{prediction};

	for (int i = 1, ilen = batch.action.size(1); i < ilen; ++i)
	{
		prediction.action = batch.action.narrow(1, i, 1).squeeze(1);
		prediction = model_->predict_recurrent(prediction);
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

	// Optimise
	optimiser_.update(timestep);
	optimiser_.step(loss);

	buffer_.update_priorities(priorities, batch.indicies);

	Metrics metrics;
	metrics.add({"loss", TrainResultType::kLoss, loss.item<float>()});
	metrics.add({"loss_value", TrainResultType::kLoss, value_loss.mean().item<float>()});
	metrics.add({"loss_reward", TrainResultType::kLoss, reward_loss.mean().item<float>()});
	metrics.add({"loss_policy", TrainResultType::kLoss, policy_loss.mean().item<float>()});
	metrics.add({"learning_rate", TrainResultType::kLearningRate, optimiser_.get_lr()});
	return metrics;
}

void MuZero::save(const std::filesystem::path& path) const
{
	torch::save(optimiser_.get_optimiser(), path / "optimiser.pt");
	model_->save(path);
}

void MuZero::load(const std::filesystem::path& path)
{
	auto opt_path = path / "optimiser.pt";
	if (std::filesystem::exists(opt_path))
	{
		torch::load(optimiser_.get_optimiser(), opt_path);
		spdlog::info("Optimiser loaded");
	}
	model_->load(path);
	model_->train();
}
