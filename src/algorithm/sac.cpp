#include "sac.h"

#include "soft_actor_critic_model.h"
#include "utils.h"

#include <spdlog/spdlog.h>
#include <torch/torch.h>

#include <chrono>
#include <filesystem>
#include <memory>

using namespace drla;

SAC::SAC(
	const Config::AgentTrainAlgorithm& config,
	const ActionSpace& action_space,
	ReplayBuffer& buffer,
	std::shared_ptr<Model> model,
	torch::Tensor gamma)
		: config_(std::get<Config::SAC>(config))
		, action_space_(action_space)
		, buffer_(buffer)
		, model_(std::dynamic_pointer_cast<SoftActorCriticModel>(model))
		, log_ent_coef_(torch::ones(1, buffer_.get_device()).log().requires_grad_())
		, actor_optimiser_(
				model_->actor_parameters(), torch::optim::AdamOptions(config_.learning_rate).eps(config_.epsilon))
		, critic_optimiser_(
				model_->critic_parameters(), torch::optim::AdamOptions(config_.learning_rate).eps(config_.epsilon))
		, ent_coef_optimiser_({log_ent_coef_}, torch::optim::AdamOptions(config_.learning_rate).eps(config_.epsilon))
		, gamma_(gamma)
		, target_entropy_(-static_cast<double>(action_space_.shape.size()))
{
	if (is_action_discrete(action_space_))
	{
		target_entropy_ = -config_.target_entropy_scale *
											std::log(1.0 / std::accumulate(action_space_.shape.begin(), action_space_.shape.end(), 0));
	}
	model_->train();
}

std::string SAC::name() const
{
	return "SAC";
}

std::vector<UpdateResult> SAC::update(int timestep)
{
	update_learning_rate(timestep);

	float ent_coefs = 0.0F;
	float ent_coef_losses = 0.0F;
	float critic_losses = 0.0F;
	float actor_losses = 0.0F;

	for (int gradient_step = 0; gradient_step < config_.gradient_steps; gradient_step++)
	{
		auto replay_data = buffer_.sample(config_.batch_size);

		// Action by the current actor for the sampled state
		auto action_output = model_->action_output(replay_data.observations);

		// The entropy coefficient or entropy can be learned automatically see Automating Entropy Adjustment for Maximum
		// Entropy RL section of https://arxiv.org/abs/1812.05905
		auto ent_coef = log_ent_coef_.detach().exp();
		ent_coefs += ent_coef.item<float>();
		auto ent_coef_loss = -(log_ent_coef_ * (action_output.log_prob + target_entropy_).detach());
		if (is_action_discrete(action_space_))
		{
			auto probs = torch::softmax(action_output.actions_pi, -1);
			ent_coef_loss *= probs.detach();
		}
		ent_coef_loss = ent_coef_loss.mean();
		ent_coef_losses += ent_coef_loss.item<float>();

		// Optimize entropy coefficient, also called entropy temperature or alpha in the paper
		ent_coef_optimiser_.zero_grad();
		ent_coef_loss.backward();
		ent_coef_optimiser_.step();

		torch::Tensor target_q_values;
		{
			torch::NoGradGuard no_grad;
			// Select action according to policy
			auto next_action_output = model_->action_output(replay_data.next_observations);
			// Compute the next Q values: min over all critics targets
			auto next_q_values =
				torch::stack(model_->critic_target(replay_data.next_observations, next_action_output.actions));
			next_q_values = std::get<0>(torch::min(next_q_values, 0));
			if (is_action_discrete(action_space_))
			{
				// get expected value and add entropy term
				auto next_probs = torch::softmax(next_action_output.actions_pi, -1);
				next_q_values = (next_probs * (next_q_values - ent_coef * next_action_output.log_prob)).sum(-1, true);
			}
			else
			{
				// add entropy term
				next_q_values -= ent_coef * next_action_output.log_prob;
			}
			// td error
			target_q_values = replay_data.rewards + replay_data.episode_non_terminal * gamma_ * next_q_values;
		}

		// Get current Q-values estimates for each critic network using action from the replay buffer
		auto current_q_values_list = model_->critic(replay_data.observations, replay_data.actions);
		torch::Tensor critic_loss = torch::zeros({1}, target_q_values.device());
		for (auto& current_q_values : current_q_values_list)
		{
			if (is_action_discrete(action_space_))
			{
				current_q_values = current_q_values.gather(1, replay_data.actions.to(torch::kLong));
			}
			// Compute critic loss
			critic_loss += config_.value_loss_coef * torch::nn::functional::mse_loss(current_q_values, target_q_values);
		}

		critic_losses += critic_loss.item<float>();

		// Optimize the critic
		critic_optimiser_.zero_grad();
		critic_loss.backward();
		critic_optimiser_.step();

		// Compute actor loss
		torch::Tensor min_qf_pi;
		{
			torch::NoGradGuard no_grad;
			auto q_values_pi = model_->critic(replay_data.observations, action_output.actions_pi);
			min_qf_pi = std::get<0>(torch::min(torch::stack(q_values_pi), 0));
		}
		auto actor_loss = config_.actor_loss_coef * (ent_coef * action_output.log_prob - min_qf_pi);
		if (is_action_discrete(action_space_))
		{
			auto probs = torch::softmax(action_output.actions_pi, -1);
			actor_loss *= probs;
		}
		actor_loss = actor_loss.mean();
		actor_losses += actor_loss.item<float>();

		// Optimize the actor
		actor_optimiser_.zero_grad();
		actor_loss.backward();
		actor_optimiser_.step();
	}

	// Update target networks
	if (timestep % config_.target_update_interval == 0)
	{
		model_->update(config_.tau);
	}

	ent_coefs /= static_cast<float>(config_.gradient_steps);
	ent_coef_losses /= static_cast<float>(config_.gradient_steps);
	critic_losses /= static_cast<float>(config_.gradient_steps);
	actor_losses /= static_cast<float>(config_.gradient_steps);

	return {
		{TrainResultType::kValueLoss, critic_losses},
		{TrainResultType::kPolicyLoss, actor_losses},
		{TrainResultType::kEntropyLoss, ent_coef_losses},
		{TrainResultType::kLearningRate, static_cast<float>(lr_param_)},
		{TrainResultType::kEntropyCoeficients, ent_coefs},
	};
}

void SAC::save(const std::filesystem::path& path) const
{
	torch::save(actor_optimiser_, path / "actor_optimiser.pt");
	torch::save(critic_optimiser_, path / "critic_optimiser.pt");
	torch::save(ent_coef_optimiser_, path / "ent_coef_optimiser.pt");
	model_->save(path);
}

void SAC::load(const std::filesystem::path& path)
{
	auto opt_actor_path = path / "actor_optimiser.pt";
	auto opt_critic_path = path / "critic_optimiser.pt";
	auto opt_ent_coef_path = path / "ent_coef_optimiser.pt";
	if (std::filesystem::exists(opt_actor_path) && std::filesystem::exists(opt_critic_path))
	{
		torch::load(actor_optimiser_, opt_actor_path);
		torch::load(critic_optimiser_, opt_critic_path);
		torch::load(ent_coef_optimiser_, opt_ent_coef_path);
		spdlog::info("Optimisers loaded");
	}
	model_->load(path);
	model_->train();
}

void SAC::update_learning_rate(int timestep)
{
	double alpha = learning_rate_decay(&config_, timestep, config_.total_timesteps);
	lr_param_ = config_.learning_rate * alpha;
	for (auto& group : actor_optimiser_.param_groups())
	{
		if (group.has_options())
		{
			dynamic_cast<torch::optim::AdamOptions&>(group.options()).lr(lr_param_);
		}
	}
	for (auto& group : critic_optimiser_.param_groups())
	{
		if (group.has_options())
		{
			dynamic_cast<torch::optim::AdamOptions&>(group.options()).lr(lr_param_);
		}
	}
	for (auto& group : ent_coef_optimiser_.param_groups())
	{
		if (group.has_options())
		{
			dynamic_cast<torch::optim::AdamOptions&>(group.options()).lr(lr_param_);
		}
	}
}