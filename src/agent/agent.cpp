#include "agent.h"

#include "actor_critic_model.h"
#include "interactive_agent.h"
#include "off_policy_agent.h"
#include "on_policy_agent.h"
#include "qnet_model.h"
#include "random_model.h"
#include "threadpool.h"

#include <spdlog/spdlog.h>

#include <filesystem>

using namespace drla;

Agent::Agent(
	const Config::Agent& config,
	EnvironmentManager* environment_manager,
	AgentCallbackInterface* callback,
	std::filesystem::path data_path)
		: base_config_(std::visit([](const auto& config) { return static_cast<const Config::AgentBase>(config); }, config))
		, device_(torch::kCPU)
		, model_(nullptr)
		, environment_manager_(environment_manager)
		, agent_callback_(callback)
		, data_path_(std::move(data_path))
{
	if (environment_manager == nullptr)
	{
		throw std::runtime_error("An environment manager interface must be provided.");
	}
	if (agent_callback_ == nullptr)
	{
		throw std::runtime_error("A callback interface must be provided.");
	}

	if (std::visit([&](auto& config) { return config.use_cuda; }, config))
	{
		if (torch::cuda::is_available())
		{
			// Note the first time a CUDA operation is run it takes a second or so to initialise the cuda libraries
			// All subsequent usage is significantly faster
			spdlog::info("Using CUDA device!");
			spdlog::debug("{:<8}{}", "CUDA:", at::detail::getCUDAHooks().versionCUDART());
			spdlog::debug("{:<8}{}", "CuDNN:", at::detail::getCUDAHooks().versionCuDNN());
			device_ = torch::kCUDA;
		}
		else
		{
			spdlog::warn("CUDA unavailable!");
		}
	}
}

Agent::~Agent()
{
}

void Agent::stop_train()
{
	training_ = false;
}

void Agent::set_data_path(const std::filesystem::path& path)
{
	data_path_ = path;
}

std::filesystem::path Agent::get_save_path(const std::string& postfix) const
{
	auto path = data_path_;
	if (!postfix.empty())
	{
		path = path / postfix;
		std::filesystem::create_directory(path);
	}
	return path;
}

void Agent::make_environments(ThreadPool& threadpool, int env_count)
{
	if (static_cast<int>(envs_.size()) >= env_count)
	{
		return;
	}
	envs_.resize(env_count);
	for (size_t i = env_count - envs_.size(); i < envs_.size(); i++)
	{
		threadpool.queue_task([&, i]() { envs_[i] = environment_manager_->make_environment(); });
	}
}

void Agent::run(const std::vector<State>& initial_state, RunOptions options)
{
	const int env_count = initial_state.size();
	if (env_count <= 0)
	{
		spdlog::error("The number of environments must be greater than 0!");
		return;
	}
	if (options.max_steps <= 0)
	{
		// Run the environment until it terminates (setting to max int should be sufficient)
		options.max_steps = std::numeric_limits<int>::max();
	}

	// Create environments and allocate threads
	ThreadPool threadpool(base_config_.env_count);
	make_environments(threadpool, env_count);

	std::vector<StepResult> step_results;
	step_results.resize(env_count);

	// Wait for environments to complete initialising
	threadpool.wait_queue_empty();

	for (int env = 0; env < env_count; ++env)
	{
		threadpool.queue_task([&, env]() { step_results[env] = envs_[env]->reset(initial_state[env]); });
	}

	// Wait for environment reset to complete
	threadpool.wait_queue_empty();

	load_model(options.force_model_reload);

	auto env_config = envs_.front()->get_configuration();

	model_->eval();

	for (int env = 0; env < env_count; env++)
	{
		threadpool.queue_task([&, env]() {
			auto& environment = envs_[env];
			// Get initial observation and state, running a env_step callback to update externally
			StepData step_data;
			step_data.env = env;
			step_data.step = 0;
			step_data.step_result = std::move(step_results[env]);
			step_data.predict_result.action = torch::zeros(env_config.action_space.shape);
			step_data.reward = step_data.step_result.reward.clone();

			auto agent_reset_config = agent_callback_->env_reset(step_data);
			if (agent_reset_config.stop)
			{
				return;
			}
			options.capture_observations |= agent_reset_config.raw_capture;
			if (options.capture_observations)
			{
				step_data.raw_observation = environment->get_raw_observations();
			}

			bool stop = agent_callback_->env_step(step_data);
			if (stop)
			{
				return;
			}

			torch::NoGradGuard no_grad;

			for (int step = 0; step < options.max_steps; step++)
			{
				auto observation = step_data.step_result.observation;
				for (auto& obs : observation) { obs = obs.unsqueeze(0).to(device_); }

				step_data.step = step;
				step_data.predict_result = model_->predict(observation, options.deterministic);
				step_data.step_result = environment->step(step_data.predict_result.action);
				step_data.reward = step_data.step_result.reward.clone();
				if (base_config_.rewards.reward_clamp_min != 0)
				{
					step_data.reward.clamp_max_(-base_config_.rewards.reward_clamp_min);
				}
				if (base_config_.rewards.reward_clamp_max != 0)
				{
					step_data.reward.clamp_min_(-base_config_.rewards.reward_clamp_max);
				}
				if (options.capture_observations)
				{
					step_data.raw_observation = environment->get_raw_observations();
				}

				bool stop = agent_callback_->env_step(step_data);
				if (stop)
				{
					break;
				}

				if (step_data.step_result.state.episode_end)
				{
					environment->reset(initial_state[env]);
					auto agent_reset_config = agent_callback_->env_reset(step_data);
					if (agent_reset_config.stop)
					{
						break;
					}
				}
			}
		});
	}

	// Wait for all environments to finish their batch
	threadpool.wait_queue_empty();
}

PredictOutput Agent::predict_action(const Observations& input_observations, bool deterministic)
{
	if (model_ == nullptr)
	{
		spdlog::error("No model loaded! Call 'load_model()' at least once before 'predict_action()' is called.");
		throw std::runtime_error("No model loaded!");
	}

	torch::NoGradGuard no_grad;

	Observations observations;
	for (auto& obs : input_observations) { observations.push_back(obs.unsqueeze(0).to(device_)); }

	return model_->predict(observations, deterministic);
}

void Agent::reset()
{
	training_ = false;
	model_.reset();
	envs_.clear();
}

void Agent::train()
{
	spdlog::error("Training not available in default agent mode!");
}

void Agent::load_model(bool force_reload)
{
	if (force_reload || model_ == nullptr)
	{
		// If there are no envs, then make a single env to get the required information to initialise the model
		if (envs_.empty())
		{
			envs_.push_back(environment_manager_->make_environment());
		}

		// Observation shape and action space are the same for all envs
		const auto env_config = envs_.front()->get_configuration();
		int reward_shape = base_config_.rewards.combine_rewards ? 1 : static_cast<int>(env_config.reward_types.size());

		switch (base_config_.model_type)
		{
			case AgentPolicyModelType::kRandom:
			{
				model_ = std::make_shared<RandomModel>(base_config_.model, env_config.action_space, reward_shape);
				break;
			}
			case AgentPolicyModelType::kActorCritic:
			{
				model_ = std::make_shared<ActorCriticModel>(base_config_.model, env_config, reward_shape);
				break;
			}
			case AgentPolicyModelType::kQNet:
			{
				model_ = std::make_shared<QNetModel>(base_config_.model, env_config, reward_shape);
				break;
			}
			default:
			{
				spdlog::error("Invalid model type selected!");
				return;
			}
		}

		model_->load(data_path_);
		model_->to(device_);
	}
}

std::unique_ptr<Agent> drla::make_agent(
	const Config::Agent& config,
	EnvironmentManager* environment_manager,
	AgentCallbackInterface* callback,
	const std::filesystem::path& data_path)
{
	// If no agent is defined then use the interactive agent with a warning
	if (config.valueless_by_exception())
	{
		spdlog::warn("No agent defined, using interactive agent");
		return std::make_unique<InteractiveAgent>(config, environment_manager, callback, data_path);
	}
	if (std::holds_alternative<Config::InteractiveAgent>(config))
	{
		return std::make_unique<InteractiveAgent>(config, environment_manager, callback, data_path);
	}
	if (std::holds_alternative<Config::OnPolicyAgent>(config))
	{
		return std::make_unique<OnPolicyAgent>(config, environment_manager, callback, data_path);
	}
	if (std::holds_alternative<Config::OffPolicyAgent>(config))
	{
		return std::make_unique<OffPolicyAgent>(config, environment_manager, callback, data_path);
	}

	return std::make_unique<Agent>(config, environment_manager, callback, data_path);
}
