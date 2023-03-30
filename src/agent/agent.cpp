#include "agent.h"

#include "actor_critic_model.h"
#include "interactive_agent.h"
#include "mcts_agent.h"
#include "muzero_model.h"
#include "off_policy_agent.h"
#include "on_policy_agent.h"
#include "qnet_model.h"
#include "random_model.h"
#include "soft_actor_critic_model.h"
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
		, devices_({torch::kCPU})
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

	auto cuda_devices = std::visit([&](auto& config) { return config.cuda_devices; }, config);
	auto max_cuda_device_count = static_cast<int>(torch::cuda::device_count());
	if (!cuda_devices.empty() && max_cuda_device_count > 0)
	{
		// Note the first time a CUDA operation is run it takes a second or so to initialise the cuda libraries
		// All subsequent usage is significantly faster
		devices_.clear();
		if (cuda_devices.back() == -1)
		{
			for (const auto index : c10::irange(max_cuda_device_count))
			{
				devices_.emplace_back(torch::kCUDA, static_cast<torch::DeviceIndex>(index));
			}
		}
		else
		{
			for (const auto index : cuda_devices)
			{
				if (index < max_cuda_device_count && index >= 0)
				{
					devices_.emplace_back(torch::kCUDA, static_cast<torch::DeviceIndex>(index));
				}
				else
				{
					spdlog::error("CUDA device {} not available", index);
				}
			}
		}

		spdlog::info("Using CUDA {} devices", devices_.size());
		spdlog::debug("{:<8}{}", "CUDA:", at::detail::getCUDAHooks().versionCUDART());
		spdlog::debug("{:<8}{}", "CuDNN:", at::detail::getCUDAHooks().versionCuDNN());
	}
	else if (!cuda_devices.empty())
	{
		spdlog::warn("CUDA unavailable!");
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
	int num_envs = environment_manager_->num_envs();
	for (int i = 0, num_make = env_count - num_envs; i < num_make; i++)
	{
		threadpool.queue_task([this]() { environment_manager_->add_environment(); });
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

	std::vector<EnvStepData> envs_data;
	envs_data.resize(env_count);

	// Wait for environments to complete initialising
	threadpool.wait_queue_empty();

	load_model(options.force_model_reload);

	model_->eval();

	for (int env = 0; env < env_count; env++)
	{
		threadpool.queue_task([&, env]() { run_episode(model_.get(), initial_state[env], env, options); });
	}

	// Wait for all environments to finish running
	threadpool.wait_queue_empty();
}

void Agent::run_episode(Model* model, const State& initial_state, int env, RunOptions options)
{
	assert(model != nullptr);
	auto environment = environment_manager_->get_environment(env);
	auto env_config = environment->get_configuration();
	if (options.max_steps <= 0)
	{
		// Run the environment until it terminates (setting to max int should be sufficient)
		options.max_steps = std::numeric_limits<int>::max();
	}

	// Get initial observation and state, running a env_step callback to update externally
	StepData step_data;
	step_data.eval_mode = true;
	step_data.env = env;
	step_data.step = 0;
	step_data.env_data = environment->reset(initial_state);
	step_data.predict_result.action = torch::zeros(env_config.action_space.shape);
	step_data.reward = step_data.env_data.reward.clone();

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
		auto observation = step_data.env_data.observation;
		for (auto& obs : observation) { obs = obs.unsqueeze(0).to(devices_.front()); }

		step_data.step = step;
		step_data.predict_result = model->predict(observation, options.deterministic);
		step_data.env_data = environment->step(step_data.predict_result.action);
		step_data.reward = step_data.env_data.reward.clone();
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

		if (step_data.env_data.state.episode_end)
		{
			environment->reset(initial_state);
			auto agent_reset_config = agent_callback_->env_reset(step_data);
			if (agent_reset_config.stop)
			{
				break;
			}
		}
	}
}

PredictOutput Agent::predict_action(const EnvStepData& env_data, bool deterministic)
{
	if (model_ == nullptr)
	{
		spdlog::error("No model loaded! Call 'load_model()' at least once before 'predict_action()' is called.");
		throw std::runtime_error("No model loaded!");
	}

	torch::NoGradGuard no_grad;

	Observations observations;
	for (auto& obs : env_data.observation) { observations.push_back(obs.unsqueeze(0).to(devices_.front())); }

	return model_->predict(env_data.observation, deterministic);
}

PredictOutput
Agent::predict_action([[maybe_unused]] const std::vector<StepData>& step_history, [[maybe_unused]] bool deterministic)
{
	spdlog::error("The agent type does not support `predict_action()` unless it has a history of previous step data.");
	throw std::runtime_error("Unsupported agent functionality");
}

void Agent::reset()
{
	training_ = false;
	model_.reset();
	environment_manager_->reset();
}

void Agent::train()
{
	spdlog::error("Training not available in default agent mode!");
}

void Agent::load_model(bool force_reload)
{
	if (force_reload || model_ == nullptr)
	{
		// Observation shape and action space are the same for all envs
		const auto env_config = environment_manager_->get_configuration();
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
			case AgentPolicyModelType::kSoftActorCritic:
			{
				model_ = std::make_shared<SoftActorCriticModel>(base_config_.model, env_config, reward_shape);
				break;
			}
			case AgentPolicyModelType::kQNet:
			{
				model_ = std::make_shared<QNetModel>(base_config_.model, env_config);
				break;
			}
			case AgentPolicyModelType::kMuZero:
			{
				model_ = std::make_shared<MuZeroModel>(base_config_.model, env_config, reward_shape);
				break;
			}
			default:
			{
				spdlog::error("Invalid model type selected!");
				return;
			}
		}

		model_->load(data_path_);
		model_->to(devices_.front());
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
	if (std::holds_alternative<Config::MCTSAgent>(config))
	{
		return std::make_unique<MCTSAgent>(config, environment_manager, callback, data_path);
	}

	return std::make_unique<Agent>(config, environment_manager, callback, data_path);
}
