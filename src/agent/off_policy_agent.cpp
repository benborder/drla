#include "off_policy_agent.h"

#include "agent_types.h"
#include "agent_utils.h"
#include "algorithm.h"
#include "dqn.h"
#include "get_time.h"
#include "model.h"
#include "qnet_model.h"
#include "random_model.h"
#include "replay_buffer.h"
#include "sac.h"
#include "soft_actor_critic_model.h"
#include "threadpool.h"

#include <spdlog/spdlog.h>

using namespace torch;
using namespace drla;

OffPolicyAgent::OffPolicyAgent(
	const Config::Agent& config,
	EnvironmentManager* environment_manager,
	AgentCallbackInterface* callback,
	std::filesystem::path data_path)
		: Agent(config, environment_manager, callback, std::move(data_path))
		, config_(std::get<Config::OffPolicyAgent>(config))
{
}

OffPolicyAgent::OffPolicyAgent(
	const Config::OffPolicyAgent& config,
	EnvironmentManager* environment_manager,
	AgentCallbackInterface* callback,
	std::filesystem::path data_path)
		: Agent(config, environment_manager, callback, std::move(data_path)), config_(config)
{
}

OffPolicyAgent::~OffPolicyAgent()
{
}

void OffPolicyAgent::train()
{
	if (config_.env_count <= 0)
	{
		spdlog::error("The number of environments must be greater than 0!");
		return;
	}
	ThreadPool threadpool(config_.env_count, config_.clamp_concurrent_envs);
	make_environments(threadpool, config_.env_count);
	if (config_.eval_period > 0)
	{
		// Make an evaluation environment
		environment_manager_->add_environment();
	}

	std::vector<EnvStepData> envs_data;
	envs_data.resize(config_.env_count);

	// Block and wait for envs to be created
	threadpool.wait_queue_empty();

	// The environments are reset and initialised before creating the algorithm as the observation shape must be known
	for (int env = 0; env < config_.env_count; env++)
	{
		threadpool.queue_task(
			[&, env]() {
				envs_data[env] = environment_manager_->get_environment(env)->reset(environment_manager_->get_initial_state());
			},
			env);
	}

	const Config::OffPolicyAlgorithm train_config = std::visit(
		[&](auto& config) -> Config::OffPolicyAlgorithm {
			using T = std::decay_t<decltype(config)>;
			if constexpr (std::is_base_of_v<Config::OffPolicyAlgorithm, T>)
			{
				return static_cast<Config::OffPolicyAlgorithm>(config);
			}
			throw std::runtime_error("Incompatible train algorithm specified. Must be derrived from 'OffPolicyAlgorithm'.");
		},
		config_.train_algorithm);

	// The discount factor for each reward type
	std::vector<float> config_gamma = train_config.gamma;
	int timestep = train_config.start_timestep;

	// Block and wait for envs to initialise
	threadpool.wait_queue_empty();

	// Configuration is the same for all envs
	auto env_config = environment_manager_->get_configuration();
	int reward_shape = config_.rewards.combine_rewards ? 1 : static_cast<int>(env_config.reward_types.size());

	agent_callback_->train_init({env_config, reward_shape, envs_data});

	std::shared_ptr<Model> model;
	switch (config_.model_type)
	{
		case AgentPolicyModelType::kQNet:
		{
			model = std::make_shared<QNetModel>(config_.model, env_config, reward_shape);
			break;
		}
		case AgentPolicyModelType::kSoftActorCritic:
		{
			model = std::make_shared<SoftActorCriticModel>(config_.model, env_config, reward_shape);
			break;
		}
		default:
		{
			spdlog::error("Invalid model selected for training!");
			return;
		}
	}
	model->to(devices_.front());

	std::string buffer_save_path;
	if (!train_config.buffer_save_path.empty())
	{
		auto path = std::filesystem::path(train_config.buffer_save_path);
		if (path.is_relative())
		{
			path = data_path_ / path;
		}
		buffer_save_path = path;
	}

	ReplayBuffer buffer(
		train_config.buffer_size,
		config_.env_count,
		env_config,
		reward_shape,
		model->get_state_shape(),
		config_gamma,
		train_config.per_alpha,
		devices_.front(),
		buffer_save_path);

	if (!train_config.buffer_load_path.empty())
	{
		buffer.set_state_shapes(model_->get_state_shape());
		auto path = std::filesystem::path(train_config.buffer_load_path);
		if (path.is_relative())
		{
			path = data_path_ / path;
		}
		if (std::filesystem::exists(path))
		{
			spdlog::info("Loading episodes into replay buffer");
			buffer.load(path);
			spdlog::info(
				"Loaded {} episodes totalling {} samples into replay buffer.",
				buffer.get_num_episodes(),
				buffer.get_num_samples());
		}
		else
		{
			spdlog::warn("Unable to load episodes into buffer: No episode data exists at '{}'", path.string());
		}
	}

	std::unique_ptr<Algorithm> algorithm;

	switch (config_.train_algorithm_type)
	{
		case TrainAlgorithmType::kDQN:
		{
			algorithm = std::make_unique<DQN>(config_.train_algorithm, buffer, model);
			break;
		}
		case TrainAlgorithmType::kSAC:
		{
			algorithm = std::make_unique<SAC>(config_.train_algorithm, env_config.action_space, buffer, model);
			break;
		}
		default:
		{
			spdlog::error("Unsupported algorithm type for train mode.");
			return;
		}
	}
	spdlog::info("Agent training algorithm: {}", algorithm->name());

	training_ = true;

	// If the start step is greter than zero, attempt to load and existing optimiser at the specified data_path_
	if (timestep > 0)
	{
		algorithm->load(data_path_);
	}

	std::vector<bool> enable_visualisations;
	enable_visualisations.resize(config_.env_count, false);

	// Get first observation and state for all envs
	for (int env = 0; env < config_.env_count; env++)
	{
		threadpool.queue_task(
			[&, env]() {
				StepData reset_data;
				reset_data.name = get_time();
				reset_data.env = env;
				reset_data.step = 0;
				reset_data.env_data = std::move(envs_data[env]);
				reset_data.predict_result = model->initial();
				reset_data.reward = clamp_reward(reset_data.env_data.reward, config_.rewards);
				auto agent_reset_config = agent_callback_->env_reset(reset_data);
				enable_visualisations[env] = agent_reset_config.enable_visualisations;
				if (agent_reset_config.enable_visualisations)
				{
					reset_data.visualisation = environment_manager_->get_environment(env)->get_visualisations();
				}
				agent_callback_->env_step(reset_data);
				buffer.add(reset_data);
			},
			env);
	}

	for (; timestep < train_config.total_timesteps; timestep++)
	{
		auto start = std::chrono::steady_clock::now();
		if (config_.asynchronous_env)
		{
			// Dispatch each environment on a seperate thread and run for horizon_steps environment steps
			for (int env = 0; env < config_.env_count; env++)
			{
				threadpool.queue_task(
					[&, env]() {
						auto environment = environment_manager_->get_environment(env);

						thread_local StepData step_data;
						step_data.env = env;
						step_data.env_data.observation = buffer.get_observations_head(env);
						step_data.predict_result.action = buffer.get_actions_head(env);
						step_data.predict_result.state = buffer.get_state_head(env);
						for (int step = 0; step < train_config.horizon_steps; step++)
						{
							++step_data.step;
							{
								torch::NoGradGuard no_grad;
								auto observations = convert_observations(step_data.env_data.observation, devices_.front());
								step_data.predict_result = model->predict({std::move(observations), step_data.predict_result, false});
							}
							step_data.env_data = environment->step(step_data.predict_result.action);

							step_data.reward = clamp_reward(step_data.env_data.reward, config_.rewards);
							if (enable_visualisations[env])
							{
								step_data.visualisation = environment->get_visualisations();
							}
							if (train_config.max_steps > 0 && step_data.step >= train_config.max_steps)
							{
								step_data.env_data.state.episode_end = true;
							}

							bool stop = agent_callback_->env_step(step_data);
							bool episode_end = step_data.env_data.state.episode_end;
							buffer.add(step_data);
							if (stop)
							{
								break;
							}

							if (episode_end)
							{
								// If the episode has ended reset the environment and call env_reset
								StepData reset_data;
								reset_data.name = get_time();
								reset_data.env = env;
								reset_data.env_data = environment->reset(environment_manager_->get_initial_state());
								reset_data.predict_result = model->initial();
								reset_data.reward = clamp_reward(reset_data.env_data.reward, config_.rewards);
								if (enable_visualisations[env])
								{
									reset_data.visualisation = environment->get_visualisations();
								}
								auto agent_reset_config = agent_callback_->env_reset(reset_data);
								enable_visualisations[env] = agent_reset_config.enable_visualisations;
								buffer.add(std::move(reset_data));
								step_data.step = 0;
								if (agent_reset_config.stop)
								{
									break;
								}
							}
						}
					},
					env);
			}

			// Wait for all environments to finish their batch
			threadpool.wait_queue_empty();
		}
		else
		{
			TimeStepData timestep_data;
			timestep_data.observations = buffer.get_observations_head();
			timestep_data.predict_results.action = buffer.get_actions_head();
			timestep_data.predict_results.state = buffer.get_state_head();
			timestep_data.rewards = torch::empty({config_.env_count, reward_shape});
			timestep_data.states.resize(config_.env_count);
			bool stop = false;
			for (int step = 0; step < train_config.horizon_steps && !stop; step++)
			{
				timestep_data.step = step;
				{
					torch::NoGradGuard no_grad;
					auto observations = convert_observations(timestep_data.observations, devices_.front(), false);
					timestep_data.predict_results =
						model->predict({std::move(observations), timestep_data.predict_results, false});
				}

				// Dispatch each environment on a seperate thread and step it
				for (int env = 0; env < config_.env_count; env++)
				{
					threadpool.queue_task(
						[&, env]() {
							auto environment = environment_manager_->get_environment(env);

							thread_local StepData step_data;
							step_data.env = env;
							++step_data.step;
							step_data.predict_result.action = timestep_data.predict_results.action[env];
							step_data.predict_result.values = timestep_data.predict_results.values[env];

							step_data.env_data = environment->step(step_data.predict_result.action);

							step_data.reward = clamp_reward(step_data.env_data.reward, config_.rewards);
							if (enable_visualisations[env])
							{
								step_data.visualisation = environment->get_visualisations();
							}
							if (train_config.max_steps > 0 && step_data.step >= train_config.max_steps)
							{
								step_data.env_data.state.episode_end = true;
							}

							stop |= agent_callback_->env_step(step_data);
							bool episode_end = step_data.env_data.state.episode_end;
							timestep_data.rewards[env] = step_data.reward;
							timestep_data.states[env] = step_data.env_data.state;
							for (size_t i = 0; i < step_data.env_data.observation.size(); i++)
							{
								timestep_data.observations[i][env] = step_data.env_data.observation[i];
							}
							buffer.add(std::move(step_data));

							if (episode_end)
							{
								// If the episode has ended reset the environment and call env_reset
								StepData reset_data;
								reset_data.name = get_time();
								reset_data.env = env;
								reset_data.env_data = environment->reset(environment_manager_->get_initial_state());
								reset_data.predict_result = model->initial();
								reset_data.reward = clamp_reward(reset_data.env_data.reward, config_.rewards);
								if (enable_visualisations[env])
								{
									reset_data.visualisation = environment->get_visualisations();
								}
								auto agent_reset_config = agent_callback_->env_reset(reset_data);
								timestep_data.states[env] = reset_data.env_data.state;
								for (size_t i = 0; i < reset_data.env_data.observation.size(); i++)
								{
									timestep_data.observations[i][env] = reset_data.env_data.observation[i];
								}
								buffer.add(std::move(reset_data));
								enable_visualisations[env] = agent_reset_config.enable_visualisations;
								stop |= agent_reset_config.stop;
								step_data.step = 0;
							}
						},
						env);
				}
				threadpool.wait_queue_empty();
			}
		}

		TrainUpdateData train_update_data;
		train_update_data.timestep = timestep;
		train_update_data.env_duration = std::chrono::steady_clock::now() - start;
		train_update_data.metrics.add(
			{"buffer_size", TrainResultType::kBufferStats, static_cast<double>(buffer.get_num_samples())});
		double period = 1.0 / train_update_data.env_duration.count();
		double env_thread_ratio =
			config_.env_count <= static_cast<int>(std::thread::hardware_concurrency())
				? 1.0
				: static_cast<double>(config_.env_count) / static_cast<double>(std::thread::hardware_concurrency());
		train_update_data.fps_env = train_config.horizon_steps * period * env_thread_ratio;
		train_update_data.fps = train_config.horizon_steps * period * config_.env_count;
		train_update_data.global_steps = timestep * config_.env_count * train_config.horizon_steps;

		if (timestep > train_config.learning_starts)
		{
			// Measure the model update step time.
			start = std::chrono::steady_clock::now();

			train_update_data.metrics = algorithm->update(timestep);

			train_update_data.update_duration = std::chrono::steady_clock::now() - start;
		}
		else
		{
			using namespace std::chrono_literals;
			train_update_data.update_duration = 0ms;
		}

		// Run evaluation every save period
		if (config_.eval_period > 0 && timestep % config_.eval_period == 0)
		{
			RunOptions options;
			options.deterministic = train_config.eval_determinisic;
			options.max_steps = train_config.eval_max_steps;
			options.enable_visualisations = true;
			int env_id = environment_manager_->num_envs() - 1;
			run_episode(model.get(), environment_manager_->get_initial_state(), env_id, options);
		}

		agent_callback_->train_update(train_update_data);

		// Periodically save
		if (timestep % config_.timestep_save_period == 0)
		{
			auto path = get_save_path();
			algorithm->save(path);
			agent_callback_->save(timestep, path);
		}

		if (config_.checkpoint_save_period > 0 && timestep > 0 && timestep % config_.checkpoint_save_period == 0)
		{
			auto path = get_save_path("checkpoint_" + std::to_string(train_update_data.timestep));
			algorithm->save(path);
			agent_callback_->save(train_update_data.timestep, path);
		}

		if (!training_)
		{
			break;
		}
	}

	model_ = model;
	training_ = false;
}
