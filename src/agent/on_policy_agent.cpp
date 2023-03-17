#include "on_policy_agent.h"

#include "a2c.h"
#include "actor_critic_model.h"
#include "agent_types.h"
#include "model.h"
#include "ppo.h"
#include "random_model.h"
#include "rollout_buffer.h"
#include "threadpool.h"

#include <spdlog/spdlog.h>
#include <torch/torch.h>

using namespace torch;
using namespace drla;

OnPolicyAgent::OnPolicyAgent(
	const Config::Agent& config,
	EnvironmentManager* environment_manager,
	AgentCallbackInterface* callback,
	std::filesystem::path data_path)
		: Agent(config, environment_manager, callback, std::move(data_path))
		, config_(std::get<Config::OnPolicyAgent>(config))
{
}

OnPolicyAgent::OnPolicyAgent(
	const Config::OnPolicyAgent& config,
	EnvironmentManager* environment_manager,
	AgentCallbackInterface* callback,
	std::filesystem::path data_path)
		: Agent(config, environment_manager, callback, std::move(data_path)), config_(config)
{
}

OnPolicyAgent::~OnPolicyAgent()
{
}

void OnPolicyAgent::train()
{
	if (config_.env_count <= 0)
	{
		spdlog::error("The number of environments must be greater than 0!");
		return;
	}
	ThreadPool threadpool(config_.env_count);
	make_environments(threadpool, config_.env_count + (config_.eval_period > 0 ? 1 : 0));

	std::vector<EnvStepData> envs_data;
	envs_data.resize(config_.env_count);

	// Block and wait for envs to be created
	threadpool.wait_queue_empty();

	// The environments are reset and initialised before creating the algorithm as the observation shape must be known
	for (int env = 0; env < config_.env_count; env++)
	{
		threadpool.queue_task([&, env]() {
			envs_data[env] = environment_manager_->get_environment(env)->reset(environment_manager_->get_initial_state());
		});
	}
	if (config_.eval_period > 0)
	{
		envs_data.push_back({}); // Add an extra one for the eval env
	}

	int horizon_steps = 0;
	int max_steps = 0;
	int timestep = 0;
	int eval_max_steps = 0;
	torch::Tensor gae_lambda = torch::empty({1}, device_);
	// The discount factor for each reward type
	std::vector<float> config_gamma;
	std::visit(
		[&](auto& config) {
			using T = std::decay_t<decltype(config)>;
			if constexpr (std::is_base_of_v<Config::OnPolicyAlgorithm, T>)
			{
				horizon_steps = config.horizon_steps;
				max_steps = config.total_timesteps;
				timestep = config.start_timestep;
				gae_lambda[0] = config.gae_lambda;
				config_gamma = config.gamma;
				eval_max_steps = config.eval_max_steps;
			}
		},
		config_.train_algorithm);

	// Block and wait for envs to initialise
	threadpool.wait_queue_empty();

	// Configuration is the same for all envs
	auto env_config = environment_manager_->get_configuration();
	int reward_shape = config_.rewards.combine_rewards ? 1 : static_cast<int>(env_config.reward_types.size());

	agent_callback_->train_init({env_config, reward_shape, envs_data});

	std::shared_ptr<Model> model;
	switch (config_.model_type)
	{
		case AgentPolicyModelType::kActorCritic:
		{
			model = std::make_shared<ActorCriticModel>(config_.model, env_config, reward_shape, true);
			break;
		}
		default:
		{
			spdlog::error("Invalid model selected for training!");
			return;
		}
	}
	model->to(device_);

	if (config_gamma.size() < static_cast<size_t>(reward_shape))
	{
		config_gamma.resize(reward_shape, config_gamma.front());
	}
	torch::Tensor gamma = torch::from_blob(config_gamma.data(), {reward_shape}).to(device_);

	RolloutBuffer buffer(horizon_steps, config_.env_count, env_config, reward_shape, gamma, gae_lambda, device_);

	std::unique_ptr<Algorithm> algorithm;

	switch (config_.train_algorithm_type)
	{
		case TrainAlgorithmType::kA2C:
		{
			algorithm = std::make_unique<A2C>(config_.train_algorithm, buffer, model);
			break;
		}
		case TrainAlgorithmType::kPPO:
		{
			algorithm = std::make_unique<PPO>(config_.train_algorithm, buffer, model);
			break;
		}
		default:
		{
			spdlog::error("Unsupported algorithm type for on policy train mode.");
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

	std::vector<bool> raw_capture;
	raw_capture.resize(config_.env_count, false);

	// Get first observation and state for all envs
	for (int env = 0; env < config_.env_count; env++)
	{
		StepData step_data;
		step_data.env = env;
		step_data.step = 0;
		step_data.env_data = std::move(envs_data[env]);
		step_data.predict_result.action = torch::zeros(env_config.action_space.shape);
		step_data.reward = step_data.env_data.reward.clone();
		auto agent_reset_config = agent_callback_->env_reset(step_data);
		raw_capture[env] = agent_reset_config.raw_capture;
		buffer.initialise(step_data);
	}

	for (; timestep < max_steps; timestep++)
	{
		auto start = std::chrono::steady_clock::now();
		if (config_.asynchronous_env)
		{
			// Dispatch each environment on a seperate thread and run for horizon_steps environment steps
			for (int env = 0; env < config_.env_count; env++)
			{
				threadpool.queue_task([&, env]() {
					auto environment = environment_manager_->get_environment(env);

					StepData step_data;
					step_data.env = env;
					step_data.env_data.observation = buffer.get_observations(0, env);
					for (int step = 0; step < horizon_steps; step++)
					{
						step_data.step = step;
						{
							torch::NoGradGuard no_grad;
							for (auto& obs : step_data.env_data.observation) { obs = obs.unsqueeze(0).to(device_); }
							step_data.predict_result = model->predict(step_data.env_data.observation, false);
						}
						step_data.env_data = environment->step(step_data.predict_result.action);

						step_data.reward = step_data.env_data.reward.clone();
						if (config_.rewards.reward_clamp_min != 0)
						{
							step_data.reward.clamp_max_(-config_.rewards.reward_clamp_min);
						}
						if (config_.rewards.reward_clamp_max != 0)
						{
							step_data.reward.clamp_min_(-config_.rewards.reward_clamp_max);
						}
						if (raw_capture[env])
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
							// If the episode has ended reset the environment and call env_reset
							StepData reset_data;
							reset_data.env = env;
							reset_data.env_data = environment->reset(environment_manager_->get_initial_state());
							reset_data.reward = reset_data.env_data.reward.clone();
							if (config_.rewards.reward_clamp_min != 0)
							{
								reset_data.reward.clamp_max_(-config_.rewards.reward_clamp_min);
							}
							if (config_.rewards.reward_clamp_max != 0)
							{
								reset_data.reward.clamp_min_(-config_.rewards.reward_clamp_max);
							}
							if (raw_capture[env])
							{
								reset_data.raw_observation = environment->get_raw_observations();
							}
							auto agent_reset_config = agent_callback_->env_reset(reset_data);
							step_data.env_data.observation = reset_data.env_data.observation;
							raw_capture[env] = agent_reset_config.raw_capture;
							if (agent_reset_config.stop)
							{
								break;
							}
						}

						buffer.add(step_data);
					}
				});
			}

			// Wait for all environments to finish their batch
			threadpool.wait_queue_empty();
		}
		else
		{
			TimeStepData timestep_data;
			timestep_data.observations = buffer.get_observations(0);
			timestep_data.rewards = torch::empty({config_.env_count, reward_shape});
			timestep_data.states.resize(config_.env_count);
			bool stop = false;
			for (int step = 0; step < horizon_steps && !stop; step++)
			{
				timestep_data.step = step;
				{
					torch::NoGradGuard no_grad;
					auto observations = buffer.get_observations(step);
					timestep_data.predict_results = model->predict(observations, false);
				}

				// Dispatch each environment on a seperate thread and step it
				for (int env = 0; env < config_.env_count; env++)
				{
					threadpool.queue_task([&, env]() {
						auto environment = environment_manager_->get_environment(env);

						StepData step_data;
						step_data.env = env;
						step_data.step = step;
						step_data.predict_result.action = timestep_data.predict_results.action[env];
						step_data.predict_result.action_log_probs = timestep_data.predict_results.action_log_probs[env];
						step_data.predict_result.values = timestep_data.predict_results.values[env];

						step_data.env_data = environment->step(step_data.predict_result.action);

						step_data.reward = step_data.env_data.reward.clone();
						if (config_.rewards.reward_clamp_min != 0)
						{
							step_data.reward.clamp_max_(-config_.rewards.reward_clamp_min);
						}
						if (config_.rewards.reward_clamp_max != 0)
						{
							step_data.reward.clamp_min_(-config_.rewards.reward_clamp_max);
						}
						if (raw_capture[env])
						{
							step_data.raw_observation = environment->get_raw_observations();
						}

						stop |= agent_callback_->env_step(step_data);

						if (step_data.env_data.state.episode_end)
						{
							// If the episode has ended reset the environment and call env_reset
							StepData reset_data;
							reset_data.env = env;
							reset_data.env_data = environment->reset(environment_manager_->get_initial_state());
							reset_data.reward = reset_data.env_data.reward.clone();
							if (config_.rewards.reward_clamp_min != 0)
							{
								reset_data.reward.clamp_max_(-config_.rewards.reward_clamp_min);
							}
							if (config_.rewards.reward_clamp_max != 0)
							{
								reset_data.reward.clamp_min_(-config_.rewards.reward_clamp_max);
							}
							if (raw_capture[env])
							{
								reset_data.raw_observation = environment->get_raw_observations();
							}
							auto agent_reset_config = agent_callback_->env_reset(reset_data);
							step_data.env_data.observation = reset_data.env_data.observation;
							raw_capture[env] = agent_reset_config.raw_capture;
							stop |= agent_reset_config.stop;
						}

						for (size_t i = 0; i < step_data.env_data.observation.size(); i++)
						{
							timestep_data.observations[i][env] = step_data.env_data.observation[i];
						}
						timestep_data.rewards[env] = std::move(step_data.reward);
						timestep_data.states[env] = std::move(step_data.env_data.state);
					});
				}
				threadpool.wait_queue_empty();
				buffer.add(timestep_data);
			}
		}

		TrainUpdateData train_update_data;
		train_update_data.timestep = timestep;
		train_update_data.env_duration = std::chrono::steady_clock::now() - start;
		double period = 1.0 / train_update_data.env_duration.count();
		double env_thread_ratio =
			config_.env_count <= static_cast<int>(std::thread::hardware_concurrency())
				? 1.0
				: static_cast<double>(config_.env_count) / static_cast<double>(std::thread::hardware_concurrency());
		train_update_data.fps_env = horizon_steps * period * env_thread_ratio;
		train_update_data.fps = horizon_steps * period * config_.env_count;
		train_update_data.global_steps = timestep * config_.env_count * horizon_steps;

		// Measure the model update step time.
		start = std::chrono::steady_clock::now();

		torch::Tensor last_values;
		{
			torch::NoGradGuard no_grad;
			Observations obs = buffer.get_observations(-1); // get the last step observation
			last_values = model->predict(obs).values.detach();
		}
		buffer.compute_returns_and_advantage(last_values);

		train_update_data.update_data = algorithm->update(timestep);

		buffer.prepare_next_batch();

		train_update_data.update_duration = std::chrono::steady_clock::now() - start;

		// Run evaluation every save period
		if (config_.eval_period > 0 && timestep % config_.eval_period == 0)
		{
			RunOptions options;
			options.deterministic = true;
			options.max_steps = eval_max_steps;
			options.capture_observations = true;
			run_episode(
				model.get(), environment_manager_->get_initial_state(), environment_manager_->num_envs() - 1, options);
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
