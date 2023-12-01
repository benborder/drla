#include "hybrid_agent.h"

#include "agent_types.h"
#include "agent_utils.h"
#include "algorithm.h"
#include "dreamer.h"
#include "dreamer_model.h"
#include "hybrid_episode.h"
#include "hybrid_replay_buffer.h"
#include "model.h"
#include "random_model.h"
#include "sender_reciever.h"
#include "threadpool.h"
#include "utils.h"

#include <spdlog/spdlog.h>
#include <torch/torch.h>

#include <chrono>
#include <memory>

using namespace torch;
using namespace drla;

HybridAgent::HybridAgent(
	const Config::Agent& config,
	EnvironmentManager* environment_manager,
	AgentCallbackInterface* callback,
	std::filesystem::path data_path)
		: Agent(config, environment_manager, callback, std::move(data_path)), config_(std::get<Config::HybridAgent>(config))
{
}

HybridAgent::~HybridAgent()
{
}

void HybridAgent::train()
{
	if (config_.env_count <= 0)
	{
		spdlog::error("The number of environments must be greater than 0!");
		return;
	}

	// Create enough threads for self play envs, eval env, reanalyse and train update
	size_t total_threads = config_.env_count + static_cast<int>(config_.eval_period > 0) + 2;
	ThreadPool threadpool(total_threads, /*clamp_threads=*/false, 10);
	make_environments(threadpool, config_.env_count);

	// Block and wait for envs to be created
	threadpool.wait_queue_empty();

	// Configuration is the same for all envs
	const auto env_config = environment_manager_->get_configuration();
	int reward_shape = config_.rewards.combine_rewards ? 1 : static_cast<int>(env_config.reward_types.size());

	const Config::HybridAlgorithm train_config = std::visit(
		[&](auto& config) -> Config::HybridAlgorithm {
			using T = std::decay_t<decltype(config)>;
			if constexpr (std::is_convertible_v<T, Config::HybridAlgorithm>)
			{
				return static_cast<Config::HybridAlgorithm>(config);
			}
			throw std::runtime_error("Incompatible train algorithm specified. Must be derrived from 'HybridAlgorithm'.");
		},
		config_.train_algorithm);
	int timestep = train_config.start_timestep;

	Sender<std::shared_ptr<HybridModelInterface>> model_syncer;

	EpisodicPERBufferOptions buffer_options = {
		train_config.buffer_size, reward_shape, train_config.unroll_steps, train_config.per_alpha, env_config.action_space};

	HybridReplayBuffer buffer(config_.gamma, config_.env_count, buffer_options);

	// Self play
	for (int env = 0; env < config_.env_count; ++env)
	{
		threadpool.queue_task(
			[&, env]() {
				auto model_sync_reciever = model_syncer.create_reciever();

				torch::Device device{torch::kCPU};
				{
					const auto& gpus = train_config.self_play_gpus;
					const bool use_all_gpus = !gpus.empty() && gpus.back() == -1;
					const bool is_valid_gpu =
						use_all_gpus || std::find(gpus.begin(), gpus.end(), env % config_.env_count) != gpus.end();
					auto dev = std::find_if(devices_.begin(), devices_.end(), [&](const torch::Device& dev) {
						return is_valid_gpu && dev.index() == env;
					});
					if (dev != devices_.end())
					{
						device = *dev;
					}
				}

				// Wait for the initial model to be sent
				auto model = model_sync_reciever->request();
				if (device != torch::kCPU)
				{
					model = std::dynamic_pointer_cast<HybridModelInterface>(model->clone(device));
				}
				model->eval();

				int total_timesteps = train_config.total_timesteps;
				int max_steps = train_config.max_steps;
				auto environment = environment_manager_->get_environment(env);
				while (timestep < total_timesteps && training_)
				{
					const auto initial_state = environment_manager_->get_initial_state();
					if (max_steps <= 0)
					{
						// Run the environment until it terminates (setting to max int should be sufficient)
						max_steps = std::numeric_limits<int>::max();
					}

					// Get initial observation and state, running a env_step callback to update externally
					StepData step_data;
					step_data.eval_mode = false;
					step_data.env = env;
					step_data.step = 0;
					step_data.env_data = environment->reset(initial_state);
					step_data.predict_result = model->initial();
					step_data.reward = clamp_reward(step_data.env_data.reward, config_.rewards);

					bool buffer_ready = buffer.get_num_samples() >= train_config.start_buffer_size;
					auto agent_reset_config = agent_callback_->env_reset(step_data);
					auto enable_visualisations = agent_reset_config.enable_visualisations;
					if (enable_visualisations)
					{
						step_data.visualisation = environment->get_visualisations();
					}
					agent_callback_->env_step(step_data);
					buffer.add(step_data, !buffer_ready);
					m_env_stats_.lock();
					++total_samples_;
					m_env_stats_.unlock();

					torch::NoGradGuard no_grad;

					for (int step = 1; step <= max_steps && training_; step++)
					{
						buffer_ready = buffer.get_num_samples() >= train_config.start_buffer_size;
						if (timestep > 0 || (timestep == 0 && buffer_ready))
						{
							// update to the latest model from training (wait for it to be sent)
							auto new_model = model_sync_reciever->request();
							if (device != torch::kCPU)
							{
								model.reset(); // delete the old model before allocating the new one
								// TODO: Ideally copy the weight from the CPU to the existing GPU model to avoid allocating a new model
								// and destroying the old.
								model = std::dynamic_pointer_cast<HybridModelInterface>(new_model->clone(device));
							}
							else
							{
								model = new_model;
							}
							model->eval();
							if (!training_)
							{
								return;
							}
						}

						auto start = std::chrono::high_resolution_clock::now();

						ModelInput input;
						input.deterministic = false;
						input.prev_output = step_data.predict_result;
						// Add batch dim to observations, make sure its on the correct device and scale if using 8bit/int/char
						input.observations = convert_observations(step_data.env_data.observation, device);

						step_data.step = step;

						step_data.predict_result = model->predict(input);
						step_data.env_data = environment->step(step_data.predict_result.action);
						step_data.reward = clamp_reward(step_data.env_data.reward, config_.rewards);
						if (enable_visualisations)
						{
							step_data.visualisation = environment->get_visualisations();
						}

						std::chrono::duration<double> duration =
							std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start);
						if (max_steps > 0 && step >= max_steps)
						{
							step_data.env_data.state.episode_end = true;
						}

						bool stop = agent_callback_->env_step(step_data) || step_data.env_data.state.episode_end;

						buffer.add(step_data, !buffer_ready);

						std::lock_guard lock(m_env_stats_);
						env_duration_stats_.update(duration.count());
						++total_samples_;

						if (stop)
						{
							if (step_data.env_data.state.episode_end)
							{
								env_samples_stats_.update(step);
							}
							break;
						}
					}
				}
			},
			env);
	}

	std::future<Environment*> eval_env_future;

	if (config_.eval_period > 0)
	{
		const int eval_env = environment_manager_->num_envs(); // the eval env is the last one
		eval_env_future = threadpool.queue_task([this]() { return environment_manager_->add_environment(); }, eval_env);

		// Evaluation. Run the agent every n train steps to evaluate its true performance
		threadpool.queue_task(
			[&, eval_env]() {
				std::shared_ptr<HybridModelInterface> model;

				auto model_sync_reciever = model_syncer.create_reciever();

				// Wait for the initial model to be sent
				model = model_sync_reciever->request();
				model->eval();

				// This also serves to block the thread until the env has loaded (in case its really slow to load).
				eval_env_future.get();

				RunOptions options;
				options.max_steps = train_config.eval_max_steps;
				options.deterministic = train_config.eval_determinisic;
				options.enable_visualisations = true;

				int next_eval_run = config_.eval_period;
				while (timestep < train_config.total_timesteps && training_)
				{
					// update to the latest model from training (if available)
					if (auto new_model = model_sync_reciever->wait([&] { return !training_ || (timestep > next_eval_run); }))
					{
						model = *new_model;
						model->eval();
						next_eval_run += config_.eval_period;
						Agent::run_episode(model.get(), environment_manager_->get_initial_state(), eval_env, options);
					}
				}
			},
			eval_env);
	}

	// Setup model for training
	switch (config_.model_type)
	{
		case AgentPolicyModelType::kDreamer:
		{
			model_ = std::make_shared<DreamerModel>(config_.model, env_config, reward_shape, true);
			break;
		}
		default:
		{
			spdlog::error("Invalid model selected for training!");
			return;
		}
	}
	{
		c10::Device device{torch::kCPU};
		const auto& gpus = train_config.train_gpus;
		if (gpus.back() == -1)
		{
			device = devices_.front();
		}
		else
		{
			auto dev = std::find_if(devices_.begin(), devices_.end(), [&](const c10::Device& dev) {
				return std::find(gpus.begin(), gpus.end(), dev.index()) != gpus.end();
			});
			if (dev != devices_.end())
			{
				device = *dev;
			}
		}
		model_->to(device);
	}

	// Setup algorithm for training
	std::unique_ptr<Algorithm> algorithm;

	switch (config_.train_algorithm_type)
	{
		case TrainAlgorithmType::kDreamer:
		{
			algorithm = std::make_unique<Dreamer>(
				config_.train_algorithm,
				env_config.action_space,
				std::dynamic_pointer_cast<HybridModelInterface>(model_),
				buffer);
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

	// If the start step is greter than zero, attempt to load an existing optimiser at the specified data_path_
	if (timestep > 0)
	{
		algorithm->load(data_path_);
	}

	// Reanalyse thread
	threadpool.queue_task([&]() {
		using namespace std::chrono_literals;
		auto model_sync_reciever = model_syncer.create_reciever();

		// Wait until conditions are met to start reanalyse
		while (timestep < train_config.min_reanalyse_train_steps &&
					 buffer.get_num_episodes() < train_config.min_reanalyse_buffer_size)
		{
			std::this_thread::sleep_for(1s);
		}

		auto model = model_sync_reciever->request();
		model->eval();

		while (training_)
		{
			buffer.reanalyse(model);
			// update to the latest model from training (if available)
			if (auto new_model = model_sync_reciever->check())
			{
				model = *new_model;
				model->eval();
			}
		}
	});

	{
		InitData data;
		data.env_config = env_config;
		data.reward_shape = reward_shape;
		for (int env = 0; env < config_.env_count + 1; ++env)
		{
			data.env_output.push_back({{}, torch::zeros({reward_shape}), {{}, train_config.start_timestep}});
		}
		agent_callback_->train_init(data);
	}

	// Synchronise the model
	model_syncer.send(std::dynamic_pointer_cast<HybridModelInterface>(model_->clone(torch::kCPU)));

	// Wait for the min number of episodes to be available in the buffer
	{
		using namespace std::chrono_literals;
		spdlog::info("Waiting for buffer to reach min required size");
		while (buffer.get_num_samples() < train_config.start_buffer_size)
		{
			spdlog::fmt_lib::print("\rCompleted {}/{} samples", buffer.get_num_samples(), train_config.start_buffer_size);
			std::this_thread::sleep_for(10ms);
		}
		spdlog::fmt_lib::print(
			"\rCompleted {}/{} samples\n", train_config.start_buffer_size, train_config.start_buffer_size);
	}

	buffer.flush_cache();

	// Run train loop
	for (; timestep < train_config.total_timesteps; ++timestep)
	{
		TrainUpdateData train_update_data;
		train_update_data.timestep = timestep;

		// Measure the model update step time.
		auto start = std::chrono::steady_clock::now();

		train_update_data.metrics = algorithm->update(timestep);
		train_update_data.metrics.add(
			{"reanalyse_count", TrainResultType::kBufferStats, static_cast<double>(buffer.get_reanalysed_count())});
		train_update_data.update_duration = std::chrono::steady_clock::now() - start;
		train_update_data.env_duration =
			std::chrono::duration<double>(env_duration_stats_.get_mean() * env_samples_stats_.get_mean());
		train_update_data.fps_env = env_samples_stats_.get_mean() / train_update_data.env_duration.count();
		train_update_data.fps = train_update_data.fps_env * config_.env_count;
		train_update_data.global_steps = total_samples_;

		threadpool.queue_task([this, train_update_data]() { agent_callback_->train_update(train_update_data); });

		if ((timestep % static_cast<int>(std::ceil(train_config.train_ratio))) == 0)
		{
			model_syncer.send(std::dynamic_pointer_cast<HybridModelInterface>(model_->clone(torch::kCPU)));
		}

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

		using namespace std::chrono_literals;
		while (timestep > (buffer.get_num_samples() * train_config.train_ratio)) { std::this_thread::sleep_for(10ms); }

		if (!training_)
		{
			break;
		}
	}

	training_ = false;
	model_syncer.send(std::dynamic_pointer_cast<HybridModelInterface>(model_->clone(torch::kCPU)));

	// Block unitl all other threads are completed
	threadpool.wait_queue_empty();
}
