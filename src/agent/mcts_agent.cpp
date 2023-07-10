#include "mcts_agent.h"

#include "agent_types.h"
#include "agent_utils.h"
#include "algorithm.h"
#include "episodic_per_buffer.h"
#include "mcts.h"
#include "mcts_episode.h"
#include "model.h"
#include "muzero.h"
#include "muzero_model.h"
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

MCTSAgent::MCTSAgent(
	const Config::Agent& config,
	EnvironmentManager* environment_manager,
	AgentCallbackInterface* callback,
	std::filesystem::path data_path)
		: Agent(config, environment_manager, callback, std::move(data_path))
		, config_(std::get<Config::MCTSAgent>(config))
		, gen_(std::random_device{}())
{
}

MCTSAgent::~MCTSAgent()
{
}

void MCTSAgent::train()
{
	if (config_.env_count <= 0)
	{
		spdlog::error("The number of environments must be greater than 0!");
		return;
	}

	// Create enough threads for self play envs, eval env and reanalyse
	ThreadPool threadpool(config_.env_count + 2, /*clamp_threads=*/false);
	make_environments(threadpool, config_.env_count);

	// Block and wait for envs to be created
	threadpool.wait_queue_empty();

	// Configuration is the same for all envs
	auto env_config = environment_manager_->get_configuration();
	int reward_shape = config_.rewards.combine_rewards ? 1 : static_cast<int>(env_config.reward_types.size());

	const Config::MCTSAlgorithm train_config = std::visit(
		[&](auto& config) -> Config::MCTSAlgorithm {
			using T = std::decay_t<decltype(config)>;
			if constexpr (std::is_base_of_v<Config::MCTSAlgorithm, T>)
			{
				return static_cast<Config::MCTSAlgorithm>(config);
			}
			throw std::runtime_error("Incompatible train algorithm specified. Must be derrived from 'MCTSAlgorithm'.");
		},
		config_.train_algorithm);
	int timestep = train_config.start_timestep;

	Sender<std::shared_ptr<MCTSModelInterface>> model_syncer;

	EpisodicPERBufferOptions buffer_options = {
		train_config.buffer_size,
		reward_shape,
		train_config.unroll_steps + 1, // add 1 to include the current step
		train_config.per_alpha,
		env_config.action_space};

	EpisodicPERBuffer buffer(config_.gamma, buffer_options);

	// Self play
	// TODO: add possiblity to also use a client/server via rpc
	for (int env = 0; env < config_.env_count; ++env)
	{
		threadpool.queue_task([&, env]() {
			std::shared_ptr<MCTSModelInterface> model;

			auto model_sync_reciever = model_syncer.create_reciever();

			// Wait for the initial model to be sent
			model = model_sync_reciever->wait();
			c10::Device device{torch::kCPU};
			{
				const auto& gpus = train_config.self_play_gpus;
				const bool use_all_gpus = gpus.back() == -1;
				const bool is_valid_gpu =
					use_all_gpus || std::find(gpus.begin(), gpus.end(), env % config_.env_count) != gpus.end();
				auto dev = std::find_if(
					devices_.begin(), devices_.end(), [&](const c10::Device& dev) { return is_valid_gpu && dev.index() == env; });
				if (dev != devices_.end())
				{
					device = *dev;
				}
			}

			if (device != torch::kCPU)
			{
				model = std::dynamic_pointer_cast<MCTSModelInterface>(model->clone(device));
			}
			model->eval();

			RunOptions options;
			options.deterministic = false;
			options.capture_observations = false;
			options.force_model_reload = false;
			options.temperature = 1;

			while (timestep < train_config.total_timesteps)
			{
				for (auto [step, temp] : train_config.temperature_step)
				{
					if (timestep < step)
					{
						options.temperature = temp;
						break;
					}
				}
				auto start = std::chrono::steady_clock::now();
				auto episode_data = run_episode(model.get(), environment_manager_->get_initial_state(), env, false, options);
				std::chrono::duration<double> duration =
					std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start);
				{
					std::lock_guard lock(m_env_stats_);
					total_samples_ += episode_data->length();
					env_samples_stats_.update(episode_data->length());
					env_duration_stats_.update(duration.count());
				}
				buffer.add_episode(std::move(episode_data));
				// update to the latest model from training (if available)
				if (auto new_model = model_sync_reciever->check())
				{
					if (device != torch::kCPU)
					{
						model = std::dynamic_pointer_cast<MCTSModelInterface>(new_model.value()->clone(device));
					}
					else
					{
						model = *new_model;
					}
					model->eval();
				}
			}
		});
	}

	std::future<Environment*> eval_env_future;

	if (config_.eval_period > 0)
	{
		eval_env_future = threadpool.queue_task([this]() { return environment_manager_->add_environment(); });

		// Evaluation. Run the agent every n train steps to evaluate its true performance
		threadpool.queue_task([&]() {
			std::shared_ptr<MCTSModelInterface> model;

			auto model_sync_reciever = model_syncer.create_reciever();

			// Wait for the initial model to be sent
			model = model_sync_reciever->wait();
			model->eval();

			// This also serves to block the thread until the env has loaded (in case its really slow to load).
			auto eval_env_ptr = eval_env_future.get();
			auto env_config = eval_env_ptr->get_configuration();
			int eval_env = environment_manager_->num_envs() - 1; // the eval env is the last one

			RunOptions options;
			options.max_steps = train_config.eval_max_steps;
			options.deterministic = train_config.eval_determinisic;
			options.capture_observations = true;
			options.temperature = config_.temperature;

			while (timestep < train_config.total_timesteps)
			{
				// update to the latest model from training (if available)
				if (auto new_model = model_sync_reciever->wait([&] {
							return !training_ || (timestep % config_.eval_period == 0);
						}))
				{
					model = *new_model;
					model->eval();
				}

				if (timestep % config_.eval_period == 0)
				{
					run_episode(model.get(), environment_manager_->get_initial_state(), eval_env, true, options);
				}

				if (!training_)
				{
					break;
				}
			}
		});
	}

	// Setup model for training
	switch (config_.model_type)
	{
		case AgentPolicyModelType::kMuZero:
		{
			model_ = std::make_shared<MuZeroModel>(config_.model, env_config, reward_shape);
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
	buffer.set_value_decoder([&](torch::Tensor& values) {
		return std::dynamic_pointer_cast<MCTSModelInterface>(model_)->support_to_scalar(values);
	});

	// Setup algorithm for training
	std::unique_ptr<Algorithm> algorithm;

	switch (config_.train_algorithm_type)
	{
		case TrainAlgorithmType::kMuZero:
		{
			algorithm = std::make_unique<MuZero>(
				config_.train_algorithm, std::dynamic_pointer_cast<MCTSModelInterface>(model_), buffer);
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

		auto model = model_sync_reciever->wait();
		model->eval();

		while (training_)
		{
			buffer.reanalyse(std::dynamic_pointer_cast<Model>(model));
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
	model_syncer.send(std::dynamic_pointer_cast<MCTSModelInterface>(model_->clone(torch::kCPU)));

	// Wait for the min number of episodes to be available in the buffer
	{
		using namespace std::chrono_literals;
		spdlog::info("Waiting for buffer to reach min required size");
		while (buffer.get_num_episodes() < train_config.start_buffer_size)
		{
			spdlog::fmt_lib::print("\rCompleted {}/{} episodes", buffer.get_num_episodes(), train_config.start_buffer_size);
			std::this_thread::sleep_for(200ms);
		}
	}

	// Run train loop
	for (; timestep < train_config.total_timesteps; ++timestep)
	{
		TrainUpdateData train_update_data;
		train_update_data.timestep = timestep;

		// Measure the model update step time.
		auto start = std::chrono::steady_clock::now();

		train_update_data.update_data = algorithm->update(timestep);
		train_update_data.update_data.push_back(
			{TrainResultType::kReanalyseCount, static_cast<double>(buffer.get_reanalysed_count())});
		model_syncer.send(std::dynamic_pointer_cast<MCTSModelInterface>(model_->clone(torch::kCPU)));

		train_update_data.update_duration = std::chrono::steady_clock::now() - start;
		train_update_data.env_duration = std::chrono::duration<double>(env_duration_stats_.get_mean());
		train_update_data.fps_env = env_samples_stats_.get_mean() / train_update_data.env_duration.count();
		train_update_data.fps = train_update_data.fps_env * config_.env_count;
		train_update_data.global_steps = total_samples_;

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

		using namespace std::chrono_literals;
		while ((timestep * train_config.train_ratio) > buffer.get_num_episodes()) { std::this_thread::sleep_for(10ms); }

		if (!training_)
		{
			break;
		}
	}

	training_ = false;

	// Block unitl all other threads are completed
	threadpool.wait_queue_empty();
}

void MCTSAgent::run(const std::vector<State>& initial_states, RunOptions options)
{
	const int env_count = initial_states.size();
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

	ThreadPool threadpool(initial_states.size());
	make_environments(threadpool, initial_states.size());

	// Block and wait for envs to be created
	threadpool.wait_queue_empty();

	load_model(options.force_model_reload);

	auto env_config = environment_manager_->get_configuration();

	for (size_t env = 0; env < initial_states.size(); ++env)
	{
		threadpool.queue_task([&, env]() {
			run_episode(std::dynamic_pointer_cast<MCTSModelInterface>(model_).get(), initial_states[env], env, true, options);
		});
	}

	// Wait for envs to terminate
	threadpool.wait_queue_empty();
}

ModelOutput MCTSAgent::predict_action(const std::vector<StepData>& step_history, bool deterministic)
{
	torch::NoGradGuard no_grad;
	return run_step(
		std::dynamic_pointer_cast<MCTSModelInterface>(model_).get(),
		step_history,
		deterministic,
		deterministic ? 0.0F : config_.temperature);
}

std::unique_ptr<MCTSEpisode> MCTSAgent::run_episode(
	MCTSModelInterface* model, const State& initial_state, int env, bool eval_mode, RunOptions options)
{
	auto env_config = environment_manager_->get_configuration();
	if (options.max_steps <= 0)
	{
		// Run the environment until it terminates (setting to max int should be sufficient)
		options.max_steps = std::numeric_limits<int>::max();
	}

	OpponentType op_type = eval_mode && env_config.num_actors > 1 ? config_.opponent_type : OpponentType::kSelf;

	std::vector<StepData> episode_data;

	auto environment = environment_manager_->get_environment(env);
	StepData step_data;
	step_data.eval_mode = eval_mode;
	step_data.env = env;
	step_data.step = 0;
	step_data.env_data = environment->reset(initial_state);
	step_data.predict_result = model->initial();
	step_data.reward = clamp_reward(step_data.env_data.reward, config_.rewards);
	auto agent_reset_config = agent_callback_->env_reset(step_data);
	episode_data.push_back(std::move(step_data));

	torch::NoGradGuard no_grad;

	bool raw_capture = agent_reset_config.raw_capture || options.capture_observations;

	for (int step = 1; step <= options.max_steps; step++)
	{
		step_data.env = env;
		step_data.step = step;

		auto turn_index = episode_data.back().env_data.turn_index;
		if (op_type != OpponentType::kSelf && turn_index != config_.actor_index)
		{
			// TODO: maybe add ability to use other MCTS based agents, or even different versions of the same agent.
			if (op_type == OpponentType::kExpert)
			{
				step_data.predict_result.action = environment->expert_agent();
			}
		}
		else
		{
			step_data.predict_result = run_step(model, episode_data, options.deterministic, options.temperature);
		}
		step_data.env_data = environment->step(step_data.predict_result.action);

		step_data.reward = clamp_reward(step_data.env_data.reward, config_.rewards);
		if (raw_capture)
		{
			step_data.raw_observation = environment->get_raw_observations();
		}

		bool stop = agent_callback_->env_step(step_data) || step_data.env_data.state.episode_end;

		episode_data.push_back(std::move(step_data));

		if (stop)
		{
			break;
		}
	}

	// Don't store the episode data in eval mode
	if (eval_mode)
	{
		return nullptr;
	}

	const auto& train_config = std::get<Config::MuZero::TrainConfig>(config_.train_algorithm);
	MCTSEpisodeOptions ep_options;
	ep_options.num_actions = static_cast<int>(flatten(environment->get_configuration().action_space.shape));
	ep_options.td_steps = train_config.td_steps;
	ep_options.unroll_steps = train_config.unroll_steps + 1; // add 1 to include the current step
	ep_options.stack_size = model->get_stacked_observation_size();
	return std::make_unique<MCTSEpisode>(std::move(episode_data), ep_options);
}

ModelOutput MCTSAgent::run_step(
	MCTSModelInterface* model, const std::vector<StepData>& step_history, bool deterministic, float temperature)
{
	auto env_config = environment_manager_->get_configuration();

	MCTS mcts(config_, env_config.action_set, env_config.num_actors);

	MCTSInput mcts_input;
	mcts_input.observation = get_stacked_observations(
		step_history, model->get_stacked_observation_size(), flatten(env_config.action_space.shape));
	mcts_input.legal_actions = step_history.back().env_data.legal_actions;
	mcts_input.turn_index = step_history.back().env_data.turn_index;
	mcts_input.add_exploration_noise = !deterministic;
	auto result = mcts.search(model, mcts_input);

	std::vector<float> node_visits(env_config.action_set.size(), 0);
	auto& nodes = result.root.get_children();
	assert(!nodes.empty());
	int max_visits = 0;
	size_t max_action = 0;
	for (size_t i = 0; i < nodes.size(); ++i)
	{
		auto index = mcts_input.legal_actions.at(i);
		auto visits = nodes[i].get_visit_count();
		if (visits > max_visits)
		{
			max_visits = visits;
			max_action = index;
		}
		node_visits.at(index) = static_cast<float>(visits);
	}

	float sum_count = 0;
	for (auto& node : node_visits)
	{
		node = std::pow(node, 1.0F / temperature);
		sum_count += node;
	}
	for (auto& node : node_visits) { node /= sum_count; }

	size_t action_index;
	if (temperature == 0.0F)
	{
		action_index = max_action;
	}
	else if (temperature == std::numeric_limits<float>::infinity())
	{
		std::uniform_int_distribution<size_t> action_dist(0, nodes.size() - 1);
		action_index = action_dist(gen_);
	}
	else
	{
		std::discrete_distribution<size_t> action_dist(node_visits.begin(), node_visits.end());
		action_index = action_dist(gen_);
	}

	ModelOutput prediction;
	for (const auto& node : nodes)
	{
		if (node.get_action() == static_cast<int>(action_index))
		{
			prediction = node.get_prediction();
			break;
		}
	}
	assert(!prediction.state.empty());
	prediction.policy = torch::from_blob(node_visits.data(), {static_cast<int>(node_visits.size())}).clone();
	return prediction;
}

Observations
MCTSAgent::get_stacked_observations(const std::vector<StepData>& step_history, int stack_size, int num_actions)
{
	std::vector<Observations> observation_history;
	std::vector<Observations> action_history;

	// TODO: maybe determine sizes first and preallocate then just iterate over and copy data
	// Maybe use multiple dims, then resize tensor

	int len = step_history.size();
	size_t obs_dims = step_history.back().env_data.observation.size();
	observation_history.resize(obs_dims);
	action_history.resize(obs_dims);
	for (int index = len - stack_size - 1; index < len; ++index)
	{
		if (index >= 0)
		{
			auto next_index = index + 1;
			const auto& observation = step_history.at(index).env_data.observation;
			for (size_t i = 0; i < obs_dims; ++i)
			{
				observation_history[i].push_back(observation[i]);
				if (next_index < len)
				{
					auto action = step_history.at(next_index).predict_result.action.to(torch::kCPU);
					if (observation[i].dim() < 3)
					{
						action_history[i].push_back(action);
					}
					else
					{
						// observation should have the shape [channels, height, width]
						action_history[i].push_back(torch::ones_like(observation[i][0]).mul(action.div(num_actions)).unsqueeze(0));
					}
				}
			}
		}
		else
		{
			auto& observation = step_history.back().env_data.observation;
			auto action = step_history.back().predict_result.action.to(torch::kCPU);
			for (size_t i = 0; i < obs_dims; ++i)
			{
				auto& obs = observation[i];
				observation_history[i].push_back(torch::zeros_like(obs));
				if (obs.dim() < 3)
				{
					action_history[i].push_back(torch::zeros_like(action));
				}
				else
				{
					auto shape = obs.sizes().vec();
					shape[0] = 1;
					action_history[i].push_back(torch::zeros(shape));
				}
			}
		}
	}

	Observations stacked_observations;
	for (size_t i = 0; i < obs_dims; ++i)
	{
		auto& obs = observation_history[i];
		auto& action = action_history[i];
		obs.insert(obs.end(), std::make_move_iterator(action.begin()), std::make_move_iterator(action.end()));
		if (obs.back().dim() < 3)
		{
			stacked_observations.emplace_back(torch::cat(obs, -1).unsqueeze(0));
		}
		else
		{
			stacked_observations.emplace_back(torch::cat(obs).unsqueeze(0));
		}
	}

	return stacked_observations;
}
