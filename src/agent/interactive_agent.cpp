#include "interactive_agent.h"

#include "model.h"
#include "threadpool.h"

#include <spdlog/spdlog.h>

using namespace torch;
using namespace drla;

InteractiveAgent::InteractiveAgent(
	const Config::Agent& config,
	EnvironmentManager* environment_manager,
	AgentCallbackInterface* callback,
	std::string data_path)
		: Agent(config, environment_manager, callback, std::move(data_path))
		, config_(std::get<Config::InteractiveAgent>(config))
{
}

InteractiveAgent::InteractiveAgent(
	const Config::InteractiveAgent& config,
	EnvironmentManager* environment_manager,
	AgentCallbackInterface* callback,
	std::string data_path)
		: Agent(config, environment_manager, callback, std::move(data_path)), config_(config)
{
}

void InteractiveAgent::train()
{
	spdlog::error("Training not available in Interactive mode!");
}

void InteractiveAgent::run(const std::vector<State>& initial_state, RunOptions options)
{
	const int env_count = initial_state.size();
	if (env_count <= 0)
	{
		spdlog::error("Error: The number of environments must be greater than 0!");
		return;
	}
	if (options.max_steps <= 0)
	{
		// Run the environment until it terminates (setting to max int should be sufficient)
		options.max_steps = std::numeric_limits<int>::max();
	}

	// Create environments and allocate threads
	ThreadPool threadpool(config_.env_count);
	make_environments(threadpool, env_count);

	std::vector<EnvStepData> env_datas;
	env_datas.resize(env_count);

	// Wait for environments to complete initialising
	threadpool.wait_queue_empty();

	for (int env = 0; env < env_count; ++env)
	{
		threadpool.queue_task(
			[&, env]() { env_datas[env] = environment_manager_->get_environment(env)->reset(initial_state[env]); });
	}

	// Wait for environment reset to complete
	threadpool.wait_queue_empty();

	auto env_config = environment_manager_->get_configuration();
	int reward_shape = config_.rewards.combine_rewards ? 1 : static_cast<int>(env_config.reward_types.size());

	for (int env = 0; env < env_count; env++)
	{
		{
			auto environment = environment_manager_->get_environment(env);
			// Get initial observation and state, running a env_step callback to update externally
			StepData step_data;
			step_data.env = env;
			step_data.step = 0;
			step_data.env_data = std::move(env_datas[env]);
			step_data.predict_result.action = torch::zeros(env_config.action_space.shape);
			step_data.reward = step_data.env_data.reward.clone();

			auto agent_reset_config = agent_callback_->env_reset(step_data);
			if (agent_reset_config.stop)
			{
				return;
			}
			options.enable_visualisations |= agent_reset_config.enable_visualisations;
			if (options.enable_visualisations)
			{
				step_data.visualisation = environment->get_visualisations();
			}

			bool stop = agent_callback_->env_step(step_data);
			if (stop)
			{
				continue;
			}
		}
		threadpool.queue_task([&, env]() {
			auto environment = environment_manager_->get_environment(env);

			StepData step_data;
			step_data.env = env;
			step_data.predict_result.values = torch::zeros({1, reward_shape});

			for (int step = 0; step < options.max_steps; step++)
			{
				step_data.step = step;
				step_data.predict_result.action = agent_callback_->interactive_step()[env];
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
				if (options.enable_visualisations)
				{
					step_data.visualisation = environment->get_visualisations();
				}

				bool stop = agent_callback_->env_step(step_data);
				if (stop)
				{
					break;
				}

				if (step_data.env_data.state.episode_end)
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

void InteractiveAgent::reset()
{
	environment_manager_->reset();
}
