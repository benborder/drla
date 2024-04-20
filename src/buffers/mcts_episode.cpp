#include "mcts_episode.h"

#include "functions.h"
#include "tensor_storage.h"
#include "utils.h"

#include <spdlog/spdlog.h>

#include <algorithm>

using namespace drla;

MCTSEpisode::MCTSEpisode(std::vector<StepData> episode_data, MCTSEpisodeOptions options)
		: options_(options), episode_length_(static_cast<int>(episode_data.size() - 1))
{
	assert(episode_length_ > 0);
	assert(options_.stack_size >= 0);

	// Skip the first step as it doesn't contain policy and value data
	const auto& initial_step_data = episode_data.at(1);
	for (auto& obs : initial_step_data.env_data.observation)
	{
		auto dims = obs.sizes().vec();
		dims.insert(dims.begin(), episode_length_ + 1);
		observations_.push_back(torch::empty(dims, obs.scalar_type()));
	}
	{
		auto dims = initial_step_data.predict_result.action.sizes().vec();
		dims.insert(dims.begin(), episode_length_ + 1);
		actions_ = torch::empty(dims);
	}
	{
		auto dims = initial_step_data.reward.sizes().vec();
		dims.insert(dims.begin(), episode_length_ + 1);
		rewards_ = torch::empty(dims);
	}
	{
		auto dims = initial_step_data.predict_result.policy.sizes().vec();
		dims.insert(dims.begin(), episode_length_);
		policy_ = torch::empty(dims);
	}
	{
		auto dims = initial_step_data.predict_result.values.sizes().vec();
		dims.insert(dims.begin(), episode_length_);
		values_ = torch::empty(dims);
	}
	turn_index_.resize(episode_length_ + 1);

	size_t obs_dims = initial_step_data.env_data.observation.size();
	for (auto& step_data : episode_data)
	{
		for (size_t i = 0; i < obs_dims; ++i) { observations_[i][step_data.step] = step_data.env_data.observation[i]; }
		actions_[step_data.step] = step_data.predict_result.action;
		rewards_[step_data.step] = step_data.reward;
		turn_index_[step_data.step] = step_data.env_data.turn_index;
		if (step_data.step > 0)
		{
			policy_[step_data.step - 1] = step_data.predict_result.policy;
			values_[step_data.step - 1] = step_data.predict_result.values;
		}
	}
}

MCTSEpisode::MCTSEpisode(const std::filesystem::path& path, MCTSEpisodeOptions options)
		: options_(options), episode_length_(0), saved_path_(path)
{
	actions_ = load_tensor(path / "actions.bin");
	rewards_ = load_tensor(path / "rewards.bin");
	policy_ = load_tensor(path / "policy.bin");
	observations_ = load_tensor_vector(path, "observations");
	turn_index_ = load_vector(path / "turn_index.bin");
	episode_length_ = static_cast<int>(actions_.size(0) - 1);
	values_ = torch::zeros(std::vector<int64_t>{episode_length_, rewards_.size(1)});
}

void MCTSEpisode::set_id(int id)
{
	id_ = id;
}

int MCTSEpisode::get_id() const
{
	return id_;
}

Observations MCTSEpisode::get_observations(int step, torch::Device device) const
{
	Observations stacked_obs;
	// Clamp the step to the max episode length
	step = std::min(step, episode_length_);
	int past_step = std::max(step - options_.stack_size, 0);
	int stack_size = step - past_step;
	int zero_size = options_.stack_size - stack_size;
	for (auto& obs : observations_)
	{
		auto dims = obs.dim();
		auto obs_slice = convert_observation(obs.narrow(0, past_step, stack_size + 1), device, false);
		auto action_slice = actions_.narrow(0, past_step + 1, stack_size).to(device);
		auto shape = obs_slice.sizes().vec();
		// Extend with zeros to make the full stack size if necessary
		if (zero_size > 0)
		{
			shape[0] = zero_size;
			obs_slice = torch::cat({obs_slice, torch::zeros(shape, device)});
			action_slice = torch::cat({action_slice, torch::zeros({zero_size, action_slice.size(1)}, device)});
		}
		if (dims < 3)
		{
			// assume shape is [step, data]
			if (options_.stack_size > 0)
			{
				stacked_obs.push_back(torch::cat({obs_slice.view({-1}), action_slice.view({-1})}, -1));
			}
		}
		else
		{
			// assume shape is [step, channels, height, width, ...]
			shape.erase(shape.begin());
			shape[0] = -1;
			// reshape to [stack, height, width, ...]
			obs_slice = obs_slice.view(shape);
			if (options_.stack_size > 0)
			{
				shape[0] = options_.stack_size;
				// Assuming we can broadcast here [stack, height, width, ...] * [stack, 1]
				auto actions = torch::ones(shape, device).mul(action_slice.div(options_.num_actions).unsqueeze(-1));
				stacked_obs.push_back(torch::cat({obs_slice, actions}));
			}
		}
		if (options_.stack_size == 0)
		{
			if (dims == 2)
			{
				obs_slice.squeeze_(0);
			}
			stacked_obs.push_back(obs_slice);
		}
	}
	return stacked_obs;
}

ObservationShapes MCTSEpisode::get_observation_shapes() const
{
	ObservationShapes shapes;
	for (const auto& obs : observations_)
	{
		// assume shape is [step, channels, ...]
		auto dims = obs.sizes().vec();
		dims.erase(dims.begin());
		if (dims.size() < 3)
		{
			dims.back() *= options_.stack_size + 1;
			dims.back() += options_.stack_size;
		}
		else
		{
			dims.front() *= options_.stack_size + 1;
			dims.front() += options_.stack_size;
		}
		shapes.push_back(dims);
	}
	return shapes;
}

StateShapes MCTSEpisode::get_state_shapes() const
{
	return {};
}

void MCTSEpisode::init_priorities(torch::Tensor gamma, float per_alpha)
{
	priorities_.resize(episode_length_);
	for (int i = 0; i < episode_length_; ++i)
	{
		auto priority = (values_[i] - compute_target_value(i, gamma)).abs().pow(per_alpha);
		if (priority.dim() > 1)
		{
			priority = priority.view({priority.size(0), -1}).sum(1);
		}
		auto p = priority.item<float>();
		if (p > episode_priority_)
		{
			episode_priority_ = p;
		}
		priorities_[i] = p;
	}
}

void MCTSEpisode::update_priorities(int index, torch::Tensor priorities)
{
	index = (index + episode_length_) % episode_length_;
	std::lock_guard lock(m_updates_);
	for (int i = index, end = std::min<int>(i + priorities.size(0), episode_length_); i < end; ++i)
	{
		priorities_[i] = priorities[i - index].item<float>();
	}
	episode_priority_ = *std::max_element(priorities_.begin(), priorities_.end());
}

float MCTSEpisode::get_priority() const
{
	return episode_priority_;
}

std::pair<int, float> MCTSEpisode::sample_position(std::mt19937& gen, bool force_uniform) const
{
	std::lock_guard lock(m_updates_);
	int step_index = 0;
	if (force_uniform)
	{
		std::uniform_int_distribution<int> step_dist(0, priorities_.size() - 1);
		step_index = step_dist(gen);
	}
	else
	{
		std::discrete_distribution<int> step_dist(priorities_.begin(), priorities_.end());
		step_index = step_dist(gen);
	}
	return {step_index, priorities_[step_index]};
}

torch::Tensor MCTSEpisode::compute_target_value(int index, torch::Tensor gamma) const
{
	std::lock_guard lock(m_updates_);
	torch::Tensor value;
	auto bootstrap_index = index + options_.td_steps;
	if (bootstrap_index < episode_length_)
	{
		auto last_step_value =
			reanalysed_values_.numel() > 0 ? reanalysed_values_[bootstrap_index] : values_[bootstrap_index];
		if (turn_index_[bootstrap_index] != turn_index_[index])
		{
			last_step_value.neg_();
		}

		value = last_step_value * gamma.pow(options_.td_steps);
	}
	else
	{
		value = torch::zeros_like(values_[0]);
	}

	for (int i = index + 1, n = std::min(bootstrap_index, episode_length_) + 1; i < n; ++i)
	{
		auto reward = rewards_[i];
		if (turn_index_[i] != turn_index_[i + 1])
		{
			reward.neg_();
		}
		value += reward * gamma.pow(i - index - 1);
	}
	return value;
}

EpisodeSampleTargets MCTSEpisode::make_target(int index, torch::Tensor gamma) const
{
	EpisodeSampleTargets target;
	{
		auto dims = actions_.sizes().vec();
		dims[0] = options_.unroll_steps;
		target.actions = torch::empty(dims);
	}
	{
		auto dims = policy_.sizes().vec();
		dims[0] = options_.unroll_steps;
		target.policies = torch::empty(dims);
	}
	{
		auto dims = values_.sizes().vec();
		dims[0] = options_.unroll_steps;
		target.rewards = torch::empty(dims);
		target.values = torch::empty(dims);
	}
	target.non_terminal = torch::ones({options_.unroll_steps});

	// TODO: potential optimisation: if j < episode_length_ then use a direct narrow for rewards/policies/actions and
	// only iterate for compute_target_value(). Not sure if will actually make a difference though

	for (int i = 0; i < options_.unroll_steps; ++i)
	{
		int j = index + i;
		if (j < episode_length_)
		{
			target.actions[i] = actions_[j];
			target.policies[i] = policy_[j];
			target.rewards[i] = rewards_[j];
			target.values[i] = compute_target_value(j, gamma);
		}
		else if (j == episode_length_)
		{
			target.actions[i] = actions_[j];
			target.policies[i].fill_(1.0F / policy_.size(1));
			target.rewards[i] = rewards_[j];
			target.values[i] = torch::zeros_like(values_[0]);
			target.non_terminal[i] = 0;
		}
		else
		{
			target.actions[i] = torch::randint_like(actions_[0], options_.num_actions);
			target.policies[i].fill_(1.0F / policy_.size(1));
			target.rewards[i] = torch::zeros_like(rewards_[0]);
			target.values[i] = torch::zeros_like(values_[0]);
			target.non_terminal[i] = 0;
		}
	}

	return target;
}

void MCTSEpisode::update_values(const torch::Tensor& values)
{
	std::lock_guard lock(m_updates_);
	reanalysed_values_ = values;
}

void MCTSEpisode::update_states([[maybe_unused]] const HiddenStates& states)
{
}

int MCTSEpisode::length() const
{
	return episode_length_;
}

void MCTSEpisode::set_sequence_length([[maybe_unused]] int length)
{
}

void MCTSEpisode::save(const std::filesystem::path& path)
{
	if (!std::filesystem::exists(path))
	{
		spdlog::error("Unable to save episode data: The path '{}' does not exist.", path.string());
		return;
	}
	auto ep_path = path / ("episode_" + options_.name);
	std::filesystem::create_directory(ep_path);

	save_tensor(actions_, ep_path / "actions.bin");
	save_tensor(rewards_, ep_path / "rewards.bin");
	save_tensor(policy_, ep_path / "policy.bin");
	save_vector(turn_index_, ep_path / "turn_index.bin");

	for (size_t i = 0; i < observations_.size(); ++i)
	{
		save_tensor(observations_[i], ep_path / (std::string{"observations"} + std::to_string(i) + ".bin"));
	}
	saved_path_ = ep_path;
}

const std::filesystem::path& MCTSEpisode::get_path() const
{
	return saved_path_;
}
