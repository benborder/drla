#include "hybrid_replay_buffer.h"

#include "functions.h"
#include "hybrid_episode.h"
#include "utils.h"

using namespace drla;

HybridReplayBuffer::HybridReplayBuffer(std::vector<float> gamma, int n_envs, EpisodicPERBufferOptions options)
		: EpisodicPERBuffer(gamma, std::move(options))
{
	new_episodes_.resize(n_envs);
	inprogress_episodes_.resize(n_envs);
}

void HybridReplayBuffer::add(StepData step_data, bool force_cache)
{
	int env = step_data.env;
	int step = step_data.step;
	bool episode_end = step_data.env_data.state.episode_end;
	auto& ep = new_episodes_.at(env);
	// At least a min of 'unroll_steps' must exist before creating a HybridEpisode in inprogress_episodes_
	if (step <= options_.unroll_steps || force_cache || static_cast<int>(ep.size()) > options_.unroll_steps)
	{
		static auto opt =
			HybridEpisodeOptions{static_cast<int>(flatten(options_.action_space.shape)), options_.unroll_steps};
		ep.push_back(std::move(step_data));
		if (episode_end)
		{
			add_episode(std::make_shared<HybridEpisode>(std::move(ep), opt));
		}
		else if (step >= options_.unroll_steps && !force_cache)
		{
			auto episode = std::make_shared<HybridEpisode>(std::move(ep), opt);
			m_episodes_.lock();
			episode->set_id(total_episodes_++);
			m_episodes_.unlock();
			episode->init_priorities(gamma_, options_.per_alpha);
			inprogress_episodes_.at(env) = std::move(episode);
		}
	}
	else
	{
		auto& episode = inprogress_episodes_.at(env);
		episode->add_step(std::move(step_data));
		episode->init_priorities(gamma_, options_.per_alpha);
		if (episode_end)
		{
			add_episode(std::move(inprogress_episodes_.at(env)));
		}
	}
}

std::vector<std::pair<std::shared_ptr<Episode>, float>>
HybridReplayBuffer::sample_episodes(int num_samples, bool force_uniform) const
{
	std::lock_guard lock(m_episodes_);
	assert(!(episodes_.empty() && inprogress_episodes_.empty()) && "Cannot sample as the episode buffer is empty");

	std::vector<std::pair<std::shared_ptr<Episode>, float>> episodes;
	size_t episode_index = 0;
	float episode_prob = 0;
	int num_eps = episodes_.size();
	// Only include inprogress episodes if they exist
	for (auto& ep : inprogress_episodes_) { num_eps += ep != nullptr ? 1 : 0; }
	if (force_uniform)
	{
		std::uniform_int_distribution<size_t> episode_dist(0, num_eps - 1);
		episode_prob = 1.0 / num_eps;
		for (int s = 0; s < num_samples; ++s)
		{
			episode_index = episode_dist(gen_);
			if (episode_index < episodes_.size())
			{
				episodes.emplace_back(episodes_[episode_index], episode_prob);
			}
			else
			{
				episodes.emplace_back(inprogress_episodes_.at(episode_index - episodes_.size()), episode_prob);
			}
		}
	}
	else
	{
		float probs_sum = 0;
		std::vector<float> ep_probs;
		ep_probs.reserve(num_eps);
		for (auto& episode : episodes_)
		{
			probs_sum += episode->get_priority();
			ep_probs.push_back(episode->get_priority());
		}
		for (auto& episode : inprogress_episodes_)
		{
			if (episode != nullptr)
			{
				probs_sum += episode->get_priority();
				ep_probs.push_back(episode->get_priority());
			}
		}
		for (auto& probs : ep_probs) { probs /= probs_sum; }

		std::discrete_distribution<size_t> episode_dist(ep_probs.begin(), ep_probs.end());
		for (int s = 0; s < num_samples; ++s)
		{
			episode_index = episode_dist(gen_);
			episode_prob = ep_probs[episode_index];
			if (episode_index < episodes_.size())
			{
				episodes.emplace_back(episodes_[episode_index], episode_prob);
			}
			else
			{
				episodes.emplace_back(inprogress_episodes_.at(episode_index - episodes_.size()), episode_prob);
			}
		}
	}
	return episodes;
}

HybridBatch HybridReplayBuffer::sample(int batch_size, torch::Device device) const
{
	HybridBatch batch;

	const auto& action_space_shape = options_.action_space.shape;
	std::vector<int64_t> action_shape{batch_size, options_.unroll_steps};
	c10::ScalarType action_type;
	if (is_action_discrete(options_.action_space))
	{
		action_type = torch::kLong;
		action_shape.push_back(static_cast<int>(action_space_shape.size()));
	}
	else
	{
		action_type = torch::kFloat;
		action_shape.push_back(std::accumulate(action_space_shape.begin(), action_space_shape.end(), 0));
	}
	batch.indicies.reserve(batch_size);
	auto episodes = sample_episodes(batch_size, !options_.use_per);
	const auto& ep = episodes.front().first;
	auto observation_shapes = ep->get_observation_shapes();
	for (auto& obs_shape : observation_shapes)
	{
		obs_shape.insert(obs_shape.begin(), batch_size);
		batch.observation.push_back(torch::empty(obs_shape, device));
	}
	auto state_shapes = ep->get_state_shapes();
	for (auto& sshape : state_shapes)
	{
		sshape.insert(sshape.begin(), batch_size);
		batch.states.push_back(torch::empty(sshape, device));
	}
	batch.action = torch::empty(action_shape, torch::TensorOptions(device).dtype(action_type));
	batch.reward = torch::empty({batch_size, options_.unroll_steps, options_.reward_shape}, device);
	batch.values = torch::empty({batch_size, options_.unroll_steps, options_.reward_shape}, device);
	batch.non_terminal = torch::empty({batch_size, options_.unroll_steps}, device);
	batch.weight = torch::empty({batch_size}, device);
	batch.is_first = torch::zeros({batch_size, options_.unroll_steps}, device);

	int batch_index = 0;
	for (const auto& [episode, episode_prob] : episodes)
	{
		auto [index, probs] = episode->sample_position(gen_, !options_.use_per);
		auto target = episode->make_target(index, gamma_);
		auto sample_obs = episode->get_observations(index, device);

		batch.indicies.emplace_back(episode->get_id(), index);
		for (size_t i = 0; i < batch.observation.size(); ++i)
		{
			batch.observation[i][batch_index] = convert_observation(sample_obs[i].detach(), device, false);
		}
		for (size_t i = 0; i < batch.states.size(); ++i) { batch.states[i][batch_index] = target.states[i].detach(); }
		batch.action[batch_index] = target.actions.detach().to(device);
		batch.reward[batch_index] = target.rewards.detach().to(device);
		batch.non_terminal[batch_index] = target.non_terminal.detach().to(device);
		batch.values[batch_index] = target.values.detach().to(device);
		batch.weight[batch_index] = 1.0F / (episode_prob * probs);
		batch.is_first[batch_index][0] = index == 0 ? 1.0F : 0.0F;
		++batch_index;
	}

	batch.weight.div_(batch.weight.max());

	return batch;
}

int HybridReplayBuffer::get_num_samples() const
{
	int in_progress_steps = 0;
	for (auto& ep : new_episodes_) { in_progress_steps += ep.size(); }
	for (auto& ep : inprogress_episodes_)
	{
		if (ep != nullptr)
		{
			in_progress_steps += ep->length();
		}
	}
	return total_steps_ + in_progress_steps;
}

void HybridReplayBuffer::reanalyse(std::shared_ptr<HybridModelInterface> model)
{
	auto [ep, probs] = sample_episode(/*force_uniform=*/true);
	auto episode = std::dynamic_pointer_cast<HybridEpisode>(ep);
	auto device = model->parameters().front().device(); // use the same device as the model

	torch::NoGradGuard no_grad;

	int len = episode->length();
	torch::Tensor values = torch::zeros({len, options_.reward_shape});
	HiddenStates states;
	auto state_shape = model->get_state_shape();
	for (auto& state : state_shape) { states.push_back(torch::empty(std::vector<int64_t>{len + 1} + state)); }
	ModelInput input;
	input.prev_output = model->initial();
	for (size_t i = 0; i < states.size(); ++i) { states[i][0] = input.prev_output.state[i].squeeze(0); }
	for (int step = 0; step < len; ++step)
	{
		input.observations = episode->get_observation(step, device);
		input.prev_output = model->predict(input);
		values[step] = input.prev_output.values.squeeze(0);
		for (size_t i = 0; i < states.size(); ++i) { states[i][step + 1] = input.prev_output.state[i].squeeze(0); }
		input.prev_output.action = episode->get_action(step + 1).to(device);
	}

	episode->update_values(values);
	episode->update_states(states);
	++reanalysed_count_;
}