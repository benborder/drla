#include "episodic_per_buffer.h"

#include <spdlog/spdlog.h>

#include <random>

using namespace drla;

EpisodicPERBuffer::EpisodicPERBuffer(std::vector<float> gamma, EpisodicPERBufferOptions options)
		: options_(std::move(options)), gen_(std::random_device{}())
{
	if (options_.buffer_size <= 0)
	{
		throw std::invalid_argument("PER Buffer size must be larger than 0");
	}

	if (gamma.size() < static_cast<size_t>(options_.reward_shape))
	{
		gamma.resize(options_.reward_shape, gamma.front());
	}
	gamma_ = torch::from_blob(gamma.data(), {options_.reward_shape}).clone();
	episodes_.reserve(options_.buffer_size);
}

void EpisodicPERBuffer::add_episode(std::shared_ptr<Episode> episode)
{
	if (episode->length() == 0)
	{
		spdlog::error("Episode length must be non zero. The episode has not been added to the buffer.");
		return;
	}

	std::lock_guard lock(m_episodes_);

	episode->set_id(total_episodes_++);
	episode->init_priorities(gamma_, options_.per_alpha);

	total_steps_ += episode->length();
	if (episodes_.size() < static_cast<size_t>(options_.buffer_size))
	{
		episodes_.push_back(std::move(episode));
	}
	else
	{
		total_steps_ -= episodes_.back()->length();
		episodes_.back() = std::move(episode);
	}
	std::sort(episodes_.begin(), episodes_.end(), [](const auto& e1, const auto& e2) {
		return e1->get_priority() > e2->get_priority();
	});
}

int EpisodicPERBuffer::get_num_episodes() const
{
	return total_episodes_;
}

int EpisodicPERBuffer::get_num_samples() const
{
	return total_steps_;
}

std::pair<std::shared_ptr<Episode>, float> EpisodicPERBuffer::sample_episode(bool force_uniform) const
{
	std::lock_guard lock(m_episodes_);
	assert(!episodes_.empty() && "Cannot sample as the episode buffer is empty");

	size_t episode_index = 0;
	float episode_prob = 0;
	if (force_uniform)
	{
		std::uniform_int_distribution<size_t> episode_dist(0, episodes_.size() - 1);
		episode_index = episode_dist(gen_);
		episode_prob = 1.0 / episodes_.size();
	}
	else
	{
		float probs_sum = 0;
		std::vector<float> ep_probs;
		ep_probs.reserve(episodes_.size());
		for (auto& episode : episodes_)
		{
			probs_sum += episode->get_priority();
			ep_probs.push_back(episode->get_priority());
		}
		for (auto& probs : ep_probs) { probs /= probs_sum; }

		std::discrete_distribution<size_t> episode_dist(ep_probs.begin(), ep_probs.end());
		episode_index = episode_dist(gen_);
		episode_prob = ep_probs[episode_index];
	}
	return {episodes_[episode_index], episode_prob};
}

std::vector<std::pair<std::shared_ptr<Episode>, float>>
EpisodicPERBuffer::sample_episodes(int num_samples, bool force_uniform) const
{
	std::lock_guard lock(m_episodes_);
	assert(!episodes_.empty() && "Cannot sample as the episode buffer is empty");

	std::vector<std::pair<std::shared_ptr<Episode>, float>> episodes;
	size_t episode_index = 0;
	float episode_prob = 0;
	if (force_uniform)
	{
		std::uniform_int_distribution<size_t> episode_dist(0, episodes_.size() - 1);
		episode_prob = 1.0 / episodes_.size();
		for (int s = 0; s < num_samples; ++s)
		{
			episode_index = episode_dist(gen_);
			episodes.emplace_back(episodes_[episode_index], episode_prob);
		}
	}
	else
	{
		float probs_sum = 0;
		std::vector<float> ep_probs;
		ep_probs.reserve(episodes_.size());
		for (auto& episode : episodes_)
		{
			probs_sum += episode->get_priority();
			ep_probs.push_back(episode->get_priority());
		}
		for (auto& probs : ep_probs) { probs /= probs_sum; }

		std::discrete_distribution<size_t> episode_dist(ep_probs.begin(), ep_probs.end());
		for (int s = 0; s < num_samples; ++s)
		{
			episode_index = episode_dist(gen_);
			episode_prob = ep_probs[episode_index];
			episodes.emplace_back(episodes_[episode_index], episode_prob);
		}
	}
	return episodes;
}

Batch EpisodicPERBuffer::sample(int batch_size, torch::Device device) const
{
	Batch batch;

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
	auto policy_actions = std::accumulate(action_space_shape.begin(), action_space_shape.end(), 1, std::multiplies<>());

	batch.indicies.reserve(batch_size);
	auto observation_shapes = episodes_.front()->get_observation_shapes();
	for (auto& obs_shape : observation_shapes)
	{
		obs_shape.insert(obs_shape.begin(), batch_size);
		batch.observation.push_back(torch::empty(obs_shape, device));
	}
	batch.action = torch::empty(action_shape, torch::TensorOptions(device).dtype(action_type));
	batch.policy = torch::empty({batch_size, options_.unroll_steps, policy_actions}, device);
	batch.reward = torch::empty({batch_size, options_.unroll_steps, options_.reward_shape}, device);
	batch.values = torch::empty({batch_size, options_.unroll_steps, options_.reward_shape}, device);
	batch.non_terminal = torch::empty({batch_size, options_.unroll_steps}, device);
	batch.weight = torch::empty({batch_size}, device);
	batch.gradient_scale = torch::empty({batch_size, options_.unroll_steps}, device);

	int batch_index = 0;
	auto episodes = sample_episodes(batch_size);
	for (const auto& [episode, episode_prob] : episodes)
	{
		auto [index, probs] = episode->sample_position(gen_);
		auto target = episode->make_target(index, gamma_);
		auto sample_obs = episode->get_stacked_observations(index, device);

		batch.indicies.emplace_back(episode->get_id(), index);
		for (size_t i = 0; i < batch.observation.size(); ++i)
		{
			batch.observation[i][batch_index] = sample_obs[i].detach().to(device);
		}
		batch.action[batch_index] = target.actions.detach().to(device);
		batch.policy[batch_index] = target.policies.detach().to(device);
		batch.reward[batch_index] = target.rewards.detach().to(device);
		batch.values[batch_index] = target.values.detach().to(device);
		batch.non_terminal[batch_index] = target.non_terminal.detach().to(device);
		batch.weight[batch_index] = 1.0F / (total_steps_ * episode_prob * probs);
		batch.gradient_scale[batch_index].fill_(std::min(options_.unroll_steps, episode->length() - index));
		++batch_index;
	}

	batch.weight.div_(batch.weight.max());

	return batch;
}

void EpisodicPERBuffer::update_priorities(torch::Tensor priorities, const std::vector<std::pair<int, int>>& indicies)
{
	std::lock_guard lock(m_episodes_);
	int i = 0;
	for (auto [id, pos] : indicies)
	{
		auto episode = std::find_if(episodes_.begin(), episodes_.end(), [epid = id](const std::shared_ptr<Episode>& ep) {
			return ep->get_id() == epid;
		});
		if (episode != episodes_.end())
		{
			(*episode)->update_priorities(pos, priorities.narrow(0, i, 1).squeeze(0));
		}
		++i;
	}
}

void EpisodicPERBuffer::reanalyse(std::shared_ptr<Model> model)
{
	auto [episode, probs] = sample_episode(/*force_uniform=*/true);
	auto device = model->parameters().front().device(); // use the same device as the model

	torch::NoGradGuard no_grad;

	int len = episode->length();
	torch::Tensor values = torch::zeros({len, options_.reward_shape});
	ModelOutput prediction;
	for (int step = 0; step < len; ++step)
	{
		ModelInput input;
		input.observations = episode->get_stacked_observations(step, device);
		for (auto& obs : input.observations) { obs.unsqueeze_(0); }
		input.prev_output = prediction;
		prediction = model->predict(input);
		values[step] = (value_decoder_ ? value_decoder_(prediction.values) : prediction.values).squeeze(0);
	}

	episode->update_values(values);
	++reanalysed_count_;
}

int EpisodicPERBuffer::get_reanalysed_count() const
{
	return reanalysed_count_;
}

void EpisodicPERBuffer::set_value_decoder(std::function<torch::Tensor(torch::Tensor&)> decoder)
{
	value_decoder_ = std::move(decoder);
}

torch::Tensor EpisodicPERBuffer::get_gamma() const
{
	return gamma_;
}
