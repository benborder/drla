#include "episodic_per_buffer.h"

#include "functions.h"
#include "tensor_storage.h"

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
	if (!options_.path.empty())
	{
		std::filesystem::create_directories(options_.path);
	}
}

void EpisodicPERBuffer::add_episode(std::shared_ptr<Episode> episode)
{
	if (episode->length() == 0)
	{
		spdlog::error("Episode length must be non zero. The episode has not been added to the buffer.");
		return;
	}

	if (!options_.path.empty() && episode->get_path().empty())
	{
		episode->save(options_.path);
	}

	std::lock_guard lock(m_episodes_);

	// Initialise episode if it hasn't already been initialised
	if (episode->get_id() < 0)
	{
		episode->set_id(total_episodes_++);
		episode->init_priorities(gamma_, options_.per_alpha);
	}

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
	return sample_episodes(1, force_uniform).front();
}

std::vector<std::pair<std::shared_ptr<Episode>, float>>
EpisodicPERBuffer::sample_episodes(int num_samples, bool force_uniform) const
{
	std::lock_guard lock(m_episodes_);
	assert(!episodes_.empty() && "Cannot sample as the episode buffer is empty");

	std::vector<std::pair<std::shared_ptr<Episode>, float>> episodes;
	size_t episode_index = 0;
	float episode_prob = 0;
	if (force_uniform || !options_.use_per)
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
		if (probs_sum > 0)
		{
			for (auto& probs : ep_probs) { probs /= probs_sum; }
		}

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
			(*episode)->update_priorities(pos, priorities[i]);
		}
		++i;
	}
}

int EpisodicPERBuffer::get_reanalysed_count() const
{
	return reanalysed_count_;
}

torch::Tensor EpisodicPERBuffer::get_gamma() const
{
	return gamma_;
}

void EpisodicPERBuffer::load(const std::filesystem::path& path)
{
	for (const auto& entry : std::filesystem::directory_iterator(path))
	{
		if (entry.is_directory())
		{
			std::string directory_name = entry.path().filename().string();
			// Check if the directory name matches the pattern "episode_*"
			if (directory_name.compare(0, 8, "episode_") == 0)
			{
				add_episode(load_episode(entry.path()));
			}
		}
	}
}

void EpisodicPERBuffer::set_state_shapes(const StateShapes& shapes)
{
	state_shapes_ = shapes;
}
