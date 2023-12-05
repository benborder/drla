#pragma once

#include "agent_types.h"
#include "configuration.h"
#include "episode.h"
#include "model.h"
#include "types.h"

#include <torch/torch.h>

#include <deque>
#include <filesystem>
#include <mutex>
#include <random>
#include <vector>

namespace drla
{

struct EpisodicPERBufferOptions
{
	int buffer_size;
	int reward_shape;
	int unroll_steps;
	ActionSpace action_space;
	bool use_per;
	float per_alpha;
	std::string path = {};
};

/// @brief Episodic Prioritised Experience Replay Buffer
class EpisodicPERBuffer
{
public:
	EpisodicPERBuffer(std::vector<float> gamma, EpisodicPERBufferOptions options);

	/// @brief Adds an episode to the buffer
	/// @param episode The episode to add
	virtual void add_episode(std::shared_ptr<Episode> episode);

	/// @brief Gets the number of episodes currently stored in the buffer
	/// @return The number of episodes stored in the buffer
	int get_num_episodes() const;

	/// @brief Gets the total number of samples stored in the buffer
	/// @return The number of samples stored in the buffer
	virtual int get_num_samples() const;

	/// @brief Randomly sample an episode from the buffer using the distribution formed from the saved priorities.
	/// @return The sampled episode
	/// @param force_uniform Forces the sample distribution to be uniform rather than prioritised
	/// @return The sampled episode and its policy probability
	std::pair<std::shared_ptr<Episode>, float> sample_episode(bool force_uniform = false) const;

	/// @brief Randomly sample a series of episodes from the buffer using the distribution formed from the saved
	/// priorities.
	/// @param num_samples The number of episodes to sample from the buffer.
	/// @param force_uniform Forces the sample distribution to be uniform rather than prioritised
	/// @return The sampled episodes and their policy probability
	virtual std::vector<std::pair<std::shared_ptr<Episode>, float>>
	sample_episodes(int num_samples, bool force_uniform = false) const;

	/// @brief Updates the priorities of a series of episodes and episode step indexes
	/// @param priorities The list of new priorities to update
	/// @param indicies The episode id and step index for each priority entry
	void update_priorities(torch::Tensor priorities, const std::vector<std::pair<int, int>>& indicies);

	/// @brief Gets the number of times reanalyse has been run
	/// @return The reanalyse count
	int get_reanalysed_count() const;

	/// @brief Returns the discount factor gamma
	/// @return the gamma tensor
	torch::Tensor get_gamma() const;

	/// @brief Loads all episodes from a directory into the buffer
	/// @param path The path where episodes will attempts to be loaded from
	void load(const std::filesystem::path& path);

	/// @brief Sets the state shapes to use.
	/// @param shapes The shape of the model internal state.
	void set_state_shapes(const StateShapes& shapes);

protected:
	virtual std::shared_ptr<Episode> load_episode(const std::filesystem::path& path) = 0;

protected:
	const EpisodicPERBufferOptions options_;

	std::vector<std::shared_ptr<Episode>> episodes_;
	mutable std::mutex m_episodes_;
	size_t buffer_index_ = 0;
	int total_episodes_ = 0;
	int total_steps_ = 0;
	StateShapes state_shapes_;

	mutable std::mt19937 gen_;

	torch::Tensor gamma_;
	int reanalysed_count_ = 0;
};

} // namespace drla
