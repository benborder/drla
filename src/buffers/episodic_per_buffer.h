#pragma once

#include "agent_types.h"
#include "configuration.h"
#include "model.h"
#include "types.h"

#include <torch/torch.h>

#include <deque>
#include <mutex>
#include <random>
#include <vector>

namespace drla
{

struct EpisodeSampleTargets
{
	torch::Tensor actions;
	torch::Tensor policies;
	torch::Tensor rewards;
	torch::Tensor values;
	torch::Tensor non_terminal;
};

class Episode
{
public:
	virtual void set_id(int id) = 0;
	virtual int get_id() const = 0;
	virtual Observations get_stacked_observations(int step, torch::Device device) const = 0;
	virtual ObservationShapes get_observation_shapes() const = 0;
	virtual void init_priorities(torch::Tensor gamma, float per_alpha = 1.0F) = 0;
	virtual void update_priorities(int index, torch::Tensor priorities) = 0;
	virtual float get_priority() const = 0;
	virtual std::pair<int, float> sample_position(std::mt19937& gen, bool force_uniform = false) const = 0;
	// The value target is the discounted root value of the search tree td_steps into the future, plus the discounted sum
	// of all rewards until then.
	virtual torch::Tensor compute_target_value(int index, torch::Tensor gamma) const = 0;
	virtual EpisodeSampleTargets make_target(int index, torch::Tensor gamma) const = 0;
	virtual void update_values(torch::Tensor values) = 0;
	virtual int length() const = 0;
};

struct Batch
{
	// episode id and episode step index
	std::vector<std::pair<int, int>> indicies;
	Observations observation;
	torch::Tensor reward;
	torch::Tensor values;
	torch::Tensor policy;
	torch::Tensor action;
	torch::Tensor non_terminal;
	torch::Tensor weight;
	torch::Tensor gradient_scale;
};

struct EpisodicPERBufferOptions
{
	int buffer_size;
	int reward_shape;
	int unroll_steps;
	float per_alpha;
	ActionSpace action_space;
};

/// @brief Episodic Prioritised Experience Replay Buffer
class EpisodicPERBuffer
{
public:
	EpisodicPERBuffer(std::vector<float> gamma, EpisodicPERBufferOptions options);

	/// @brief Adds an episode to the buffer
	/// @param episode The episode to add
	void add_episode(std::shared_ptr<Episode> episode);

	/// @brief Gets the number of episodes currently stored in the buffer
	/// @return The number of episodes stored in the buffer
	int get_num_episodes() const;

	/// @brief Gets the total number of samples stored in the buffer
	/// @return The number of samples stored in the buffer
	int get_num_samples() const;

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
	std::vector<std::pair<std::shared_ptr<Episode>, float>>
	sample_episodes(int num_samples, bool force_uniform = false) const;

	/// @brief Samples from the buffer, using the priorities to form a distribution to sample from
	/// @param batch_size The number of step index samples to retrieve
	/// @param device The device the samples should be on
	Batch sample(int batch_size, torch::Device device = torch::kCPU) const;

	/// @brief Updates the priorities of a series of episodes and episode step indexes
	/// @param priorities The list of new priorities to update
	/// @param indicies The episode id and step index for each priority entry
	void update_priorities(torch::Tensor priorities, const std::vector<std::pair<int, int>>& indicies);

	/// @brief Reanalyse a random epiosde in the buffer
	/// @param model The model to use when reanalysing
	void reanalyse(std::shared_ptr<Model> model);

	/// @brief Gets the number of times reanalyse has been run
	/// @return The reanalyse count
	int get_reanalysed_count() const;

	/// @brief Set a function which decodes values from a model when reanalysing
	/// @param decoder The decoder function to use
	void set_value_decoder(std::function<torch::Tensor(torch::Tensor&)> decoder);

private:
	const EpisodicPERBufferOptions options_;

	std::vector<std::shared_ptr<Episode>> episodes_;
	mutable std::mutex m_episodes_;
	size_t buffer_index_ = 0;
	int total_episodes_ = 0;
	int total_steps_ = 0;

	mutable std::mt19937 gen_;

	torch::Tensor gamma_;
	int reanalysed_count_ = 0;

	std::function<torch::Tensor(torch::Tensor&)> value_decoder_;
};

} // namespace drla
