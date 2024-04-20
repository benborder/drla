#pragma once

#include "types.h"

#include <ATen/core/Tensor.h>

#include <filesystem>
#include <random>

namespace drla
{

struct EpisodeSampleTargets
{
	torch::Tensor actions;
	torch::Tensor policies;
	torch::Tensor rewards;
	torch::Tensor values;
	HiddenStates states;
	torch::Tensor non_terminal;
};

/// @brief Interface for episodes stored in replay buffers
class Episode
{
public:
	/// @brief Sets the episode id. This id should be unique.
	/// @param id The id to set. A value < 0 implies uninitialised
	virtual void set_id(int id) = 0;

	/// @brief Get the unique episode id.
	/// @return The episode id
	virtual int get_id() const = 0;

	/// @brief Retrieve a sequence of observations starting from the specified step. This method retrieves a sequence of
	/// observations starting from the specified step within an episode. The exact nature of the returned observations
	/// varies depending on the derived type implementing this method.
	/// @param step The step for which observations are requested.
	/// @param device The device to copy the observations to.
	/// @return The observations at the given `step`.
	virtual Observations get_observations(int step, torch::Device device) const = 0;

	/// @brief Gets the observation shape, where the sequence size is the 0-dim
	/// @return The shape of the observation
	virtual ObservationShapes get_observation_shapes() const = 0;

	/// @brief Gets the state shape, where the sequence size is the 0-dim
	/// @return The shape of the state
	virtual StateShapes get_state_shapes() const = 0;

	/// @brief Initialises the priorities for prioritised experience episode sampling
	/// @param gamma The discount factor
	/// @param per_alpha The ammount of prioritised experience replay to use. A value in the range of (0,1]. Defaults
	/// to 1.
	virtual void init_priorities(torch::Tensor gamma, float per_alpha = 1.0F) = 0;

	/// @brief Updates the priorities at the specific step index
	/// @param index The step index to update from
	/// @param priorities The priorities to update with
	virtual void update_priorities(int index, torch::Tensor priorities) = 0;

	/// @brief Gets the priority for this episode.
	/// @return The priority for this episode
	virtual float get_priority() const = 0;

	/// @brief Samples a random step in the episode according to the prioritised distribution.
	/// @param gen The random generator to use when sampling.
	/// @param force_uniform Forces sampling to use a uniform distribution.
	/// @return A pair of the step index and the priority of that step.
	virtual std::pair<int, float> sample_position(std::mt19937& gen, bool force_uniform = false) const = 0;

	/// @brief Generates sequence of target data (actions, rewards, values etc) for an episode at a given index.
	/// @param index Index indicating the position within the episode to start the sequence from.
	/// @param gamma Discount factor.
	/// @return The generated target data.
	virtual EpisodeSampleTargets make_target(int index, torch::Tensor gamma) const = 0;

	/// @brief Updates values when re-analysed
	/// @param values The new values to use
	virtual void update_values(const torch::Tensor& values) = 0;

	/// @brief Updates the hidden states when re-analysed
	/// @param states The new hidden states to use
	virtual void update_states(const HiddenStates& states) = 0;

	/// @brief The length of the episode in agent steps
	/// @return The length of the episode
	virtual int length() const = 0;

	/// @brief The sequence length to use for sampling observations and targets
	/// @param length The sequence length to set
	virtual void set_sequence_length(int length) = 0;

	/// @brief Saves the episode to disk at the specified path in binary format
	/// @param path The path to save the episode to. Thw path must exist.
	virtual void save(const std::filesystem::path& path) = 0;

	/// @brief The path where the episode was saved or loaded from
	/// @return The path, empty if the episode has not been saved or loaded from disk
	virtual const std::filesystem::path& get_path() const = 0;
};

} // namespace drla
