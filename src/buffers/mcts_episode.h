#include "episodic_per_buffer.h"

#include <torch/torch.h>

#include <vector>

namespace drla
{

struct MCTSEpisodeOptions
{
	int num_actions;
	int td_steps = 10;
	int unroll_steps = 10;
	int stack_size = 1;
};

// This is stored on the CPU
class MCTSEpisode final : public Episode
{
public:
	MCTSEpisode(std::vector<StepData> episode_data, MCTSEpisodeOptions options);
	// TODO: add another constructor for loading from file

	void set_id(int id) override;
	int get_id() const override;
	Observations get_observations(int step, torch::Device device) const override;
	ObservationShapes get_observation_shapes() const override;
	StateShapes get_state_shapes() const override;
	void init_priorities(torch::Tensor gamma, float per_alpha = 1.0F) override;
	void update_priorities(int index, torch::Tensor priorities) override;
	float get_priority() const override;
	std::pair<int, float> sample_position(std::mt19937& gen, bool force_uniform = false) const override;
	// The value target is the discounted root value of the search tree td_steps into the future, plus the discounted sum
	// of all rewards until then.
	torch::Tensor compute_target_value(int index, torch::Tensor gamma) const;
	EpisodeSampleTargets make_target(int index, torch::Tensor gamma) const override;
	void update_values(torch::Tensor values) override;
	void update_states(HiddenStates& states) override;
	int length() const override;
	void set_sequence_length(int length) override;

private:
	const MCTSEpisodeOptions options_;
	const int episode_length_;
	int id_ = -1;

	mutable std::mutex m_updates_;

	// The max of priorities_
	float episode_priority_;
	// The priority of each step proportional to the loss from a training step
	std::vector<float> priorities_;

	// The observations
	Observations observations_;
	// The actions performed
	torch::Tensor actions_;
	// The root values
	torch::Tensor values_;
	// The rewards (may be scaled if enabled)
	torch::Tensor rewards_;
	// The policy generated from mcts
	torch::Tensor policy_;
	// Indicates the agents turn
	std::vector<int> turn_index_;
	// The reanalysed root values
	torch::Tensor reanalysed_values_;
};
} // namespace drla
