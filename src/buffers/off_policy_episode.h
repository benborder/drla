#include "episode.h"

#include <ATen/core/Tensor.h>

#include <filesystem>
#include <vector>

namespace drla
{

struct OffPolicyEpisodeOptions
{
	std::string name;
	int num_actions;
};

// This is stored on the CPU
class OffPolicyEpisode final : public Episode
{
public:
	OffPolicyEpisode(std::vector<StepData> episode_data, OffPolicyEpisodeOptions options);
	OffPolicyEpisode(const std::filesystem::path& path, const StateShapes& state_shapes, OffPolicyEpisodeOptions options);

	void set_id(int id) override;
	int get_id() const override;
	Observations get_observations(int step, torch::Device device) const override;
	ObservationShapes get_observation_shapes() const override;
	StateShapes get_state_shapes() const override;
	void init_priorities(torch::Tensor gamma, float per_alpha = 1.0F) override;
	void update_priorities(int index, torch::Tensor priorities) override;
	float get_priority() const override;
	std::pair<int, float> sample_position(std::mt19937& gen, bool force_uniform = false) const override;
	EpisodeSampleTargets make_target(int index, torch::Tensor gamma) const override;
	void update_values(const torch::Tensor& values) override;
	void update_states(const HiddenStates& states) override;
	int length() const override;
	void set_sequence_length(int length) override;
	void save(const std::filesystem::path& path) override;
	const std::filesystem::path& get_path() const override;

private:
	const OffPolicyEpisodeOptions options_;
	int episode_length_;
	int id_ = -1;
	std::filesystem::path saved_path_;

	mutable std::mutex m_updates_;

	// The max of priorities_
	float episode_priority_ = 0;
	// The priority of each step proportional to the loss from a training step
	std::vector<float> priorities_;

	// The observations
	Observations observations_;
	// The actions performed
	torch::Tensor actions_;
	// The predicted values
	torch::Tensor values_;
	// The rewards (may be scaled if enabled)
	torch::Tensor rewards_;
	// The reanalysed values
	torch::Tensor reanalysed_values_;
	// The hidden state for recurrent models
	HiddenStates state_;
	// The reanalysed hidden state for recurrent models
	HiddenStates reanalysed_state_;
};
} // namespace drla
