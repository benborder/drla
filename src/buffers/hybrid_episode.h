#include "episode.h"

#include <ATen/core/Tensor.h>

#include <filesystem>
#include <string>
#include <vector>

namespace drla
{

struct HybridEpisodeOptions
{
	std::string name;
	int num_actions;
	int unroll_steps = 10;
};

// This is stored on the CPU
class HybridEpisode final : public Episode
{
public:
	HybridEpisode(std::vector<StepData> episode_data, HybridEpisodeOptions options);
	HybridEpisode(const std::filesystem::path& path, const StateShapes& state_shapes, HybridEpisodeOptions options);

	void add_step(StepData&& data);

	void set_id(int id) override;
	int get_id() const override;
	Observations get_observations(int step, torch::Device device = torch::kCPU) const override;
	Observations get_observation(int step, torch::Device device = torch::kCPU) const;
	torch::Tensor get_action(int step) const;
	ObservationShapes get_observation_shapes() const override;
	StateShapes get_state_shapes() const override;
	void init_priorities(torch::Tensor gamma, float per_alpha = 1.0F) override;
	void update_priorities(int index, torch::Tensor priorities) override;
	float get_priority() const override;
	std::pair<int, float> sample_position(std::mt19937& gen, bool force_uniform = false) const override;
	torch::Tensor compute_target_value(int index, torch::Tensor gamma) const;
	EpisodeSampleTargets make_target(int index, torch::Tensor gamma) const override;
	void update_values(torch::Tensor values) override;
	void update_states(HiddenStates& states) override;
	int length() const override;
	void set_sequence_length(int length) override;
	void save(const std::filesystem::path& path) override;
	const std::filesystem::path& get_path() const override;

private:
	void allocate_reserve(torch::Tensor& x);

	const HybridEpisodeOptions options_;
	int episode_length_;
	int sequence_length_;
	int id_ = -1;
	bool is_terminal_ = false;
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
	// The rewards (may be scaled if enabled)
	torch::Tensor rewards_;
	// The values
	torch::Tensor values_;
	// The hidden states from a recurrent model
	HiddenStates states_;
};

} // namespace drla
