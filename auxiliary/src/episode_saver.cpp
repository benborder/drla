#include "episode_saver.h"

#include "functions.h"
#include "tensor_storage.h"

#include <spdlog/spdlog.h>

using namespace drla;

void drla::save_episode(const std::vector<StepData>& episode_data, const std::filesystem::path& path)
{
	int64_t episode_length = static_cast<int>(episode_data.size() - 1);
	if (episode_length == 0)
	{
		return;
	}

	torch::NoGradGuard no_grad;

	const auto& initial_step_data = episode_data.at(0);
	std::vector<int64_t> epsz{episode_length + 1};
	Observations observations;
	auto actions = torch::empty(epsz + initial_step_data.predict_result.action[0].sizes().vec());
	auto rewards = torch::empty(epsz + initial_step_data.reward.sizes().vec());
	for (auto& obs : initial_step_data.env_data.observation)
	{
		observations.push_back(torch::empty(epsz + obs.sizes().vec(), obs.scalar_type()));
	}

	size_t obs_dims = initial_step_data.env_data.observation.size();
	size_t state_dims = initial_step_data.predict_result.state.size();
	for (auto& step_data : episode_data)
	{
		for (size_t i = 0; i < obs_dims; ++i) { observations[i][step_data.step] = step_data.env_data.observation[i]; }
		actions[step_data.step] = step_data.predict_result.action[0];
		rewards[step_data.step] = step_data.reward;
	}

	std::filesystem::create_directories(path);

	save_tensor(actions, path / "actions.bin");
	save_tensor(rewards, path / "rewards.bin");
	for (size_t i = 0; i < observations.size(); ++i)
	{
		save_tensor(observations[i], path / (std::string{"observations"} + std::to_string(i) + ".bin"));
	}
}
