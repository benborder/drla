#include "mcts_replay_buffer.h"

#include "functions.h"
#include "mcts_episode.h"
#include "utils.h"

using namespace drla;

MCTSReplayBuffer::MCTSReplayBuffer(std::vector<float> gamma, EpisodicPERBufferOptions options)
		: EpisodicPERBuffer(gamma, std::move(options))
{
}

MCTSBatch MCTSReplayBuffer::sample(int batch_size, torch::Device device) const
{
	MCTSBatch batch;

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
	const auto& ep = episodes_.front();
	auto observation_shapes = ep->get_observation_shapes();
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
		auto sample_obs = episode->get_observations(index, device);

		batch.indicies.emplace_back(episode->get_id(), index);
		for (size_t i = 0; i < batch.observation.size(); ++i)
		{
			batch.observation[i][batch_index] = convert_observation(sample_obs[i].detach(), device, false);
		}
		batch.action[batch_index] = target.actions.detach().to(device);
		batch.reward[batch_index] = target.rewards.detach().to(device);
		batch.non_terminal[batch_index] = target.non_terminal.detach().to(device);
		batch.policy[batch_index] = target.policies.detach().to(device);
		batch.values[batch_index] = target.values.detach().to(device);
		batch.weight[batch_index] = 1.0F / (total_steps_ * episode_prob * probs);
		batch.gradient_scale[batch_index].fill_(std::min(options_.unroll_steps, episode->length() - index));
		++batch_index;
	}

	batch.weight.div_(batch.weight.max());

	return batch;
}

void MCTSReplayBuffer::reanalyse(std::shared_ptr<MCTSModelInterface> model)
{
	auto [episode, probs] = sample_episode(/*force_uniform=*/true);
	auto device = model->parameters().front().device(); // use the same device as the model

	torch::NoGradGuard no_grad;

	int len = episode->length();
	ModelInput input;
	auto observation_shapes = episodes_.front()->get_observation_shapes();
	for (auto& obs_shape : observation_shapes)
	{
		input.observations.push_back(torch::empty(std::vector<int64_t>{len} + obs_shape, device));
	}
	for (int step = 0; step < len; ++step)
	{
		auto stacked_obs = episode->get_observations(step, device);
		for (size_t obs_grp = 0; obs_grp < input.observations.size(); ++obs_grp)
		{
			input.observations[obs_grp][step] = stacked_obs[obs_grp];
		}
	}
	auto prediction = model->predict(input);
	episode->update_values(model->support_to_scalar(prediction.values));
	reanalysed_count_ += len;
}

std::shared_ptr<Episode> MCTSReplayBuffer::load_episode(const std::filesystem::path& path)
{
	MCTSEpisodeOptions opt = {path.stem(), static_cast<int>(flatten(options_.action_space.shape))};
	return std::make_shared<MCTSEpisode>(path, opt);
}
