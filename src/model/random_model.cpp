#include "random_model.h"

#include "distribution.h"

using namespace drla;
using namespace torch;

RandomModel::RandomModel(const Config::ModelConfig& config, const ActionSpace& action_space, int value_shape)
		: action_space_(action_space)
		, value_shape_(value_shape)
		, policy_action_output_(register_module(
					"policy_action_output", PolicyActionOutput(Config::PolicyActionOutputConfig(), 1, action_space)))
{
}

PredictOutput RandomModel::predict(const Observations& observations, bool deterministic)
{
	// Create an output distribution to select a uniformly random action
	auto dist = policy_action_output_(torch::ones({1, 1}, observations.front().device()));
	auto action = dist->sample();
	return {torch::zeros({1, value_shape_}), action, {}};
}

void RandomModel::save(const std::filesystem::path& path)
{
}

void RandomModel::load(const std::filesystem::path& path)
{
}
