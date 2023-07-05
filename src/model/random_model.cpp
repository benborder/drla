#include "random_model.h"

#include "distribution.h"

using namespace drla;
using namespace torch;

RandomModel::RandomModel(
	[[maybe_unused]] const Config::ModelConfig& config, const ActionSpace& action_space, int value_shape)
		: action_space_(action_space)
		, value_shape_(value_shape)
		, policy_action_output_(register_module(
				"policy_action_output", PolicyActionOutput(Config::PolicyActionOutputConfig(), 1, action_space)))
{
}

RandomModel::RandomModel(const RandomModel& other, const c10::optional<torch::Device>& device)
		: action_space_(other.action_space_)
		, value_shape_(other.value_shape_)
		, policy_action_output_(register_module(
				"policy_action_output",
				std::dynamic_pointer_cast<PolicyActionOutputImpl>(other.policy_action_output_->clone(device))))
{
}

ModelOutput RandomModel::predict(const ModelInput& input)
{
	// Create an output distribution to select a uniformly random action
	auto envs = input.observations.front().size(0);
	auto dist = policy_action_output_(torch::ones({envs, 1}, input.observations.front().device()));
	auto action = dist->sample().unsqueeze(-1);
	return {action, torch::zeros({envs, value_shape_})};
}

StateShapes RandomModel::get_state_shape() const
{
	return {};
}

void RandomModel::save([[maybe_unused]] const std::filesystem::path& path)
{
}

void RandomModel::load([[maybe_unused]] const std::filesystem::path& path)
{
}

std::shared_ptr<torch::nn::Module> RandomModel::clone(const c10::optional<torch::Device>& device) const
{
	torch::NoGradGuard no_grad;
	return std::make_shared<RandomModel>(static_cast<const RandomModel&>(*this), device);
}
