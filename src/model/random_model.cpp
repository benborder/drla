#include "random_model.h"

#include "distribution.h"

using namespace drla;
using namespace torch;

RandomModel::RandomModel(
	[[maybe_unused]] const Config::ModelConfig& config, const ActionSpace& action_space, int value_shape)
		: action_space_(action_space)
		, value_shape_(value_shape)
		, actor_(
				register_module("actor", Actor(Config::ActorConfig{{}, Config::InitType::kConstant, 1.0}, 1, action_space)))
{
}

RandomModel::RandomModel(const RandomModel& other, const c10::optional<torch::Device>& device)
		: action_space_(other.action_space_)
		, value_shape_(other.value_shape_)
		, actor_(register_module("actor", std::dynamic_pointer_cast<ActorImpl>(other.actor_->clone(device))))
{
}

ModelOutput RandomModel::predict(const ModelInput& input)
{
	// Create an output distribution to select a uniformly random action
	auto envs = input.observations.front().size(0);
	auto dist = actor_(torch::ones({envs, 1}, input.observations.front().device()));
	auto action = dist->sample().unsqueeze(-1);
	return {action, torch::zeros({envs, value_shape_})};
}

ModelOutput RandomModel::initial() const
{
	ModelOutput output;
	if (is_action_discrete(action_space_))
	{
		output.action = torch::zeros(static_cast<int>(action_space_.shape.size()));
	}
	else
	{
		output.action = torch::zeros(action_space_.shape);
	}
	auto device = actor_->parameters().front().device();
	output.values = torch::zeros(value_shape_, device);
	return output;
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
