#include "muzero_model.h"

#include "distribution.h"
#include "utils.h"

#include <spdlog/spdlog.h>

#include <filesystem>

using namespace drla;
using namespace torch;

DynamicsNetworkImpl::DynamicsNetworkImpl(
	const Config::DynamicsNetworkConfig& config,
	const std::vector<std::vector<int64_t>>& input_shape,
	const ActionSpace& action_space,
	int reward_shape)
		: config_(config), fc_reward_(config_.fc_reward, "reward", flatten(input_shape), reward_shape)
{
	int group = 0;
	auto action_size = flatten(action_space.shape);
	for (auto& shape : input_shape)
	{
		auto postfix = std::to_string(group);
		auto size = shape.size();
		if (size == 3)
		{
			Dynamics2D dyn{
				torch::nn::Conv2d(
					torch::nn::Conv2dOptions(config_.num_channels + 1, config_.num_channels, 3).stride(1).padding(1).bias(false)),
				torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(config_.num_channels)),
				torch::nn::Conv2d(torch::nn::Conv2dOptions(config_.num_channels, config_.reduced_channels_reward, 1)),
				{}};
			register_module("dyn_conv_" + postfix, dyn.conv_);
			register_module("dyn_bn_" + postfix, dyn.bn_);
			register_module("dyn_conv1x1_reward_" + postfix, dyn.conv1x1_reward_);
			for (int i = 0; i < config_.num_blocks; ++i)
			{
				dyn.resblocks_.emplace_back(config_.num_channels, config_.resblock);
				register_module("dyn_resblocks_" + postfix + std::to_string(i), dyn.resblocks_.back());
			}

			dynamics_encoding_.emplace_back(std::move(dyn));
		}
		else if (size <= 2)
		{
			auto encoding_shape = flatten(shape);
			dynamics_encoding_.emplace_back(
				FCBlock(config_.fc_dynamics, "fc_dyn", encoding_shape + action_size, encoding_shape));
			register_module("dyn_fcblock_" + postfix, std::get<FCBlock>(dynamics_encoding_.back()));
		}
		else
		{
			spdlog::error("Invalid representation network output shape: [{}]", fmt::join(shape, ", "));
			throw std::runtime_error("Invalid representation network output shape");
		}
		++group;
	}
	register_module("dyn_fc_reward", fc_reward_);
}

DynamicsNetworkImpl::DynamicsNetworkImpl(const DynamicsNetworkImpl& other, const c10::optional<torch::Device>& device)
		: config_(other.config_), fc_reward_(std::dynamic_pointer_cast<FCBlock::Impl>(other.fc_reward_->clone(device)))
{
	int index = 0;
	for (auto& dynamics_encoding : other.dynamics_encoding_)
	{
		std::visit(
			[&](auto& dyn_enc) {
				using dyn_type = std::decay_t<decltype(dyn_enc)>;
				if constexpr (std::is_same_v<Dynamics2D, dyn_type>)
				{
					Dynamics2D dyn{
						std::dynamic_pointer_cast<torch::nn::Conv2d::Impl>(dyn_enc.conv_->clone(device)),
						std::dynamic_pointer_cast<torch::nn::BatchNorm2d::Impl>(dyn_enc.bn_->clone(device)),
						std::dynamic_pointer_cast<torch::nn::Conv2d::Impl>(dyn_enc.conv1x1_reward_->clone(device)),
						{}};
					register_module(other.named_children()[index++].key(), dyn.conv_);
					register_module(other.named_children()[index++].key(), dyn.bn_);
					register_module(other.named_children()[index++].key(), dyn.conv1x1_reward_);
					for (auto& resblock : dyn_enc.resblocks_)
					{
						auto new_resblock = std::dynamic_pointer_cast<ResBlock2d::Impl>(resblock->clone(device));
						register_module(other.named_children()[index++].key(), new_resblock);
						dyn.resblocks_.emplace_back(std::move(new_resblock));
					}
					dynamics_encoding_.emplace_back(std::move(dyn));
				}
				else
				{
					auto fc_block = std::dynamic_pointer_cast<FCBlock::Impl>(dyn_enc->clone(device));
					register_module(other.named_children()[index++].key(), fc_block);
					dynamics_encoding_.emplace_back(std::move(fc_block));
				}
			},
			dynamics_encoding);
	}
	register_module("dyn_fc_reward", fc_reward_);
}

std::pair<std::vector<torch::Tensor>, torch::Tensor>
DynamicsNetworkImpl::forward(const std::vector<torch::Tensor>& next_state)
{
	std::vector<torch::Tensor> state;
	std::vector<torch::Tensor> reward;
	for (size_t i = 0; i < next_state.size(); ++i)
	{
		auto x = next_state[i];
		auto& dyn_enc = dynamics_encoding_[i];
		// The tensor dims must be of the format [batch, channels, height, width] to use resblock/convolutional networks
		if (std::holds_alternative<Dynamics2D>(dyn_enc))
		{
			auto& dyn = std::get<Dynamics2D>(dyn_enc);
			x = torch::relu(dyn.bn_(dyn.conv_(x)));
			for (auto& resblock : dyn.resblocks_) { x = resblock(x); }
			state.push_back(normalise(x));
			x = dyn.conv1x1_reward_(x);
			reward.push_back(x);
		}
		// data based tensor dims are assumed to be of the format [batch, 1, data] and appended in the data dim
		else
		{
			x = std::get<FCBlock>(dyn_enc)(x);
			state.push_back(normalise(x));
			reward.push_back(x);
		}
	}
	return {std::move(state), fc_reward_(flatten(reward))};
}

std::shared_ptr<torch::nn::Module> DynamicsNetworkImpl::clone(const c10::optional<torch::Device>& device) const
{
	torch::NoGradGuard no_grad;
	return std::make_shared<DynamicsNetworkImpl>(static_cast<const DynamicsNetworkImpl&>(*this), device);
}

PredictionNetworkImpl::PredictionNetworkImpl(
	const Config::PredictionNetworkConfig& config,
	const std::vector<std::vector<int64_t>>& input_shape,
	const ActionSpace& action_space,
	int value_shape)
		: config_(config)
		, fc_value_(config_.fc_value, "value", flatten(input_shape), value_shape)
		, fc_policy_(config_.fc_policy, "policy", flatten(input_shape), flatten(action_space.shape))
{
	int group = 0;
	for (auto& shape : input_shape)
	{
		auto postfix = std::to_string(group);
		auto size = shape.size();
		if (size == 3)
		{
			Prediction2D pred{
				{},
				torch::nn::Conv2d(torch::nn::Conv2dOptions(config_.num_channels, config_.reduced_channels_value, 1)),
				torch::nn::Conv2d(torch::nn::Conv2dOptions(config_.num_channels, config_.reduced_channels_policy, 1))};
			register_module("pred_conv1x1_value_" + postfix, pred.conv1x1_value_);
			register_module("pred_conv1x1_policy_" + postfix, pred.conv1x1_policy_);

			for (int i = 0; i < config_.num_blocks; ++i)
			{
				pred.resblocks_.emplace_back(config_.num_channels, config_.resblock);
				register_module("pred_resblocks_" + postfix + std::to_string(i), pred.resblocks_.back());
			}

			prediction_encoding_.emplace_back(std::move(pred));
		}
		else
		{
			prediction_encoding_.emplace_back(std::monostate());
		}
		++group;
	}

	register_module("pred_fc_value", fc_value_);
	register_module("pred_fc_policy", fc_policy_);
}

PredictionNetworkImpl::PredictionNetworkImpl(
	const PredictionNetworkImpl& other, const c10::optional<torch::Device>& device)
		: config_(other.config_)
		, fc_value_(std::dynamic_pointer_cast<FCBlock::Impl>(other.fc_value_->clone(device)))
		, fc_policy_(std::dynamic_pointer_cast<FCBlock::Impl>(other.fc_policy_->clone(device)))
{
	int index = 0;
	for (auto& prediction_encoding : other.prediction_encoding_)
	{
		std::visit(
			[&](auto& pred_enc) {
				using pred_type = std::decay_t<decltype(pred_enc)>;
				if constexpr (std::is_same_v<Prediction2D, pred_type>)
				{
					Prediction2D pred{
						{},
						std::dynamic_pointer_cast<torch::nn::Conv2d::Impl>(pred_enc.conv1x1_value_->clone(device)),
						std::dynamic_pointer_cast<torch::nn::Conv2d::Impl>(pred_enc.conv1x1_policy_->clone(device)),
					};
					register_module(other.named_children()[index++].key(), pred.conv1x1_value_);
					register_module(other.named_children()[index++].key(), pred.conv1x1_policy_);
					for (auto& resblock : pred_enc.resblocks_)
					{
						auto new_resblock = std::dynamic_pointer_cast<ResBlock2d::Impl>(resblock->clone(device));
						register_module(other.named_children()[index++].key(), new_resblock);
						pred.resblocks_.emplace_back(std::move(new_resblock));
					}
					prediction_encoding_.emplace_back(std::move(pred));
				}
				else
				{
					prediction_encoding_.emplace_back(std::monostate());
				}
			},
			prediction_encoding);
	}

	register_module("pred_fc_value", fc_value_);
	register_module("pred_fc_policy", fc_policy_);
}

std::pair<torch::Tensor, torch::Tensor> PredictionNetworkImpl::forward(const std::vector<torch::Tensor>& input)
{
	std::vector<torch::Tensor> value;
	std::vector<torch::Tensor> policy;
	for (size_t i = 0; i < input.size(); ++i)
	{
		auto x = input[i];
		auto& pred_enc = prediction_encoding_[i];
		// The tensor dims must be of the format [batch, channels, height, width] to use resblock/convolutional networks
		if (std::holds_alternative<Prediction2D>(pred_enc))
		{
			auto& pred = std::get<Prediction2D>(pred_enc);
			for (auto& resblock : pred.resblocks_) { x = resblock(x); }
			value.push_back(pred.conv1x1_value_(x));
			policy.push_back(pred.conv1x1_policy_(x));
		}
		else
		{
			value.push_back(x);
			policy.push_back(x);
		}
	}

	return {fc_policy_(flatten(policy)), fc_value_(flatten(value))};
}

std::shared_ptr<torch::nn::Module> PredictionNetworkImpl::clone(const c10::optional<torch::Device>& device) const
{
	torch::NoGradGuard no_grad;
	return std::make_shared<PredictionNetworkImpl>(static_cast<const PredictionNetworkImpl&>(*this), device);
}

// muzero can only support one action at a time, hence action_space_.shape can only be 1 dimensional
MuZeroModel::MuZeroModel(
	const Config::ModelConfig& config, const EnvironmentConfiguration& env_config, int reward_shape)
		: config_(std::get<Config::MuZeroModelConfig>(config))
		, action_space_size_(flatten(env_config.action_space.shape))
		, reward_shape_(reward_shape)
		, representation_network_(
				config_.representation_network,
				stacked_observation_shape(env_config.observation_shapes, config_.stacked_observations))
		, dynamics_network_(
				config_.dynamics_network,
				condense_shape(representation_network_->get_output_shape()),
				env_config.action_space,
				reward_shape * (2 * config_.support_size + 1))
		, prediction_network_(
				config_.prediction_network,
				condense_shape(representation_network_->get_output_shape()),
				env_config.action_space,
				reward_shape * (2 * config_.support_size + 1))
{
	register_module("representation_network", representation_network_);
	register_module("dynamics_network", dynamics_network_);
	register_module("prediction_network", prediction_network_);

	int parameter_size = 0;
	auto params = parameters();
	for (auto& p : params)
	{
		if (p.requires_grad())
		{
			parameter_size += p.numel();
		}
	}
	spdlog::debug("Total parameters: {}", parameter_size);
}

MuZeroModel::MuZeroModel(const MuZeroModel& other, const c10::optional<torch::Device>& device)
		: config_(other.config_)
		, action_space_size_(other.action_space_size_)
		, reward_shape_(other.reward_shape_)
		, representation_network_(nullptr)
		, dynamics_network_(nullptr)
		, prediction_network_(nullptr)
{
	representation_network_ =
		std::dynamic_pointer_cast<FeatureExtractorImpl>(other.representation_network_->clone(device));
	register_module("representation_network", representation_network_);

	dynamics_network_ = std::dynamic_pointer_cast<DynamicsNetworkImpl>(other.dynamics_network_->clone(device));
	register_module("dynamics_network", dynamics_network_);

	prediction_network_ = std::dynamic_pointer_cast<PredictionNetworkImpl>(other.prediction_network_->clone(device));
	register_module("prediction_network", prediction_network_);
}

PredictOutput MuZeroModel::predict(const Observations& observations, [[maybe_unused]] bool deterministic)
{
	PredictOutput output;

	output.state = condense(representation_network_(observations));
	for (auto& state : output.state) { state = normalise(state); }
	std::tie(output.policy, output.values) = prediction_network_(output.state);
	output.reward = scalar_to_support(torch::zeros({output.values.size(0), 1, reward_shape_}, output.values.device()));

	return output;
}

PredictOutput MuZeroModel::predict(const PredictOutput& previous_output, [[maybe_unused]] bool deterministic)
{
	PredictOutput output;
	std::vector<torch::Tensor> previous_state;

	for (auto& state : previous_output.state)
	{
		torch::Tensor action_one_hot;
		// The tensor dims must be of the format [batch, channels, height, width] to use resblock/convolutional networks
		if (state.dim() == 4)
		{
			action_one_hot = torch::ones({state.size(0), 1, state.size(2), state.size(3)}, previous_output.action.device());
			action_one_hot = previous_output.action.view({-1, 1, 1, 1}) * action_one_hot / action_space_size_;
		}
		// data based tensor dims are assumed to be of the format [batch, data] and appended in the data dim
		else
		{
			action_one_hot = torch::zeros({state.size(0), action_space_size_}, state.device()).to(torch::kFloat);
			action_one_hot.scatter_(1, previous_output.action.to(torch::kLong), 1.0F);
		}
		previous_state.push_back(torch::cat({normalise(state), action_one_hot}, 1));
	}

	std::tie(output.state, output.reward) = dynamics_network_(previous_state);
	std::tie(output.policy, output.values) = prediction_network_(output.state);

	return output;
}

namespace
{
constexpr float kepsilon = 0.001F;
}

torch::Tensor MuZeroModel::support_to_scalar(torch::Tensor logits)
{
	auto shape = logits.sizes().vec();
	shape.insert(std::prev(shape.end()), reward_shape_);
	shape.back() /= reward_shape_;
	logits = logits.view(shape);
	auto probs = torch::softmax(logits, -1);
	auto support =
		torch::range(-config_.support_size, config_.support_size, probs.device()).expand(probs.sizes()).to(torch::kFloat);
	auto x = (support * probs).sum(-1);
	x =
		torch::sign(x) *
		(torch::pow((torch::sqrt(1.0F + 4.0F * kepsilon * (x.abs() + 1.0F + kepsilon)) - 1.0F) / (2.0F * kepsilon), 2.0F) -
		 1.0F);
	return x;
}

torch::Tensor MuZeroModel::scalar_to_support(torch::Tensor x)
{
	x.unsqueeze_(-1);
	x = torch::sign(x) * (torch::sqrt(x.abs() + 1.0F) - 1.0F) + kepsilon * x;
	x.clamp_(-config_.support_size, config_.support_size);
	auto floor = x.floor();
	auto prob = x - floor;
	auto logits = torch::zeros({x.size(0), x.size(1), reward_shape_, 2 * config_.support_size + 1}, x.device());
	auto indexes = floor + config_.support_size;
	logits.scatter_(3, indexes.to(torch::kLong), (1.0F - prob));
	indexes.add_(1);
	prob.masked_fill_(2 * config_.support_size < indexes, 0.0F);
	indexes.masked_fill_(2 * config_.support_size < indexes, 0.0F);
	logits.scatter_(3, indexes.to(torch::kLong), prob);
	return logits.view({x.size(0), x.size(1), -1});
}

int MuZeroModel::get_stacked_observation_size() const
{
	return config_.stacked_observations;
}

void MuZeroModel::save(const std::filesystem::path& path)
{
	torch::save(std::dynamic_pointer_cast<torch::nn::Module>(shared_from_this()), path / "model.pt");
}

void MuZeroModel::load(const std::filesystem::path& path)
{
	auto model_path = path / "model.pt";
	if (std::filesystem::exists(model_path))
	{
		auto model = std::dynamic_pointer_cast<torch::nn::Module>(shared_from_this());
		torch::load(model, model_path);
		spdlog::debug("MuZero model loaded");
	}
}

std::shared_ptr<torch::nn::Module> MuZeroModel::clone(const c10::optional<torch::Device>& device) const
{
	torch::NoGradGuard no_grad;
	return std::make_shared<MuZeroModel>(static_cast<const MuZeroModel&>(*this), device);
}
