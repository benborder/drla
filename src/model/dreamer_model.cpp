// This is an implementation of the Dreamer v3 model. The following papers were used as reference:
// Dreamerv3 - https://arxiv.org/abs/2301.04104
// Dreamerv2 - https://arxiv.org/abs/2010.02193
// Dreamerv1 - https://arxiv.org/abs/1912.01603
// Many implementation details were not included in the paper, but code was provided by the author:
// https://github.com/danijar/dreamerv3

#include "dreamer_model.h"

#include "bernoulli.h"
#include "categorical.h"
#include "discrete.h"
#include "functions.h"
#include "utils.h"

#include <spdlog/spdlog.h>
#include <torch/serialize.h>

using namespace drla;
using namespace dreamer;

namespace
{
constexpr float keps = 1e-3;
}

inline FCBlock make_linear(std::string name, int input, int output)
{
	Config::FCConfig config;
	config.layers = {
		Config::LinearConfig{output, Config::InitType::kXavierNormal, 1.0, false},
		Config::LayerNormConfig{keps},
		Config::Activation::kSiLU,
	};
	return FCBlock(config, std::move(name), input);
}

inline FCBlock make_dual_linear(std::string name, int input, int hidden, int output)
{
	Config::FCConfig config;
	config.layers = {
		Config::LinearConfig{hidden, Config::InitType::kXavierNormal, 1.0, false},
		Config::LayerNormConfig{keps},
		Config::Activation::kSiLU,
		Config::LinearConfig{output, Config::InitType::kXavierUniform}};
	return FCBlock(config, std::move(name), input);
}

WorldModelImpl::WorldModelImpl(
	const Config::Dreamer::WorldModel& config, const EnvironmentConfiguration& env_config, int reward_size)
		: config_(config)
		, z_size_(config_.class_size * config_.stoch_size)
		, reward_size_(reward_size)
		, encoder_network_(make_multi_encoder(config_.encoder_network, env_config.observation_shapes))
		, decoder_network_(make_multi_decoder(
				config_.decoder_network, encoder_network_->get_output_shape(), env_config.observation_shapes))
		, dec_in_(make_linear("dec_in", z_size_ + config_.deter_state_size, encoder_network_->get_output_size()))
		, seq_in_(make_linear("seq_in", z_size_ + flatten(env_config.action_space.shape), config_.hidden_size))
		, seq_gru_linear_(
				torch::nn::LinearOptions(config_.hidden_size + config_.deter_state_size, 3 * config_.deter_state_size)
					.bias(false))
		, seq_norm_(torch::nn::LayerNormOptions({3 * config_.deter_state_size}).eps(keps))
		, dyn_logits_(make_dual_linear("dyn_logits", config_.deter_state_size, config_.hidden_size, z_size_))
		, fc_rep_(make_dual_linear(
				"enc_fc_rep", encoder_network_->get_output_size() + config.deter_state_size, config_.hidden_size, z_size_))
		, reward_(
				config.reward,
				"reward",
				z_size_ + config.deter_state_size,
				Config::LinearConfig{reward_size_ * config_.bins, Config::InitType::kConstant, 0})
		, continue_(
				config.contin,
				"continue",
				z_size_ + config.deter_state_size,
				Config::LinearConfig{1, Config::InitType::kXavierUniform})
{
	weight_init(seq_gru_linear_->weight, Config::InitType::kXavierNormal, 1.0);
	register_modules();
}

WorldModelImpl::WorldModelImpl(const WorldModelImpl& other, const c10::optional<torch::Device>& device)
		: config_(other.config_)
		, z_size_(other.z_size_)
		, reward_size_(other.reward_size_)
		, encoder_network_(std::dynamic_pointer_cast<FeatureExtractorImpl>(other.encoder_network_->clone(device)))
		, decoder_network_(std::dynamic_pointer_cast<FeatureExtractorImpl>(other.decoder_network_->clone(device)))
		, dec_in_(std::dynamic_pointer_cast<FCBlockImpl>(other.dec_in_->clone(device)))
		, seq_in_(std::dynamic_pointer_cast<FCBlockImpl>(other.seq_in_->clone(device)))
		, seq_gru_linear_(std::dynamic_pointer_cast<torch::nn::LinearImpl>(other.seq_gru_linear_->clone(device)))
		, seq_norm_(std::dynamic_pointer_cast<torch::nn::LayerNormImpl>(other.seq_norm_->clone(device)))
		, dyn_logits_(std::dynamic_pointer_cast<FCBlockImpl>(other.dyn_logits_->clone(device)))
		, fc_rep_(std::dynamic_pointer_cast<FCBlockImpl>(other.fc_rep_->clone(device)))
		, reward_(std::dynamic_pointer_cast<FCBlockImpl>(other.reward_->clone(device)))
		, continue_(std::dynamic_pointer_cast<FCBlockImpl>(other.continue_->clone(device)))
{
	register_modules();
}

void WorldModelImpl::register_modules()
{
	register_module("encoder_network", encoder_network_);
	register_module("enc_fc_rep", fc_rep_);
	register_module("decoder_network", decoder_network_);
	register_module("dec_in", dec_in_);
	register_module("seq_in", seq_in_);
	register_module("seq_gru_linear", seq_gru_linear_);
	register_module("seq_norm", seq_norm_);
	register_module("dyn_logits", dyn_logits_);
	register_module("reward", reward_);
	// A module cannot be named continue as it seems to be a reserved word in libtorch's serialisation format
	register_module("non_terminal", continue_);
}

std::shared_ptr<torch::nn::Module> WorldModelImpl::clone(const c10::optional<torch::Device>& device) const
{
	torch::NoGradGuard no_grad;
	return std::make_shared<WorldModelImpl>(static_cast<const WorldModelImpl&>(*this), device);
}

RSSMState WorldModelImpl::initial_state(int batch_size, torch::Device device)
{
	auto h = torch::zeros({batch_size, config_.deter_state_size}, device);
	auto state = dynamics_predictor(h, false);
	state.h = std::move(h);
	return state;
}

RSSMState WorldModelImpl::forward(const Observations& observations, torch::Tensor action, RSSMState state)
{
	auto embeddings = flatten(embedding(observations));
	auto h = sequence_model(state.h, state.z, action);
	return encoder(embeddings, h);
}

std::tuple<RSSMState, RSSMState>
WorldModelImpl::observe(const Observations& observations, torch::Tensor action, RSSMState state, torch::Tensor is_first)
{
	// Assue dimensions are of the form [B, S, ...]
	auto device = parameters().front().device();
	auto init_state = initial_state(action.size(0), device);
	is_first = is_first.narrow(1, 0, 1).unsqueeze(-1);
	state.h = state.h.narrow(1, 0, 1).squeeze(1);
	state.z = (1.0F - is_first) * state.z.narrow(1, 0, 1).squeeze(1) + is_first * init_state.z;

	auto embeddings = flatten(embedding(observations), 1);

	std::vector<torch::Tensor> state_h;
	std::vector<torch::Tensor> post_z;
	std::vector<torch::Tensor> post_logits;
	std::vector<torch::Tensor> prior_z;
	std::vector<torch::Tensor> prior_logits;

	int depth = action.size(1);
	for (int i = 0; i < depth; ++i)
	{
		auto [post, prior] = observe_step(state, action.narrow(1, i, 1).squeeze(1), embeddings.narrow(1, i, 1).squeeze(1));
		state = post;
		state_h.push_back(std::move(post.h));
		post_z.push_back(std::move(post.z));
		post_logits.push_back(std::move(post.logits));
		prior_z.push_back(std::move(prior.z));
		prior_logits.push_back(std::move(prior.logits));
	}

	static const std::vector<int64_t> zshape = {-1, config_.stoch_size * config_.class_size};
	static const std::vector<int64_t> logitsshape = {-1, config_.stoch_size, config_.class_size};
	RSSMState post;
	post.h = torch::stack(state_h, 1).view({-1, config_.deter_state_size});
	post.z = torch::stack(post_z, 1).view(zshape);
	post.logits = torch::stack(post_logits, 1).view(logitsshape);
	RSSMState prior;
	prior.h = post.h;
	prior.z = torch::stack(prior_z, 1).view(zshape);
	prior.logits = torch::stack(prior_logits, 1).view(logitsshape);

	// dims output as [B*S, ...]
	return {post, prior};
}

std::tuple<Observations, torch::Tensor, torch::Tensor> WorldModelImpl::predict_output(const RSSMState& state)
{
	auto latents = flatten({state.h, state.z});
	auto obs = decoder(latents);
	auto reward_pred = reward_(latents).view({latents.size(0), reward_size_, -1});
	auto continue_pred = continue_(latents);
	return {std::move(obs), std::move(reward_pred), std::move(continue_pred)};
}

std::tuple<RSSMState, RSSMState> WorldModelImpl::observe_step(
	const RSSMState& prev_state, const torch::Tensor& prev_action, const torch::Tensor& embedding)
{
	auto prior_state = imagine_step(prev_state, prev_action);
	auto post_state = encoder(embedding, prior_state.h);

	return {std::move(post_state), std::move(prior_state)};
}

RSSMState WorldModelImpl::imagine_step(const RSSMState& prev_state, const torch::Tensor& prev_action)
{
	auto h = sequence_model(prev_state.h, prev_state.z, prev_action);
	auto state = dynamics_predictor(h);
	state.h = std::move(h);
	return state;
}

std::tuple<torch::Tensor, torch::Tensor> WorldModelImpl::imagined_output(const torch::Tensor& latent)
{
	auto reward_pred = reward_(latent).view({latent.size(0), latent.size(1), reward_size_, -1});
	auto continue_pred = continue_(latent);
	return {std::move(reward_pred), std::move(continue_pred)};
}

torch::Tensor WorldModelImpl::sequence_model(const torch::Tensor& h, const torch::Tensor& z, const torch::Tensor& a)
{
	// a has shape [batch, one_hot_enc_actions]
	// z has shape [batch, stoch_size, class_size]
	// So flatten the final 2 dims of z and concat with a
	auto x = seq_in_(torch::cat({a, z.view({z.size(0), -1})}, -1));
	// pass through normalised GRU cell
	x = torch::cat({x, h}, -1);
	x = seq_gru_linear_(x);
	x = seq_norm_(x);
	auto chunks = x.chunk(3, -1);
	auto reset = torch::sigmoid(chunks[0]);
	auto cand = torch::tanh(reset * chunks[1]);
	auto update = torch::sigmoid(chunks[2] - 1.0F);
	return update * cand + (1 - update) * h;
}

RSSMState WorldModelImpl::encoder(const torch::Tensor& embedding, torch::Tensor& h)
{
	RSSMState state;
	state.h = h;
	auto latent_rep = fc_rep_(torch::cat({embedding.view({embedding.size(0), -1}), h.view({h.size(0), -1})}, -1));
	// Generate the logits and reshape to [batch, stoch_size, class_size]
	std::tie(state.z, state.logits) = encode_stochastic(latent_rep.view({-1, config_.stoch_size, config_.class_size}));
	return state;
}

Observations WorldModelImpl::decoder(const torch::Tensor& latent_state)
{
	return decoder_network_(reconstruct(dec_in_(latent_state), encoder_network_->get_output_shape()));
}

RSSMState WorldModelImpl::dynamics_predictor(const torch::Tensor& h, bool sample)
{
	RSSMState state;
	// Generate the logits and reshape to [batch, stoch_size, class_size]
	auto logits = dyn_logits_(h.view({h.size(0), -1})).view({-1, config_.stoch_size, config_.class_size});
	std::tie(state.z, state.logits) = encode_stochastic(logits, sample);
	return state;
}

std::tuple<torch::Tensor, torch::Tensor> WorldModelImpl::encode_stochastic(torch::Tensor logits, bool sample) const
{
	auto probs = torch::softmax(logits, -1);

	if (config_.unimix > 0)
	{
		auto uniform = torch::ones_like(probs) / config_.class_size;
		probs = (1.0F - config_.unimix) * probs + config_.unimix * uniform;
		logits = probs.log();
	}
	OneHotCategorical dist({}, logits);
	if (sample)
	{
		// Inject the gradient into sample via probs
		return {dist.sample().detach() + probs - probs.detach(), std::move(logits)};
	}
	else
	{
		return {dist.mode(), std::move(logits)};
	}
}

std::vector<torch::Tensor> WorldModelImpl::embedding(const Observations& observations)
{
	Observations obs;
	auto input_shape = decoder_network_->get_output_shape();
	for (size_t i = 0; i < observations.size(); ++i)
	{
		auto shape = std::vector<int64_t>{-1} + input_shape[i];
		auto observation = observations[i].view(shape);
		// Only apply symlog to observations that are not images (assume obs with dimensions > 2 are images)
		if (input_shape[i].size() < 3)
		{
			observation = symlog(observation);
		}
		obs.push_back(observation);
	}
	auto features = encoder_network_(obs);
	for (size_t i = 0; i < features.size(); ++i)
	{
		auto& f = features[i];
		auto obs_shape = observations[i].sizes().vec();
		auto obs_slice = slice(obs_shape, 0, -input_shape[i].size());
		auto fslice = slice<1>(f.sizes().vec());
		auto fshape = obs_slice + fslice;
		f = f.view(fshape);
	}
	return features;
}

DreamerModel::DreamerModel(
	const Config::ModelConfig& config, const EnvironmentConfiguration& env_config, int reward_size, bool predict_values)
		: config_(std::get<Config::Dreamer::ModelConfig>(config))
		, action_space_(env_config.action_space)
		, reward_size_(reward_size)
		, predict_values_(predict_values)
		, feature_size_(
				config_.world_model.deter_state_size + config_.world_model.stoch_size * config_.world_model.class_size)
		, world_model_(config_.world_model, env_config, reward_size_)
		, actor_(config_.actor, feature_size_, env_config.action_space)
		, critic_(
				config_.critic,
				"critic",
				feature_size_,
				Config::LinearConfig{reward_size_ * config_.world_model.bins, Config::InitType::kConstant, 0})
		, critic_target_(
				config_.critic,
				"critic_target",
				feature_size_,
				Config::LinearConfig{reward_size_ * config_.world_model.bins, Config::InitType::kConstant, 0})
{
	register_modules();
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

DreamerModel::DreamerModel(const DreamerModel& other, const c10::optional<torch::Device>& device)
		: config_(other.config_)
		, action_space_(other.action_space_)
		, reward_size_(other.reward_size_)
		, predict_values_(other.predict_values_)
		, feature_size_(other.feature_size_)
		, world_model_(std::dynamic_pointer_cast<WorldModelImpl>(other.world_model_->clone(device)))
		, actor_(std::dynamic_pointer_cast<ActorImpl>(other.actor_->clone(device)))
		, critic_(std::dynamic_pointer_cast<FCBlockImpl>(other.critic_->clone(device)))
		, critic_target_(std::dynamic_pointer_cast<FCBlockImpl>(other.critic_->clone(device)))
{
	register_modules();
}

void DreamerModel::register_modules()
{
	register_module("world_model", world_model_);
	register_module("actor", actor_);
	register_module("critic", critic_);
	register_module("critic_target", critic_target_);
}

std::shared_ptr<torch::nn::Module> DreamerModel::clone(const c10::optional<torch::Device>& device) const
{
	torch::NoGradGuard no_grad;
	return std::make_shared<DreamerModel>(static_cast<const DreamerModel&>(*this), device);
}

ModelOutput DreamerModel::predict(const ModelInput& input)
{
	RSSMState state;
	state.h = input.prev_output.state[0];
	state.z = input.prev_output.state[1];
	state = world_model_(input.observations, encode_actions(input.prev_output.action), state);
	auto latent = flatten({state.h, state.z});

	ModelOutput output;
	auto dist = actor_(latent);
	if (input.deterministic)
	{
		output.action = dist->mode();
	}
	else
	{
		output.action = dist->sample();
	}

	if (is_action_discrete(action_space_))
	{
		output.action.unsqueeze_(-1);
	}
	output.state = {std::move(state.h), std::move(state.z), std::move(state.logits)};
	if (predict_values_)
	{
		output.values = symexp(Discrete(critic_(latent).view({latent.size(0), reward_size_, -1})).mode());
	}

	return output;
}

ModelOutput DreamerModel::initial()
{
	ModelOutput output;
	auto device = actor_->parameters().front().device();
	if (is_action_discrete(action_space_))
	{
		output.action = torch::zeros({static_cast<int>(action_space_.shape.size()), 1}, device);
	}
	else
	{
		output.action = torch::zeros(action_space_.shape, device);
	}
	output.values = torch::zeros({reward_size_}, device);
	auto state = world_model_->initial_state(1, device);
	output.state = {std::move(state.h), std::move(state.z), std::move(state.logits)};
	return output;
}

StateShapes DreamerModel::get_state_shape() const
{
	return {
		{config_.world_model.deter_state_size},
		{config_.world_model.stoch_size, config_.world_model.class_size},
		{config_.world_model.stoch_size, config_.world_model.class_size}};
}

WorldModelOutput DreamerModel::evaluate_world_model(
	const Observations& observations,
	const torch::Tensor& actions,
	const HiddenStates& states,
	const torch::Tensor& is_first)
{
	RSSMState state;
	state.h = states[0];
	state.z = states[1];
	auto [post, prior] = world_model_->observe(observations, encode_actions(actions), state, is_first);

	WorldModelOutput output;
	std::tie(output.observation, output.reward, output.non_terminal) = world_model_->predict_output(post);
	output.values = symexp(Discrete(critic_(flatten({post.h, post.z})).view({post.h.size(0), reward_size_, -1})).mode());
	output.latents = {
		std::move(post.h), // Needed for behavioural model
		std::move(post.z),
		std::move(post.logits), // Needed for KL loss calculation
		std::move(prior.logits)};

	return output;
}

ImaginedTrajectory DreamerModel::imagine_trajectory(int horizon, const WorldModelOutput& initial_states)
{
	using namespace torch::indexing;
	torch::NoGradGuard no_grad;
	std::vector<torch::Tensor> actions;
	std::vector<torch::Tensor> latents;
	RSSMState state;
	state.h = initial_states.latents[0];
	state.z = initial_states.latents[1];
	// imagine latent trajectories
	for (int i = 0; i <= horizon; ++i)
	{
		latents.push_back(flatten({state.h, state.z}));
		auto dist = actor_(latents.back().detach());
		actions.push_back(dist->sample());
		if (is_action_discrete(action_space_))
		{
			actions.back().unsqueeze_(-1);
		}
		if (i < horizon)
		{
			state = world_model_->imagine_step(state, encode_actions(actions.back()));
		}
	}

	ImaginedTrajectory output;
	output.latents = torch::stack(latents);
	std::tie(output.reward, output.non_terminal) = world_model_->imagined_output(output.latents);
	output.reward = symexp(Discrete(output.reward.index({Slice(1)})).mode());
	output.action = torch::stack(actions).squeeze(-1);
	return output;
}

BehaviouralModelOutput
DreamerModel::evaluate_behavioural_model(const torch::Tensor& latents, const torch::Tensor& actions)
{
	using namespace torch::indexing;
	BehaviouralModelOutput output;
	auto dist = actor_(latents.detach());
	output.log_probs = dist->log_prob(actions.detach());
	output.entropy = dist->entropy();
	output.values = critic_(latents.detach());
	auto shape = output.values.sizes().vec();
	shape.back() = reward_size_;
	shape.push_back(-1);
	output.values = output.values.view(shape);
	{
		torch::NoGradGuard no_grad;
		output.target_values = critic_target_(latents).view(shape);
	}
	return output;
}

torch::Tensor DreamerModel::encode_actions(const torch::Tensor& actions)
{
	if (is_action_discrete(action_space_))
	{
		return torch::one_hot(actions.to(torch::kLong), flatten(action_space_.shape)).squeeze(-2);
	}
	return actions;
}

std::vector<torch::Tensor> DreamerModel::world_model_parameters() const
{
	return world_model_->parameters();
}

std::vector<torch::Tensor> DreamerModel::actor_parameters() const
{
	return actor_->parameters();
}

std::vector<torch::Tensor> DreamerModel::critic_parameters() const
{
	return critic_->parameters();
}

void DreamerModel::update(double tau)
{
	torch::NoGradGuard no_grad;
	update_params(critic_->parameters(), critic_target_->parameters(), tau);
}

void DreamerModel::save(const std::filesystem::path& path)
{
	torch::save(std::dynamic_pointer_cast<torch::nn::Module>(shared_from_this()), path / "model.pt");
}

void DreamerModel::load(const std::filesystem::path& path)
{
	auto model_path = path / "model.pt";
	if (std::filesystem::exists(model_path))
	{
		auto model = std::dynamic_pointer_cast<torch::nn::Module>(shared_from_this());
		torch::load(model, model_path);
		spdlog::debug("Dreamer model loaded");
	}
}

void DreamerModel::copy(const Model* model)
{
	if (auto other = dynamic_cast<const DreamerModel*>(model))
	{
		auto params = parameters();
		auto other_params = other->parameters();
		auto device = params.front().device();
		assert(params.size() == other_params.size());
		for (size_t i = 0; i < params.size(); ++i) { params[i] = other_params[i].to(device); }
	}
	else
	{
		spdlog::error("Unable to copy models of different types. Expecting DreamerModel.");
	}
}
