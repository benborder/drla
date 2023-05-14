#pragma once

#include "actor_net.h"
#include "configuration/model.h"
#include "fc_block.h"
#include "feature_extractor.h"
#include "model.h"
#include "types.h"

#include <torch/torch.h>

#include <memory>
#include <optional>
#include <variant>
#include <vector>

namespace drla
{

namespace dreamer
{

/// @brief The recurrent state-space model state
struct RSSMState
{
	// The deterministic continuous hidden state
	torch::Tensor h;
	// The stochastic discrete representation
	torch::Tensor z;
	// The logits for encoding the stochastic discrete representation z
	torch::Tensor logits;
};

/// @brief The world model learns compact representations of observations and enables planning by predicting future
/// representations via a recurrent state space model (RSSM). The RSSM model is incorporated into the world model for
/// simplicity.
class WorldModelImpl : public torch::nn::Module
{
public:
	WorldModelImpl(
		const Config::Dreamer::WorldModel& config, const EnvironmentConfiguration& env_config, int reward_size);
	WorldModelImpl(const WorldModelImpl& other, const c10::optional<torch::Device>& device);
	std::shared_ptr<torch::nn::Module> clone(const c10::optional<torch::Device>& device = c10::nullopt) const override;

	/// @brief Creates the initial RSSM state
	/// @param batch_size The batch size to create the initial states with
	/// @return The initial RSSM state
	RSSMState initial_state(int batch_size, torch::Device device);

	/// @brief Performs an optimised forward pass for inference
	/// @param observations The observations x_t
	/// @param action The previous action a_(t-1)
	/// @param state The previous state s_(t-1) = {h_(t-1), z_(t-1)}
	/// @return The world model latent state s_t = {h_t, z_t, logits_t}
	RSSMState forward(const Observations& observations, torch::Tensor action, RSSMState state);

	/// @brief Performs a forward pass along a sequence trajectory of observation-actions pairs generating and returning
	/// the internal recurrent states
	/// @param observations The input observations x_t
	/// @param action The previous actions a_(t-1)
	/// @param state The previous state s_(t-1) = {h_(t-1), z_(t-1)}
	/// @return The world model latent states posterior s_t = {h_t, z_t, logits_t} and prior s_t = {h_t, zhat_t, logits_t}
	/// for each input observation-actions pair. The output shapes follows [S * B, ...]
	std::tuple<RSSMState, RSSMState>
	observe(const Observations& observations, torch::Tensor action, RSSMState state, torch::Tensor is_first);

	/// @brief Predicts the output of the world model heads.
	/// @param state The posterior state s_t = {h_t, z_t, logits_t}. Note only h and z is required.
	/// @return observations, reward and non_terminal (continue)
	std::tuple<Observations, torch::Tensor, torch::Tensor> predict_output(const RSSMState& state);

	/// @brief Performs a single step through the world model using h_(t-1) and a_(t-1), predicting the stochastic
	/// representations zhat_t
	/// @param prev_state The previous RSSM state {h_(t-1), z_(t-1)}
	/// @param prev_action The previous action a_(t-1)
	/// @return The prior RSSM state {h_t, zhat_t, logits_t}
	RSSMState imagine_step(const RSSMState& prev_state, const torch::Tensor& prev_action);

	/// @brief Predicts the output of the world model heads, expect the observation head
	/// @param latent The combined {h_t, z_t} latent
	/// @return  reward and non_terminal (continue)
	std::tuple<torch::Tensor, torch::Tensor> imagined_output(const torch::Tensor& latent);

protected:
	/// @brief Registers the world models various network modules.
	void register_modules();

	/// @brief Performs a single step through the world model using h_(t-1), a_(t-1) and the latent embeddings
	/// @param prev_state The previous RSSM state {h_(t-1), z_(t-1)}
	/// @param prev_action The previous action a_(t-1)
	/// @param embedding The latent embeddings from observations x_t
	/// @return The posterior RSSM state {h_t, z_t, logits_t} and prior RSSM state {h_t, zhat_t, logits_t}
	std::tuple<RSSMState, RSSMState>
	observe_step(const RSSMState& prev_state, const torch::Tensor& prev_action, const torch::Tensor& embedding);

	/// @brief Performs a forward pass for the sequence model. Predicts the next recurrent state h_t from the previous
	/// recurrent state h_(t-1), stochastic representations zhat_(t-1) and performed action a_(t-1).
	/// @param h The recurrent hidden state h_(t-1)
	/// @param z The stochastic representations z_(t-1)
	/// @param a The one hot encoded actions a_(t-1)
	/// @return The next recurrent state h_t
	torch::Tensor sequence_model(const torch::Tensor& h, const torch::Tensor& z, const torch::Tensor& a);

	/// @brief Encodes latent embeddings and the deterministic continuous hidden recurrent state h_t to produce the
	/// stochastic discrete hidden state z_t and logits
	/// @param embedding The latent embeddings generated from observations x_t
	/// @param h The deterministic continuous hidden recurrent state h_t
	/// @return z_t and the logits used to encode it
	RSSMState encoder(const torch::Tensor& embedding, torch::Tensor& h);

	/// @brief Decodes the latent state  into predicted observations xhat_t.
	/// @param latent_state The latent state is composed of the recurrent state h_t and stochastic representations z_t
	/// @return xhat_t predicted observations
	Observations decoder(const torch::Tensor& latent_state);

	/// @brief Predicts the stochastic representations zhat_t from the recurrent state h_t.
	/// @param h The deterministic continuous hidden recurrent state h_t
	/// @param sample Set to true to sample from the z distribution, otherwise use the mode of the distribution.
	/// @return zhat_t and the logits used to encode it
	RSSMState dynamics_predictor(const torch::Tensor& h, bool sample = true);

	/// @brief Generates the stochastic discrete representation from logits
	/// @param logits The logits used to generate the stochastic discrete representation
	/// @param sample Set to true to sample from the z distribution, otherwise use the mode of the distribution.
	/// @return The stochastic discrete representation (z) and the logits used to encode it
	std::tuple<torch::Tensor, torch::Tensor> encode_stochastic(torch::Tensor logits, bool sample = true) const;

	/// @brief Creates latent embeddings from the input observations x_t
	/// @param observations The input observations x_t
	/// @return The latent embeddings
	std::vector<torch::Tensor> embedding(const Observations& observations);

private:
	// Configuration for the world model
	const Config::Dreamer::WorldModel config_;
	// Stochastic representation (z) size
	const int z_size_;
	// The number of elements in the reward output
	const int reward_size_;
	// Encoder feature extractor. Transforms observations x_t to an intermediate latent embeddings
	FeatureExtractor encoder_network_;
	// Transforms the latent features into predicted observations xhat_t
	FeatureExtractor decoder_network_;
	// Transforms the recurrent state h_t and stochastic representation z_t to latent features
	FCBlock dec_in_;

	// Takes as input an action and the stochastic representation z_t and outputs to the GRU cell
	FCBlock seq_in_;
	torch::nn::Linear seq_gru_linear_;
	torch::nn::LayerNorm seq_norm_;

	// Takes as input h_t and outputs logits to generate the distribution for z_t
	FCBlock dyn_logits_;
	// Transforms the intermediate latent state and the recurrent state h_t into the stochastic representation z_t
	FCBlock fc_rep_;
	// The reward network head predictor r_t
	FCBlock reward_;
	// The continue network head predictor c_t
	FCBlock continue_;
};

TORCH_MODULE(WorldModel);

} // namespace dreamer

/// @brief Implements the Dreamer V3 model
class DreamerModel : public HybridModelInterface
{
public:
	DreamerModel(
		const Config::ModelConfig& config,
		const EnvironmentConfiguration& env_config,
		int reward_size,
		bool predict_values = false);
	DreamerModel(const DreamerModel& other, const c10::optional<torch::Device>& device);
	std::shared_ptr<torch::nn::Module> clone(const c10::optional<torch::Device>& device = c10::nullopt) const override;

	/// @brief Predict a single step
	/// @param input
	/// @return
	ModelOutput predict(const ModelInput& input) override;

	ModelOutput initial() override;

	StateShapes get_state_shape() const override;

	/// @brief
	/// @param observations
	/// @param actions
	/// @return
	WorldModelOutput evaluate_world_model(
		const Observations& observations,
		const torch::Tensor& actions,
		const HiddenStates& states,
		const torch::Tensor& is_first) override;

	ImaginedTrajectory imagine_trajectory(int horizon, const WorldModelOutput& initial_states) override;

	BehaviouralModelOutput
	evaluate_behavioural_model(const torch::Tensor& latents, const torch::Tensor& actions) override;

	std::vector<torch::Tensor> world_model_parameters() const override;
	std::vector<torch::Tensor> actor_parameters() const override;
	std::vector<torch::Tensor> critic_parameters() const override;

	void update(double tau) override;

	void save(const std::filesystem::path& path) override;
	void load(const std::filesystem::path& path) override;

protected:
	/// @brief Registers the models various network modules.
	void register_modules();
	torch::Tensor encode_actions(const torch::Tensor& actions);

private:
	const Config::Dreamer::ModelConfig config_;
	const ActionSpace action_space_;
	const int reward_size_;
	const bool predict_values_ = false;
	const int feature_size_;

	dreamer::WorldModel world_model_;
	Actor actor_;
	FCBlock critic_;
	FCBlock critic_target_;
};

} // namespace drla
