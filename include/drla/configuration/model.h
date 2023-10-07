#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <variant>
#include <vector>

namespace drla
{

/// @brief The agent policy model type
enum class AgentPolicyModelType
{
	kRandom,
	kActorCritic,
	kSoftActorCritic,
	kQNet,
	kMuZero,
	kDreamer,
};

/// @brief The type of feature extractor. Only used in deserialisation.
enum class FeatureExtractorType
{
	kMLP,
	kCNN,
};

/// @brier The layer type of a CNN feature extractor
enum class LayerType
{
	kInvalid,
	kConv2d,
	kConvTranspose2d,
	kBatchNorm2d,
	kLayerNorm,
	kMaxPool2d,
	kAvgPool2d,
	kAdaptiveAvgPool2d,
	kResBlock2d,
	kActivation
};

namespace Config
{

/// @brief The activation functions available for fully connected blocks
enum class Activation
{
	kNone,
	kReLU,
	kLeakyReLU,
	kSigmoid,
	kTanh,
	kELU,
	kSiLU,
	kSoftplus,
};

/// @brief The type of fully connected layer
enum class FCLayerType
{
	kLinear,					// Standard linear/fully connected topology.
	kLayerNorm,				// Layer normalisation
	kLayerConnection, // connects the current layer to another specified layer using either a residual connection type or
										// just concatenation
	kActivation,			// The activation type to use
	kLayerRepeat,			// Repeats previous layers
};

/// @brief The initialisation type to use for
enum class InitType
{
	kDefault,
	kConstant,
	kOrthogonal,
	kKaimingUniform,
	kKaimingNormal,
	kXavierUniform,
	kXavierNormal,
};

struct LinearConfig
{
	// The number of neural units in a layer
	int size = 0;
	// The type of initialisation for the weights
	InitType init_weight_type = InitType::kDefault;
	// The weight values to initialise the network with (if relevant)
	double init_weight = 1.0;
	// Enable bias for linear based layers
	bool use_bias = true;
	// The type of initialisation for the bias
	InitType init_bias_type = InitType::kDefault;
	// The bias values to initialise the network with
	double init_bias = 0.0;
};

/// @brief Layer Normalisation configuration for a fully connected network and feature extractor.
struct LayerNormConfig
{
	// The epsilon value added for numerical stability.
	double eps = 1e-5;
};

/// @brief Layer connections configuration
struct LayerConnectionConfig
{
	// The layer index to connect to. This connection cannot connect to a previous layer and is 0 index based.
	int connection;
	// Use residual connections if true, otherwise concatenate the connection. If using residual, the connecting layers
	// must have the same size.
	bool residual = true;
};

/// @brief Repeats preceding layers
struct LayerRepeatConfig
{
	// The number of times to repeat previous layers
	int repeats;
	// The number of layers preceeding this one to repeat. 0 implies all previous layers.
	int layers = 0;
	// The factor to multiply each repeated linear layer by (the result is rounded to the nearest integer based on the
	// resolution)
	double factor = 1;
	// The resolution to use for rounding (i.e. a resolution of 8 will round 44 to 48)
	int resolution = 1;
};

using FCLayerConfig = std::variant<LinearConfig, LayerNormConfig, LayerConnectionConfig, Activation, LayerRepeatConfig>;

/// @brief Fully connected block configuration
struct FCConfig
{
	// Defines each layer in the block. Default to none, passing the original tensor through unmodified.
	std::vector<FCLayerConfig> layers = {};
};

/// @brief Multi Layer Perceptron feature extractor config. Has identical config to the fully connected block config.
struct MLPConfig : FCConfig
{
	// The name of the MLP fully conected block
	std::string name = "mlp";
};

/// @brief Convolutional layer configuration for a feature extractor.
struct Conv2dConfig
{
	// The number of input channels. 0 is used to automatically determine the input channels.
	int in_channels = 0;
	// The number of output channels. The same as number of passes of the kernel window over the input tensor.
	int out_channels;
	// The size of the kernel window
	int kernel_size;
	// The stride of the kernel window
	int stride;
	// The padding, defaults to 0
	int padding = 0;
	// The type of initialisation for the weights
	InitType init_weight_type = InitType::kDefault;
	// The weight to initialise
	float init_weight = std::sqrt(2.0f);
	// The type of initialisation for the bias
	InitType init_bias_type = InitType::kDefault;
	// The bias to initialise
	float init_bias = 0.0f;
	// Use bias for kernel weights
	bool use_bias = true;
};

/// @brief Transpose convolutional layer configuration for a feature extractor.
struct ConvTranspose2dConfig : public Conv2dConfig
{
	// Additional size added to one side of each dimension in the output shape. Defaults to 0
	int output_padding = 0;
};

/// @brief Batch Normalisation layer configuration for a feature extractor.
struct BatchNorm2dConfig
{
	// Whether to learn a scale and bias that are applied in an affine transformation on the input.
	bool affine = true;
	// The epsilon value added for numerical stability.
	double eps = 1e-5;
	// A momentum multiplier for the mean and variance.
	double momentum = 0.1;
	// Whether to store and update batch statistics (mean and variance) in the module.
	bool track_running_stats = true;
};

/// @brief Max Pooling layer configuration for a feature extractor.
struct MaxPool2dConfig
{
	// The size of the kernel window
	int kernel_size;
	// The stride of the kernel window
	int stride;
	// The padding, defaults to 0
	int padding = 0;
};

/// @brief Average Pooling layer configuration for a feature extractor.
struct AvgPool2dConfig
{
	// The size of the kernel window
	int kernel_size;
	// The stride of the kernel window
	int stride;
	// The padding, defaults to 0
	int padding = 0;
};

/// @brief Adaptive Average Pooling layer configuration for a feature extractor.
struct AdaptiveAvgPool2dConfig
{
	// The output height and width
	std::array<int64_t, 2> size;
};

/// @brief Residual Block layer for a feature extractor.
struct ResBlock2dConfig
{
	// The number of convolutional layers
	int layers = 2;
	// The size of the kernel window of the convolutional layers
	int kernel_size = 3;
	// The stride of the kernel window
	int stride = 1;
	// The type of initialisation for the weights
	InitType init_weight_type = InitType::kDefault;
	// The weight to initialise
	float init_weight = std::sqrt(2.0f);
	// Normalise layers
	bool normalise = false;
};

/// @brief The CNN layer config
using CNNLayerConfig = std::variant<
	Conv2dConfig,
	ConvTranspose2dConfig,
	BatchNorm2dConfig,
	LayerNormConfig,
	MaxPool2dConfig,
	AvgPool2dConfig,
	AdaptiveAvgPool2dConfig,
	ResBlock2dConfig,
	Activation>;

/// @brief Convolutional Neural Network feature extractor configuration.
struct CNNConfig
{
	// Defines groups of CNN layers. Each group can have different numbers of input and output channels.
	std::vector<CNNLayerConfig> layers = {};
};

/// @brief A feature extractor group config.
using FeatureExtractorGroup = std::variant<MLPConfig, CNNConfig>;

///@brief Feature extractor configuration.
struct FeatureExtractorConfig
{
	// The feature extractors for each observation group.
	std::vector<FeatureExtractorGroup> feature_groups;
};

/// @brief A CNN and MLP latent feature encoder configuration
struct MultiEncoderNetworkConfig
{
	// The number of mlp layers to use
	int mlp_layers = 2;
	// The size of each layer
	int mlp_units = 512;

	// The number of cnn features
	int cnn_depth = 32;
	// The size of the kernel window
	int kernel_size = 4;
	// The stride of the kernel window
	int stride = 2;
	// The kernel window padding
	int padding = 1;
	// The size of the encoder output (i.e. a n*n output)
	int minres = 4;

	// The activation type
	Activation activations = Activation::kSiLU;
	// Enables layer norm when true
	bool use_layer_norm = true;
	// The epsilon argument for layer norm
	double eps = 1e-5;
	// The type of initialisation for the weights
	InitType init_weight_type = InitType::kXavierNormal;
	// The weight values to initialise the network layers with
	double init_weight = 1.0;
};

struct MultiDecoderNetworkConfig : public MultiEncoderNetworkConfig
{
	// The output padding for ConvTranspose2dConfig
	int output_padding = 0;
	// The final layer output initialisation type for the weights
	InitType init_out_weight_type = InitType::kXavierUniform;
	// The weight values to initialise the final layer with
	double init_out_weight = 1.0;
};

/// @brief The multi encoder config
using MultiEncoderConfig = std::variant<FeatureExtractorConfig, MultiEncoderNetworkConfig>;
/// @brief The multi decoder config
using MultiDecoderConfig = std::variant<FeatureExtractorConfig, MultiDecoderNetworkConfig>;

/// @brief The policy action output configuration. Used to transform input neural units to match the action space of
/// the environment.
struct ActorConfig : public FCConfig
{
	// The type of initialisation for the output layer weights
	InitType init_weight_type = InitType::kDefault;
	// The weight values to initialise the network output layer with
	double init_weight = 0.01;
	// Enable bias for output layer
	bool use_bias = true;
	// The type of initialisation for the output layer bias
	InitType init_bias_type = InitType::kDefault;
	// The bias values to initialise the network output layer with
	double init_bias = 0.0;
	// The amount of uniform probability distribution to mix into the actor distribution
	double unimix = 0.0;
};

/// @brief Configuration for the random model
struct RandomConfig
{
};

/// @brief Configuration for the on policy actor critic model
struct ActorCriticConfig
{
	// Use a shared feature extractor and optional shared fully connected block
	bool use_shared_extractor = true;
	// The feature extractor configuration
	FeatureExtractorConfig feature_extractor;
	// The shared fully connected block configuration
	FCConfig shared = {};
	// The actor config to generate appropriate actions based on the environment
	ActorConfig actor = {};
	// The critic fully connected block configuration
	FCConfig critic = {};
	// Indicates if the a forward pass should be performed with the critic to estimate the return values
	bool predict_values = false;
	// The number of units to use for the GRU cell. 0 Disables the cell.
	int gru_hidden_size = 0;
};

/// @brief Configuration for the of policy Q network model
struct QNetModelConfig
{
	// The feature extractor configuration
	FeatureExtractorConfig feature_extractor;
	// The Q network fully connected block configuration
	FCConfig q_net = {};
	// The number of units to use for the GRU cell. 0 Disables the cell.
	int gru_hidden_size = 0;
};

/// @brief Configuration for the off policy soft actor critic model
struct SoftActorCriticConfig
{
	// Use a shared feature extractor and optional shared fully connected block
	bool shared_feature_extractor = false;
	// The feature extractor configuration
	FeatureExtractorConfig feature_extractor;
	// The actor config to generate appropriate actions based on the environment
	ActorConfig actor = {};
	// The critic fully connected block configuration
	FCConfig critic = {};
	// The critic target fully connected block configuration
	FCConfig critic_target = {};
	// The number of critics to use
	size_t n_critics = 2;
	// Indicates if the a forward pass should be performed with the critic to estimate the return values
	bool predict_values = false;
	// The number of units to use for the GRU cell. 0 Disables the cell.
	int gru_hidden_size = 0;
};

namespace MuZero
{

/// @brief MuZero dynamics network configuration
struct DynamicsNetwork
{
	// The number of residual blocks
	int num_blocks = 16;
	// The number of channels for the residual blocks
	int num_channels = 256;
	// The number of channels in the reward head
	int reduced_channels_reward = 256;
	// The config for the residual blocks
	ResBlock2dConfig resblock;
	// The reward fully connected layers
	FCConfig fc_reward =
		FCConfig{std::vector<FCLayerConfig>{Config::LinearConfig{256}, Activation::kReLU, Config::LinearConfig{256}}};
	// The dynamics net fully connected layers
	FCConfig fc_dynamics;
};

/// @brief MuZero prediction network configuration
struct PredictionNetwork
{
	// The number of residual blocks
	int num_blocks = 16;
	// The number of channels for the residual blocks
	int num_channels = 256;
	// The number of channels in the value head
	int reduced_channels_value = 256;
	// The number of channels in the policy head
	int reduced_channels_policy = 256;
	// The config for the residual blocks
	ResBlock2dConfig resblock;
	// The value fully connected layers
	FCConfig fc_value =
		FCConfig{std::vector<FCLayerConfig>{Config::LinearConfig{256}, Activation::kReLU, Config::LinearConfig{256}}};
	// The policy fully connected layers
	FCConfig fc_policy =
		FCConfig{std::vector<FCLayerConfig>{Config::LinearConfig{256}, Activation::kReLU, Config::LinearConfig{256}}};
};

/// @brief Configuration for the muzero model
struct ModelConfig
{
	// The feature extractor configuration
	FeatureExtractorConfig representation_network;
	// The dynamics network configuration
	DynamicsNetwork dynamics_network;
	// The prediction network configuration
	PredictionNetwork prediction_network;

	// Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to
	// support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))
	int support_size = 300;
	// The number of previous observations to present to the model as input in addition to the most recent observation
	int stacked_observations = 0;
};

} // namespace MuZero

namespace Dreamer
{

/// @brief The world model configuration for the dreamer model
struct WorldModel
{
	// The encoder network configuration. Inputs observations x_t, oututs stochastic representation z_t.
	MultiEncoderConfig encoder_network;
	// The decoder network configuration. Inputs recurrent state h_t and stochastic representation z_t/zhat_t, outputs
	// predicted observations xhat_t.
	MultiDecoderConfig decoder_network;

	// How much to mix a uniform probability with the stochastic probability.
	float unimix = 0.01F;

	// The size of the input passed into the GRU
	int hidden_size = 512;

	// The deterministic continuous hidden state h size
	int deter_state_size = 512;
	// The size of stochastic representations in z
	int stoch_size = 32;
	// The size of each stochastic representation in z
	int class_size = 32;

	// The number of bins to encode reward/values into
	int bins = 255;

	// The reward network
	FCConfig reward = {};
	// The continue network
	FCConfig contin = {};
};

/// @brief Configuration for the dreamer model
struct ModelConfig
{
	// The world model config
	WorldModel world_model;
	// The actor config to generate appropriate actions based on the environment
	ActorConfig actor = {};
	// The critic fully connected block configuration
	FCConfig critic = {};
};

} // namespace Dreamer

/// @brief The model configuration for the agent.
using ModelConfig = std::variant<
	RandomConfig,
	ActorCriticConfig,
	QNetModelConfig,
	SoftActorCriticConfig,
	MuZero::ModelConfig,
	Dreamer::ModelConfig>;

} // namespace Config

} // namespace drla
