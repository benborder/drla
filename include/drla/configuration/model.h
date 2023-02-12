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
};

/// @brief The type of feature extractor. Only used in deserialisation.
enum class FeatureExtractorType
{
	kMLP,
	kCNN,
};

/// @brier The layer type of a feature extractor
enum class LayerType
{
	kConv2d,
	kBatchNorm2d,
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
	kSoftplus,
};

/// @brief The type of fully connected layer
enum class FCLayerType
{
	kLinear,				 // Standard linear/fully connected topology.
	kInputConnected, // Same as linear, but input is also connected to this layer.
	kMultiConnected, // Same as linear, but input and all previous layers are connected to this layers input.
	kResidual,			 // Adds the input to the output of this layer. The output size will be the same as the input.
	kForwardInput,	 // Forwards the input and the final layer to the output
	kForwardAll,		 // Forwards the input and all layers to the output
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

/// @brief Fully connected block configuration
struct FCConfig
{
	struct fc_layer
	{
		// The number of neural units in a layer
		int size = 0;
		// The activation function used for forward passes
		Activation activation = Activation::kNone;
		// The type of initialisation for the weights
		InitType init_weight_type = InitType::kDefault;
		// The weight values to initialise the network with (if relevant)
		double init_weight = 1.0;
		// The type of initialisation for the bias
		InitType init_bias_type = InitType::kDefault;
		// The bias values to initialise the network with
		double init_bias = 0.0;
		// The type of fully connected network
		FCLayerType type = FCLayerType::kLinear;
	};

	// Defines each layer in the block. Default to none, passing the original tensor through unmodified.
	std::vector<fc_layer> layers = {};
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
	BatchNorm2dConfig,
	MaxPool2dConfig,
	AvgPool2dConfig,
	AdaptiveAvgPool2dConfig,
	ResBlock2dConfig,
	Activation>;

/// @brief Convolutional Neural Network feature extractor configuration.
struct CNNConfig
{
	// Defines groups of CNN layers. Each group can have different numbers of input and output channels.
	std::vector<CNNLayerConfig> layers = {
		Conv2dConfig{0, 32, 8, 4}, Conv2dConfig{0, 64, 4, 2}, Conv2dConfig{0, 64, 3, 1}};
};

/// @brief A feature extractor group config.
using FeatureExtractorGroup = std::variant<MLPConfig, CNNConfig>;

///@brief Feature extractor configuration.
struct FeatureExtractorConfig
{
	// The feature extractors for each observation group.
	std::vector<FeatureExtractorGroup> feature_groups;
};

/// @brief The policy action output configuration. Used to transform input neural units to match the action space of
/// the environment.
struct PolicyActionOutputConfig
{
	// The activations function to use for the action head
	Activation activation = Activation::kNone;
	// The type of initialisation for the weights
	InitType init_weight_type = InitType::kDefault;
	// The weight values to initialise the network with
	double init_weight = 0.01;
	// The type of initialisation for the bias
	InitType init_bias_type = InitType::kDefault;
	// The bias values to initialise the network with
	double init_bias = 0.0;
};

/// @brief Configuration common to all models
struct CommonModelConfig
{
};

/// @brief Configuration for the random model
struct RandomConfig : public CommonModelConfig
{
};

/// @brief Configuration for the on policy actor critic model
struct ActorCriticConfig : public CommonModelConfig
{
	// Use a shared feature extractor and optional shared fully connected block
	bool use_shared_extractor = true;
	// The feature extractor configuration
	FeatureExtractorConfig feature_extractor;
	// The shared fully connected block configuration
	FCConfig shared = {{{512, Activation::kReLU}}};
	// The actor fully connected block configuration
	FCConfig actor = {};
	// The critic fully connected block configuration
	FCConfig critic = {};
	// The policy action output to convert the actor output to the environment action space.
	PolicyActionOutputConfig policy_action_output;
	// Indicates if the a forward pass should be performed with the critic to estimate the return values
	bool predict_values = false;
};

/// @brief Configuration for the of policy Q network model
struct QNetModelConfig : public CommonModelConfig
{
	// The feature extractor configuration
	FeatureExtractorConfig feature_extractor;
	// The Q network fully connected block configuration
	FCConfig q_net = {};
};

/// @brief Configuration for the off policy soft actor critic model
struct SoftActorCriticConfig : public CommonModelConfig
{
	// Use a shared feature extractor and optional shared fully connected block
	bool shared_feature_extractor = false;
	// The feature extractor configuration
	FeatureExtractorConfig feature_extractor;
	// The policy action output to convert the actor output to the environment action space.
	PolicyActionOutputConfig policy_action_output;
	// The actor fully connected block configuration
	FCConfig actor = {};
	// The critic fully connected block configuration
	FCConfig critic = {};
	// The critic target fully connected block configuration
	FCConfig critic_target = {};
	// The number of critics to use
	size_t n_critics = 2;
	// Indicates if the a forward pass should be performed with the critic to estimate the return values
	bool predict_values = false;
};

/// @brief The model configuration for the agent.
using ModelConfig = std::variant<RandomConfig, ActorCriticConfig, QNetModelConfig, SoftActorCriticConfig>;

} // namespace Config

} // namespace drla
