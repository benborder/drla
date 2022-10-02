#pragma once

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
	kQNet,
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

/// @brief Fully connected block configuration
struct FCConfig
{
	struct fc_layer
	{
		// The number of neural units in a layer
		int size = 0;
		// The activation function used for forward passes
		Activation activation = Activation::kNone;
		// The weight values to initialise the network with
		double init_weight = 1.0;
		// The bias values to initialise the network with
		double init_bias = 0.0;
		// Enable a dense net configuration, which provides the block input to this layer
		bool use_densenet = false;
	};

	// The name of the fully conected block
	std::string name;
	// Defines each layer in the block. Default to none, passing the original tensor through unmodified.
	std::vector<fc_layer> layers = {};
};

/// @brief Multi Layer Perceptron feature extractor config. Is identical to the fully connected block config.
struct MLPConfig : FCConfig
{
};

/// @brief Convolutional Neural Network feature extractor configuration.
struct CNNConfig
{
	struct conv_layer_config
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
		// The weight to initialise
		float init_weight = std::sqrt(2.0f);
		// The bias to initialise
		float init_bias = 0.0f;
	};

	// Defines groups of CNN layers. Each group can have different numbers of input and output channels.
	std::vector<std::vector<conv_layer_config>> conv_layers = {{{0, 32, 8, 4}, {0, 64, 4, 2}, {0, 64, 3, 1}}};
};

///@brief Feature extractor configuration. Either MLP or CNN.
using FeatureExtractorConfig = std::variant<MLPConfig, CNNConfig>;

/// @brief The policy action output configuration. Used to transform input neural units to match the action space of the
/// environment.
struct PolicyActionOutputConfig
{
	// The activations function to use for the action head
	Activation activation = Activation::kNone;
	// The weight values to initialise the network with
	double init_weight = 0.01;
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
	FCConfig shared = {"shared", {{512, Activation::kReLU}}};
	// The actor fully connected block configuration
	FCConfig actor = {"actor"};
	// The critic fully connected block configuration
	FCConfig critic = {"critic"};
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
	FCConfig q_net = {"q_net"};
};

/// @brief The model configuration for the agent.
using ModelConfig = std::variant<RandomConfig, ActorCriticConfig, QNetModelConfig>;

} // namespace Config

} // namespace drla
