#include "fc_block.h"

#include "functions.h"
#include "model/utils.h"

#include <spdlog/spdlog.h>
#include <torch/torch.h>

#include <vector>

using namespace drla;
using namespace torch;

Config::FCConfig drla::make_output_fc(Config::FCConfig config, Config::LinearConfig output_layer_config)
{
	// Find all occurences of a 0 size and replace with the output size. If there are none, then just add it onto the back
	bool found = false;
	for (auto& layer : config.layers)
	{
		std::visit(
			[&](auto& l) {
				using layer_type = std::decay_t<decltype(l)>;
				if constexpr (std::is_same_v<Config::LinearConfig, layer_type>)
				{
					if (l.size == 0)
					{
						l.size = output_layer_config.size;
						found = true;
					}
				}
			},
			layer);
	}

	if (!found)
	{
		config.layers.emplace_back(std::move(output_layer_config));
	}

	return config;
}

Config::FCConfig make_fc_config(Config::FCConfig config)
{
	Config::FCConfig output_config;
	int repeats = 0;
	int resolution = 1;
	double factor = 1;
	for (auto it = config.layers.begin(); it != config.layers.end(); ++it)
	{
		std::visit(
			[&](auto& layer) {
				using layer_type = std::decay_t<decltype(layer)>;
				if constexpr (std::is_same_v<Config::LayerRepeatConfig, layer_type>)
				{
					if (repeats < std::abs(layer.repeats))
					{
						factor = layer.factor;
						resolution = layer.resolution;
						++repeats;
						const int num_layers = config.layers.size();
						if ((layer.layers < 1 || layer.layers > num_layers) && num_layers > 0)
						{
							layer.layers = num_layers;
						}
						it = std::prev(it, layer.layers);
					}
					else
					{
						it = std::prev(config.layers.erase(it));
						repeats = 0;
						resolution = 1;
						factor = 1;
					}
				}
				else
				{
					if constexpr (std::is_same_v<Config::LinearConfig, layer_type>)
					{
						layer.size = std::lround(layer.size * std::abs(factor) / resolution) * resolution;
					}
					output_config.layers.push_back(layer);
				}
			},
			*it);
	}
	return output_config;
}

FCBlockImpl::FCBlockImpl(const FCBlockImpl& other, const c10::optional<torch::Device>& device)
		: config_(other.config_), output_size_(other.output_size_), connections_(other.connections_)
{
	int index = 0;
	for (auto& fc_layer : other.layers_)
	{
		std::visit(
			[&](auto& layer) {
				using layer_type = std::decay_t<decltype(layer)>;
				if constexpr (std::is_same_v<ActivationFunction, layer_type>)
				{
					layers_.emplace_back(layer);
				}
				else
				{
					auto l = std::dynamic_pointer_cast<typename layer_type::Impl>(layer->clone(device));
					register_module(other.named_children()[index++].key(), l);
					layers_.emplace_back(std::move(l));
				}
			},
			fc_layer);
	}
}

FCBlockImpl::FCBlockImpl(const Config::FCConfig& config, const std::string& name, int input_size)
		: config_(make_fc_config(config))
{
	output_size_ = input_size;
	if (config_.layers.empty())
	{
		return;
	}

	make_fc(input_size, name);
}

FCBlockImpl::FCBlockImpl(
	const Config::FCConfig& config, const std::string& name, int input_size, Config::LinearConfig output_layer_config)
		: config_(make_output_fc(make_fc_config(config), std::move(output_layer_config)))
{
	make_fc(input_size, name);
}

void FCBlockImpl::make_fc(int input_size, const std::string& name)
{
	spdlog::debug("Constructing {}", name);

	int layer_size = input_size;
	int num_layers =
		std::transform_reduce(config_.layers.begin(), config_.layers.end(), 0, std::plus{}, [](const auto& layer) {
			return std::holds_alternative<Config::LayerConnectionConfig>(layer) ? 0 : 1;
		});
	std::multimap<int, std::pair<int, bool>> layer_connections;
	auto add_connections = [&](int layer) {
		if (layer < 0)
		{
			layer += num_layers + 1;
		}
		auto [lc_iter, lcend] = layer_connections.equal_range(layer);
		for (; lc_iter != lcend; ++lc_iter)
		{
			auto [connection_size, use_residual] = lc_iter->second;
			if (use_residual)
			{
				if (layer_size != connection_size)
				{
					spdlog::error("Residual connection expected size {}, but got {}", connection_size, layer_size);
					throw std::runtime_error("Invalid residual connection");
				}
			}
			else
			{
				layer_size += connection_size;
			}
		}
		layer_connections.erase(layer);
	};

	int l = 0;
	for (const auto& layer_config : config_.layers)
	{
		std::visit(
			[&](const auto& layer) {
				using T = std::decay_t<decltype(layer)>;
				if constexpr (std::is_same_v<Config::LinearConfig, T>)
				{
					if (layer.size <= 0)
					{
						spdlog::error("Invalid layer {} size: {}", l, layer.size);
						throw std::runtime_error("Invalid layer size");
					}
					add_connections(l);
					auto linear = torch::nn::Linear(torch::nn::LinearOptions(layer_size, layer.size).bias(layer.use_bias));
					register_module("linear" + std::to_string(l++), linear);
					weight_init(linear->weight, layer.init_weight_type, layer.init_weight);
					if (layer.use_bias)
					{
						weight_init(linear->bias, layer.init_bias_type, layer.init_bias);
					}
					layers_.emplace_back(std::move(linear));
					layer_size = layer.size;
					spdlog::debug("Layer {:<2}: {:<18}[{}]", l, "linear", layer.size);
				}
				if constexpr (std::is_same_v<Config::LayerNormConfig, T>)
				{
					add_connections(l);
					auto layer_norm = torch::nn::LayerNorm(torch::nn::LayerNormOptions({layer_size}).eps(layer.eps));
					register_module("layer_norm" + std::to_string(l++), layer_norm);
					layers_.emplace_back(std::move(layer_norm));
					spdlog::debug("Layer {:<2}: layer-norm", l);
				}
				if constexpr (std::is_same_v<Config::LayerConnectionConfig, T>)
				{
					int con = layer.connection + (layer.connection < 0 ? (num_layers + 1) : 0);
					if (con <= l)
					{
						spdlog::error("Connection {} connects to current or previous layer {}.", l, layer.connection);
						throw std::runtime_error("Invalid connection");
					}
					connections_.emplace(l, layer);
					layer_connections.emplace(con, std::make_pair(layer_size, layer.residual));
					spdlog::debug(
						"{:<28} [{} -> {}]", layer.residual ? "connecting-residual" : "connecting-concatenaton", l, con);
				}
				if constexpr (std::is_same_v<Config::Activation, T>)
				{
					add_connections(l);
					++l;
					layers_.emplace_back(make_activation(layer));
					spdlog::debug("Layer {:<2}: {:<18}[{}]", l, "activation", activation_name(layer));
				}
			},
			layer_config);
	}

	add_connections(-1);

	output_size_ = layer_size;
}

torch::Tensor FCBlockImpl::forward(const torch::Tensor& input)
{
	const int num_layers = static_cast<int>(layers_.size());
	torch::Tensor x = input;
	std::multimap<int, std::pair<torch::Tensor, bool>> layer_connections;
	auto add_connections = [&](int layer) {
		auto [lc_iter, lcend] = layer_connections.equal_range(layer);
		for (; lc_iter != lcend; ++lc_iter)
		{
			auto [con, use_residual] = lc_iter->second;
			if (use_residual)
			{
				x += con;
			}
			else
			{
				x = torch::cat({x, std::move(con)}, -1);
			}
		}
		layer_connections.erase(layer);
	};

	auto store_connection = [&](int layer) {
		auto [c_iter, cend] = connections_.equal_range(layer);
		for (; c_iter != cend; ++c_iter)
		{
			auto [layer_connection, use_residual] = c_iter->second;
			int con = layer_connection + (layer_connection < 0 ? (num_layers + 1) : 0);
			layer_connections.emplace(con, std::make_pair(x, use_residual));
		}
	};
	store_connection(0);
	for (int l = 0; l < num_layers;)
	{
		std::visit([&](auto& layer) { x = layer(x); }, layers_[l]);
		++l;
		store_connection(l);
		add_connections(l);
	}

	return x;
}

int FCBlockImpl::get_output_size() const
{
	return output_size_;
}

std::shared_ptr<torch::nn::Module> FCBlockImpl::clone(const c10::optional<torch::Device>& device) const
{
	torch::NoGradGuard no_grad;
	return std::make_shared<FCBlockImpl>(static_cast<const FCBlockImpl&>(*this), device);
}
