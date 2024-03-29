#include "feature_extractor.h"

#include "cnn_extractor.h"
#include "mlp_extractor.h"
#include "utils.h"

#include <ATen/Parallel.h>
#include <spdlog/spdlog.h>

using namespace drla;

FeatureExtractorImpl::FeatureExtractorImpl(
	const Config::FeatureExtractorConfig& config, const ObservationShapes& observation_shape)
		: output_size_(0)
{
	size_t groups = config.feature_groups.size();
	if (groups != observation_shape.size())
	{
		spdlog::error(
			"Mismatching feature and observation groups. {} feature extractor groups are defined, but there are {} "
			"observation groups.",
			groups,
			observation_shape.size());
		throw std::runtime_error("Mismatching feature and observation groups");
	}
	for (size_t i = 0; i < groups; i++)
	{
		const auto& feature_group = config.feature_groups[i];
		if (std::holds_alternative<Config::MLPConfig>(feature_group))
		{
			auto mlp = std::make_shared<MLPExtractor>(std::get<Config::MLPConfig>(feature_group), observation_shape[i]);
			register_module("mlp_feature_extractor" + std::to_string(i), mlp);
			feature_extractors_.push_back(std::move(mlp));
		}
		else if (std::holds_alternative<Config::CNNConfig>(feature_group))
		{
			auto cnn = std::make_shared<CNNExtractor>(std::get<Config::CNNConfig>(feature_group), observation_shape[i]);
			register_module("cnn_feature_extractor" + std::to_string(i), cnn);
			feature_extractors_.push_back(std::move(cnn));
		}
		else
		{
			spdlog::error("Invalid feature extractor type. Only MLP and CNN are supported.");
			throw std::runtime_error("Invalid feature extractor type");
		}
		output_shape_.push_back(feature_extractors_.back()->get_output_shape());
		const auto& output_shape = output_shape_.back();
		int elements = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<>());
		output_size_ += elements;

		spdlog::debug(
			"{:<28}[{}] -> [{}] -> {}",
			"Observation:",
			fmt::join(observation_shape[i], ", "),
			fmt::join(output_shape, ", "),
			elements);
	}
	spdlog::debug("{:<28}[{}]", "Total observation features: ", output_size_);
}

FeatureExtractorImpl::FeatureExtractorImpl(
	const FeatureExtractorImpl& other, const c10::optional<torch::Device>& device)
		: output_shape_(other.output_shape_), output_size_(other.output_size_)
{
	int index = 0;
	for (auto& feature_extractor : other.feature_extractors_)
	{
		auto fex = feature_extractor->clone(device);
		register_module(other.named_children()[index++].key(), fex);
		feature_extractors_.emplace_back(std::move(std::dynamic_pointer_cast<FeatureExtractorGroup>(fex)));
	}
}

std::vector<torch::Tensor> FeatureExtractorImpl::forward(const Observations& observations)
{
	std::vector<torch::Tensor> output;
	output.resize(feature_extractors_.size());
	at::parallel_for(0, feature_extractors_.size(), 1, [this, &observations, &output](int64_t index, int64_t stop) {
		for (; index < stop; ++index) { output[index] = feature_extractors_[index]->forward(observations[index]); }
	});
	return output;
}

std::vector<std::vector<int64_t>> FeatureExtractorImpl::get_output_shape() const
{
	return output_shape_;
}

int FeatureExtractorImpl::get_output_size() const
{
	return output_size_;
}

std::shared_ptr<torch::nn::Module> FeatureExtractorImpl::clone(const c10::optional<torch::Device>& device) const
{
	torch::NoGradGuard no_grad;
	return std::make_shared<FeatureExtractorImpl>(static_cast<const FeatureExtractorImpl&>(*this), device);
}

Config::FeatureExtractorConfig
make_encoder_config(const Config::MultiEncoderConfig& config, const ObservationShapes& input_shape)
{
	Config::FeatureExtractorConfig fex_config;
	if (std::holds_alternative<Config::FeatureExtractorConfig>(config))
	{
		fex_config = std::get<Config::FeatureExtractorConfig>(config);
	}
	else if (std::holds_alternative<Config::MultiEncoderNetworkConfig>(config))
	{
		const auto& enc_config = std::get<Config::MultiEncoderNetworkConfig>(config);
		for (size_t i = 0; i < input_shape.size(); ++i)
		{
			// assume images take the shape of [channel, height, width] anything else is an mlp
			const auto& shape = input_shape[i];
			if (shape.size() == 3)
			{
				Config::CNNConfig cnn;
				int h = std::max<int>(shape[1], 1);
				int channels = std::max<int>(enc_config.cnn_depth, 1);

				for (int l = 0; h > enc_config.minres; ++l)
				{
					Config::Conv2dConfig conv;
					conv.out_channels = channels;
					conv.kernel_size = enc_config.kernel_size;
					conv.stride = enc_config.stride;
					conv.padding = enc_config.padding;
					conv.init_weight_type = enc_config.init_weight_type;
					conv.init_weight = enc_config.init_weight;
					cnn.layers.push_back(std::move(conv));
					if (enc_config.use_layer_norm)
					{
						cnn.layers.push_back(Config::LayerNormConfig{enc_config.eps});
					}
					cnn.layers.push_back(enc_config.activations);
					h = (h - enc_config.kernel_size + 2 * enc_config.padding) / enc_config.stride + 1;
					channels *= 2;
				}

				fex_config.feature_groups.push_back(std::move(cnn));
			}
			else
			{
				Config::MLPConfig mlp;
				for (int l = 0; l < enc_config.mlp_layers; ++l)
				{
					mlp.layers.push_back(Config::LinearConfig{
						enc_config.mlp_units, enc_config.init_weight_type, enc_config.init_weight, !enc_config.use_layer_norm});
					if (enc_config.use_layer_norm)
					{
						mlp.layers.push_back(Config::LayerNormConfig{enc_config.eps});
					}
					mlp.layers.push_back(enc_config.activations);
				}

				fex_config.feature_groups.push_back(std::move(mlp));
			}
		}
	}
	return fex_config;
}

Config::FeatureExtractorConfig make_decoder_config(
	const Config::MultiDecoderConfig& config, const ObservationShapes& input_shape, const ObservationShapes& output_shape)
{
	Config::FeatureExtractorConfig fex_config;

	if (std::holds_alternative<Config::FeatureExtractorConfig>(config))
	{
		const auto& fex = std::get<Config::FeatureExtractorConfig>(config);
		for (size_t i = 0; i < input_shape.size(); ++i)
		{
			const auto& fexg = fex.feature_groups.at(i);
			if (std::holds_alternative<Config::MLPConfig>(fexg))
			{
				auto mlp = std::get<Config::MLPConfig>(fexg);
				int size = flatten(output_shape[i]);
				mlp.layers = make_output_fc(mlp, Config::LinearConfig{size, Config::InitType::kConstant, 0}).layers;
				fex_config.feature_groups.push_back(std::move(mlp));
			}
			else if (std::holds_alternative<Config::CNNConfig>(fexg))
			{
				auto cnn = std::get<Config::CNNConfig>(fexg);
				int channels = output_shape[i].front();
				for (auto& layer : cnn.layers)
				{
					std::visit(
						[&](auto& l) {
							using layer_type = std::decay_t<decltype(l)>;
							if constexpr (std::is_same_v<Config::ConvTranspose2dConfig, layer_type>)
							{
								if (l.out_channels == 0)
								{
									l.out_channels = channels;
								}
							}
						},
						layer);
				}

				fex_config.feature_groups.push_back(std::move(cnn));
			}
		}
	}
	else if (std::holds_alternative<Config::MultiDecoderNetworkConfig>(config))
	{
		const auto& dec_config = std::get<Config::MultiDecoderNetworkConfig>(config);
		for (size_t i = 0; i < input_shape.size(); ++i)
		{
			// assume images take the shape of [channel, height, width] anything else is an mlp
			const auto& inshape = input_shape[i];
			const auto& outshape = output_shape[i];
			if (inshape.size() == 3)
			{
				Config::CNNConfig cnn;
				int h = std::max<int>(inshape[1], 1);
				int channels = std::max<int>(inshape[0], 1);

				h = (h - 1) * dec_config.stride - 2 * dec_config.padding + dec_config.kernel_size + dec_config.output_padding;
				do {
					Config::ConvTranspose2dConfig conv;
					channels /= 2;
					conv.out_channels = channels;
					conv.kernel_size = dec_config.kernel_size;
					conv.stride = dec_config.stride;
					conv.padding = dec_config.padding;
					conv.init_weight_type = dec_config.init_weight_type;
					conv.init_weight = dec_config.init_weight;
					conv.output_padding = dec_config.output_padding;
					cnn.layers.push_back(std::move(conv));
					if (dec_config.use_layer_norm)
					{
						cnn.layers.push_back(Config::LayerNormConfig{dec_config.eps});
					}
					cnn.layers.push_back(dec_config.activations);
					h = (h - 1) * dec_config.stride - 2 * dec_config.padding + dec_config.kernel_size + dec_config.output_padding;
				}
				while (h < outshape[1]);

				Config::ConvTranspose2dConfig conv;
				conv.out_channels = outshape[0];
				conv.kernel_size = dec_config.kernel_size;
				conv.stride = dec_config.stride;
				conv.padding = dec_config.padding;
				conv.init_weight_type = dec_config.init_out_weight_type;
				conv.init_weight = dec_config.init_out_weight;
				conv.output_padding = dec_config.output_padding;
				cnn.layers.push_back(std::move(conv));

				fex_config.feature_groups.push_back(std::move(cnn));
			}
			else
			{
				Config::MLPConfig mlp;
				for (int l = 0; l < (dec_config.mlp_layers - 1); ++l)
				{
					mlp.layers.push_back(Config::LinearConfig{
						dec_config.mlp_units, dec_config.init_weight_type, dec_config.init_weight, !dec_config.use_layer_norm});
					if (dec_config.use_layer_norm)
					{
						mlp.layers.push_back(Config::LayerNormConfig{dec_config.eps});
					}
					mlp.layers.push_back(dec_config.activations);
				}
				int size = flatten(outshape);
				mlp.layers.push_back(Config::LinearConfig{
					size, dec_config.init_out_weight_type, dec_config.init_out_weight, !dec_config.use_layer_norm});
				fex_config.feature_groups.push_back(std::move(mlp));
			}
		}
	}

	return fex_config;
}

FeatureExtractor
drla::make_multi_encoder(const Config::MultiEncoderConfig& config, const ObservationShapes& input_shape)
{
	return FeatureExtractor(make_encoder_config(config, input_shape), input_shape);
}

FeatureExtractor drla::make_multi_decoder(
	const Config::MultiDecoderConfig& config, const ObservationShapes& input_shape, const ObservationShapes& output_shape)
{
	return FeatureExtractor(make_decoder_config(config, input_shape, output_shape), input_shape);
}
