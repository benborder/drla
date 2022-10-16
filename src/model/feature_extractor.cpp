#include "feature_extractor.h"

#include "cnn_extractor.h"
#include "mlp_extractor.h"

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
			spdlog::error("Invalid feature extractor type. Only MLP, CNN and Resnet are supported.");
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

std::vector<torch::Tensor> FeatureExtractorImpl::forward(const Observations& observations)
{
	std::vector<torch::Tensor> output;
	for (size_t i = 0; i < feature_extractors_.size(); ++i)
	{
		output.push_back(feature_extractors_[i]->forward(observations[i]));
	}
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
