#pragma once

#include "drla/configuration/model.h"
#include "drla/types.h"

#include <torch/torch.h>

#include <memory>
#include <variant>
#include <vector>

namespace drla
{

class FeatureExtractorGroup : public torch::nn::Module
{
public:
	~FeatureExtractorGroup() = default;

	virtual torch::Tensor forward(const torch::Tensor& observation) = 0;
	virtual std::vector<int64_t> get_output_shape() const = 0;
	virtual std::shared_ptr<torch::nn::Module> clone(const c10::optional<torch::Device>& device = c10::nullopt) const = 0;
};

class FeatureExtractorImpl : public torch::nn::Module
{
public:
	FeatureExtractorImpl(const Config::FeatureExtractorConfig& config, const ObservationShapes& observation_shape);
	FeatureExtractorImpl(const FeatureExtractorImpl& other, const c10::optional<torch::Device>& device);

	std::vector<torch::Tensor> forward(const Observations& observations);
	std::vector<std::vector<int64_t>> get_output_shape() const;
	int get_output_size() const;
	std::shared_ptr<torch::nn::Module> clone(const c10::optional<torch::Device>& device = c10::nullopt) const override;

private:
	std::vector<std::shared_ptr<FeatureExtractorGroup>> feature_extractors_;
	std::vector<std::vector<int64_t>> output_shape_;
	int output_size_;
};

TORCH_MODULE(FeatureExtractor);

} // namespace drla
