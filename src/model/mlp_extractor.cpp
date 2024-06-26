#include "mlp_extractor.h"

#include <spdlog/spdlog.h>

using namespace drla;

MLPExtractor::MLPExtractor(const Config::MLPConfig& config, const std::vector<int64_t>& observation_shape)
		: hidden_(nullptr)
{
	output_size_ = std::accumulate(observation_shape.begin(), observation_shape.end(), 1, std::multiplies<>());
	hidden_ = register_module(config.name, FCBlock(config, config.name, output_size_));
	output_size_ = hidden_->get_output_size();
}

MLPExtractor::MLPExtractor(const MLPExtractor& other, const c10::optional<torch::Device>& device)
		: hidden_(std::dynamic_pointer_cast<FCBlockImpl>(other.hidden_->clone(device))), output_size_(other.output_size_)
{
	register_module(other.named_children().front().key(), hidden_);
}

torch::Tensor MLPExtractor::forward(const torch::Tensor& observation)
{
	return hidden_(torch::flatten(observation, 1));
}

std::vector<int64_t> MLPExtractor::get_output_shape() const
{
	return {output_size_};
}

std::shared_ptr<torch::nn::Module> MLPExtractor::clone(const c10::optional<torch::Device>& device) const
{
	torch::NoGradGuard no_grad;
	return std::make_shared<MLPExtractor>(static_cast<const MLPExtractor&>(*this), device);
}
