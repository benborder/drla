#pragma once

#include "drla/configuration/model.h"
#include "drla/types.h"

#include <torch/torch.h>

#include <vector>

namespace drla
{

class FCBlockImpl : public torch::nn::Module
{
public:
	FCBlockImpl(const Config::FCConfig& config, const std::string& name, int input_size);
	FCBlockImpl(
		const Config::FCConfig& config,
		const std::string& name,
		int input_size,
		int output_size,
		Config::FCConfig::fc_layer output_layer_config = {});
	FCBlockImpl(const FCBlockImpl& other, const c10::optional<torch::Device>& device);

	torch::Tensor forward(const torch::Tensor& input);
	int get_output_size() const;
	std::shared_ptr<torch::nn::Module> clone(const c10::optional<torch::Device>& device = c10::nullopt) const override;

private:
	void make_fc(int input_size, const std::string& name);

	const Config::FCConfig config_;

	std::vector<torch::nn::Linear> layers_;
	int output_size_;
	bool has_multi_connected_ = false;
};

TORCH_MODULE(FCBlock);

} // namespace drla
