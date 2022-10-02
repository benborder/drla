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
	FCBlockImpl(const Config::FCConfig& config, int input_size);
	FCBlockImpl(
		const Config::FCConfig& config,
		int input_size,
		int output_size,
		Config::FCConfig::fc_layer output_layer_config = {});

	torch::Tensor forward(const torch::Tensor& input);
	int get_output_size() const;

private:
	void make_fc(int input_size);

	const Config::FCConfig config_;

	std::vector<torch::nn::Linear> layers_;
	int output_size_;
};

TORCH_MODULE(FCBlock);

} // namespace drla
