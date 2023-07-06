#pragma once

#include "configuration/model.h"
#include "model.h"
#include "policy_action_output.h"
#include "types.h"

#include <torch/torch.h>

#include <variant>
#include <vector>

namespace drla
{

class RandomModel : public Model
{
public:
	RandomModel(const Config::ModelConfig& config, const ActionSpace& action_space, int value_shape);
	RandomModel(const RandomModel& other, const c10::optional<torch::Device>& device);

	ModelOutput predict(const ModelInput& input) override;

	ModelOutput initial() const override;

	StateShapes get_state_shape() const override;

	void save(const std::filesystem::path& path) override;
	void load(const std::filesystem::path& path) override;

	std::shared_ptr<torch::nn::Module> clone(const c10::optional<torch::Device>& device = c10::nullopt) const override;

private:
	const ActionSpace action_space_;
	int value_shape_;
	PolicyActionOutput policy_action_output_;
};

} // namespace drla
