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

	PredictOutput predict(const Observations& observations, bool deterministic = true) override;

	void save(const std::filesystem::path& path) override;
	void load(const std::filesystem::path& path) override;

private:
	const ActionSpace action_space_;
	int value_shape_;
	PolicyActionOutput policy_action_output_;
};

} // namespace drla
