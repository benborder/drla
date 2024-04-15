#pragma once

#include <ATen/core/Tensor.h>
#include <drla/types.h>
#include <torch/types.h>

#include <filesystem>
#include <vector>

namespace drla
{

void save_episode(const std::vector<StepData>& episode_data, const std::filesystem::path& path);

}
