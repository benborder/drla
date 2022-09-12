#pragma once

#include "types.h"

#include <torch/torch.h>

#include <filesystem>
#include <string>
#include <vector>

namespace drla
{

class Algorithm
{
public:
	virtual ~Algorithm();
	virtual std::vector<UpdateResult> update(int batch) = 0;

	virtual void save(const std::filesystem::path& path) const = 0;
	virtual void load(const std::filesystem::path& path) = 0;
};

inline Algorithm::~Algorithm()
{
}

} // namespace drla
