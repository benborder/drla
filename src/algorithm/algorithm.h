#pragma once

#include "metrics.h"
#include "types.h"

#include <filesystem>
#include <string>
#include <vector>

namespace drla
{

class Algorithm
{
public:
	virtual ~Algorithm() = default;
	virtual std::string name() const = 0;
	virtual Metrics update(int timestep) = 0;

	virtual void save(const std::filesystem::path& path) const = 0;
	virtual void load(const std::filesystem::path& path) = 0;
};

} // namespace drla
