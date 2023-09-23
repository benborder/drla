#pragma once

#include "drla/environment.h"

#include <memory>
#include <mutex>
#include <vector>

namespace drla
{

/// @brief A general environment manager
class GenericEnvironmentManager : public EnvironmentManager
{
public:
	/// @brief Gets the specified environment. Throws if the index is out of bounds.
	/// @param i The index of the environment to return
	/// @return A non owning raw pointer to the environment interface
	Environment* get_environment(int i) override;
	/// @brief Adds a new environment to the manager
	/// @return A non owning pointer to the new environment
	Environment* add_environment() override;
	/// @brief Gets the environment configuration which describes the observation shape and data types, action space and
	/// reward types for all the environments. Creates an environment if one doesn't yet exist, otherwise uses the first
	/// environment created.
	/// @return The environment configuration used for all environments
	EnvironmentConfiguration get_configuration() override;
	/// @brief Returns the number of enviroments that have been created
	/// @return The number of envs
	int num_envs() const override;
	/// @brief Removes all environments, running their destructors
	void reset() override;

protected:
	std::mutex m_envs_;
	std::vector<std::unique_ptr<Environment>> envs_;
};

} // namespace drla
