#include "env_manager.h"

using namespace drla;

Environment* GenericEnvironmentManager::get_environment(int i)
{
	return envs_.at(i).get();
}

Environment* GenericEnvironmentManager::add_environment()
{
	auto env = make_environment();
	std::lock_guard guard(m_envs_);
	return envs_.emplace_back(std::move(env)).get();
}

EnvironmentConfiguration GenericEnvironmentManager::get_configuration()
{
	std::lock_guard guard(m_envs_);
	if (envs_.empty())
	{
		return add_environment()->get_configuration();
	}
	return envs_.front()->get_configuration();
}

int GenericEnvironmentManager::num_envs() const
{
	return static_cast<int>(envs_.size());
}

void GenericEnvironmentManager::reset()
{
	std::lock_guard guard(m_envs_);
	envs_.clear();
}
