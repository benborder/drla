cmake_minimum_required(VERSION 3.14)

# ----------------------------------------------------------------------------
# Defining the Deep Reinforcement Learning Agent (DRLA) library
# ----------------------------------------------------------------------------

project(drla
	VERSION 0.1.0
	DESCRIPTION "Deep Reinforcement Learning Agent (DRLA) library"
	LANGUAGES CXX
)

option(BUILD_AUXILIARY "Build the auxiliary library" ON)

# ----------------------------------------------------------------------------
# Building the DRLA library
# ----------------------------------------------------------------------------

include(cmake/spdlog.cmake)

set(THREADS_PREFER_PTHREAD_FLAG ON)  # Set -pthreads
find_package(Threads REQUIRED)
find_package(Torch REQUIRED)

# The core drla library
add_library(drla STATIC
	# Common
	src/common/get_time.cpp

	# Agent
	src/agent/agent.cpp
	src/agent/on_policy_agent.cpp
	src/agent/off_policy_agent.cpp
	src/agent/interactive_agent.cpp
	src/agent/mcts_agent.cpp
	src/agent/mcts.cpp
	src/agent/hybrid_agent.cpp

	# Buffers
	src/buffers/rollout_buffer.cpp
	src/buffers/replay_buffer.cpp
	src/buffers/minibatch_buffer.cpp
	src/buffers/episodic_per_buffer.cpp
	src/buffers/mcts_episode.cpp
	src/buffers/mcts_replay_buffer.cpp
	src/buffers/hybrid_episode.cpp
	src/buffers/hybrid_replay_buffer.cpp
	src/buffers/off_policy_episode.cpp
	src/buffers/tensor_storage.cpp

	# Training algorithms
	src/algorithm/a2c.cpp
	src/algorithm/dqn.cpp
	src/algorithm/ppo.cpp
	src/algorithm/sac.cpp
	src/algorithm/muzero.cpp
	src/algorithm/dreamer.cpp
	src/algorithm/optimiser.cpp

	# Models
	src/model/utils.cpp
	src/model/feature_extractor.cpp
	src/model/cnn_extractor.cpp
	src/model/mlp_extractor.cpp
	src/model/fc_block.cpp
	src/model/res_block.cpp
	src/model/actor_net.cpp
	src/model/actor_critic_model.cpp
	src/model/soft_actor_critic_model.cpp
	src/model/random_model.cpp
	src/model/qnet_model.cpp
	src/model/muzero_model.cpp
	src/model/dreamer_model.cpp

	# Output distributions
	src/distributions/bernoulli.cpp
	src/distributions/categorical.cpp
	src/distributions/distribution.cpp
	src/distributions/normal.cpp
	src/distributions/diagonal_gaussian.cpp
)
add_library(drla::drla ALIAS drla)

target_compile_options(drla PRIVATE -Wall -Wextra -Werror -Wpedantic $<$<CONFIG:RELEASE>:-O2 -flto>)
target_compile_features(drla PRIVATE cxx_std_17)

target_include_directories(drla
	PUBLIC
		$<INSTALL_INTERFACE:include>
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
	PRIVATE
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/drla>
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/drla/model>
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/agent>
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/algorithm>
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/buffers>
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/distributions>
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/model>
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/mcts>
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/common>
)

target_link_libraries(drla
PUBLIC
	${TORCH_LIBRARIES}
	Threads::Threads
PRIVATE
	$<BUILD_INTERFACE:spdlog::spdlog>
)

if(BUILD_AUXILIARY)
	add_subdirectory(auxiliary)
endif()

# ----------------------------------------------------------------------------
# Installing the DRLA library
# ----------------------------------------------------------------------------

install(
	TARGETS drla
	EXPORT drlaTargets
	ARCHIVE DESTINATION lib
	LIBRARY DESTINATION lib
	INCLUDES DESTINATION include
)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
	"drlaConfigVersion.cmake"
	VERSION ${PROJECT_VERSION}
	COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(
	"${PROJECT_SOURCE_DIR}/cmake/drlaConfig.cmake.in"
	"${PROJECT_BINARY_DIR}/drlaConfig.cmake"
	INSTALL_DESTINATION ${CMAKE_INSTALL_PREFIX}/cmake
)

install(
	EXPORT drlaTargets
	FILE drlaTargets.cmake
	NAMESPACE drla::
	DESTINATION ${CMAKE_INSTALL_PREFIX}/cmake
)

install(
	FILES "${PROJECT_BINARY_DIR}/drlaConfig.cmake"
				"${PROJECT_BINARY_DIR}/drlaConfigVersion.cmake"
	DESTINATION ${CMAKE_INSTALL_PREFIX}/cmake
)

install(
	DIRECTORY include/
	DESTINATION include
)
