cmake_minimum_required(VERSION 3.14)

# ----------------------------------------------------------------------------
# Defining the Deep Reinforcement Learning Agent (DRLA) library
# ----------------------------------------------------------------------------

project(drla
	VERSION 0.1.0
	DESCRIPTION "Deep Reinforcement Learning Agent (DRLA) library"
	LANGUAGES CXX
)

# ----------------------------------------------------------------------------
# Building the DRLA library
# ----------------------------------------------------------------------------

find_package(spdlog QUIET)
if(${spdlog_FOUND})
	message(STATUS "Found spdlog ${spdlog_DIR}")
else()
	include(FetchContent)
	FetchContent_Declare(
		spdlog
		GIT_REPOSITORY https://github.com/gabime/spdlog.git
		GIT_TAG        v1.10.0
	)
	FetchContent_MakeAvailable(spdlog)
	# Add an alias to use the same target name as find_package mode when linking
	add_library(spdlog::spdlog ALIAS spdlog)
endif()

find_package(nlohmann_json QUIET)
if(${nlohmann_json_FOUND})
	message(STATUS "Found nlohmann_json ${nlohmann_json_DIR}")
else()
	FetchContent_Declare(
		nlohmann_json
		GIT_REPOSITORY https://github.com/nlohmann/json.git
		GIT_TAG        v3.10.5
	)
	FetchContent_MakeAvailable(nlohmann_json)
endif()

set(THREADS_PREFER_PTHREAD_FLAG ON)  # Set -pthreads
find_package(Threads REQUIRED)
find_package(Torch REQUIRED)

# The core drla library
add_library(drla STATIC
	# Agent
	src/agent/agent.cpp
	src/agent/on_policy_agent.cpp
	src/agent/off_policy_agent.cpp
	src/agent/interactive_agent.cpp

	# Buffers
	src/buffers/rollout_buffer.cpp
	src/buffers/replay_buffer.cpp
	src/buffers/minibatch_buffer.cpp

	# Training algorithms
	src/algorithm/a2c.cpp
	src/algorithm/dqn.cpp
	src/algorithm/ppo.cpp
	src/algorithm/sac.cpp

	# Models
	src/model/feature_extractor.cpp
	src/model/cnn_extractor.cpp
	src/model/mlp_extractor.cpp
	src/model/fc_block.cpp
	src/model/res_block.cpp
	src/model/policy_action_output.cpp
	src/model/actor_critic_model.cpp
	src/model/soft_actor_critic_model.cpp
	src/model/random_model.cpp
	src/model/qnet_model.cpp

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
)

target_link_libraries(drla
PUBLIC
	${TORCH_LIBRARIES}
	Threads::Threads
PRIVATE
	$<BUILD_INTERFACE:spdlog::spdlog>
)

# aux drla library
add_library(drla_aux STATIC
	src/auxiliary/env_manager.cpp
	src/auxiliary/serialise_json.cpp
)
add_library(drla::aux ALIAS drla_aux)

target_compile_options(drla_aux PRIVATE -Wall -Wextra -Werror -Wpedantic $<$<CONFIG:RELEASE>:-O2 -flto>)
target_compile_features(drla_aux PRIVATE cxx_std_17)

target_include_directories(drla_aux
	PUBLIC
		$<INSTALL_INTERFACE:include>
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
	PRIVATE
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/drla>
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/drla/auxiliary>
)

target_link_libraries(drla_aux
PUBLIC
	drla::drla
	${TORCH_LIBRARIES}
	Threads::Threads
PRIVATE
	$<BUILD_INTERFACE:nlohmann_json::nlohmann_json>
	$<BUILD_INTERFACE:spdlog::spdlog>
)

# ----------------------------------------------------------------------------
# Installing the DRLA library
# ----------------------------------------------------------------------------

install(
	TARGETS drla drla_aux
	EXPORT drla_Targets
	ARCHIVE DESTINATION lib
	LIBRARY DESTINATION lib
	INCLUDES DESTINATION include
)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
	"drlaConfigVersion.cmake"
	VERSION ${PROJECT_VERSION}
	COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
	"${PROJECT_SOURCE_DIR}/cmake/drlaConfig.cmake.in"
	"${PROJECT_BINARY_DIR}/drlaConfig.cmake"
	INSTALL_DESTINATION ${CMAKE_INSTALL_PREFIX}/cmake
)

install(
	EXPORT drla_Targets
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
