cmake_minimum_required(VERSION 3.14)

# ----------------------------------------------------------------------------
# Building the DRLA Auxiliary library
# ----------------------------------------------------------------------------

# Dependencies
include(${PROJECT_SOURCE_DIR}/cmake/nlohmann_json.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/tensorboard_logger.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/lodepng.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/GifEncoder.cmake)

set(THREADS_PREFER_PTHREAD_FLAG ON)  # Set -pthreads
find_package(Threads REQUIRED)
find_package(Torch REQUIRED)

# aux drla library
add_library(drla_aux STATIC
	src/env_manager.cpp
	src/metrics_logger.cpp
	src/serialise_json.cpp
	src/tensor_media.cpp
)
add_library(drla::aux ALIAS drla_aux)

target_compile_options(drla_aux PRIVATE -Wall -Wextra -Werror $<$<CONFIG:RELEASE>:-O2 -flto>)
target_compile_features(drla_aux PRIVATE cxx_std_17)

target_include_directories(drla_aux
	PUBLIC
		$<INSTALL_INTERFACE:include>
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
	PRIVATE
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/drla/auxiliary>
		$<BUILD_INTERFACE:${GifEncoder_SOURCE_DIR}/egif>
		$<BUILD_INTERFACE:${lodepng_SOURCE_DIR}>
)

target_link_libraries(drla_aux
PUBLIC
	drla::drla
	${TORCH_LIBRARIES}
	Threads::Threads
PRIVATE
	$<BUILD_INTERFACE:nlohmann_json::nlohmann_json>
	$<BUILD_INTERFACE:spdlog::spdlog>
	$<BUILD_INTERFACE:tensorboard_logger::tensorboard_logger>
	$<BUILD_INTERFACE:egif>
	$<BUILD_INTERFACE:lodepng>
)

# ----------------------------------------------------------------------------
# Installing the DRLA Auxiliary library
# ----------------------------------------------------------------------------

install(
	TARGETS drla_aux
	EXPORT drla_auxTargets
	ARCHIVE DESTINATION lib
	LIBRARY DESTINATION lib
	INCLUDES DESTINATION include
	COMPONENT aux
)

install(
	EXPORT drla_auxTargets
	FILE drla_auxTargets.cmake
	NAMESPACE drla::
	DESTINATION ${CMAKE_INSTALL_PREFIX}/cmake
	COMPONENT aux
)

install(
	DIRECTORY include/
	DESTINATION include
	COMPONENT aux
)
