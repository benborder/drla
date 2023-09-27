find_package(lodepng QUIET)
if(${lodepng_FOUND})
	message(STATUS "Found lodepng ${lodepng_DIR}")
else()
	include(FetchContent)
	FetchContent_Declare(
		lodepng
		GIT_REPOSITORY https://github.com/lvandeve/lodepng.git
		GIT_TAG        master
	)
	FetchContent_MakeAvailable(lodepng)

	# ----------------------------------------------------------------------------
	# Building the lodepng library
	# ----------------------------------------------------------------------------

	add_library(lodepng STATIC
	${lodepng_SOURCE_DIR}/lodepng.cpp
	${lodepng_SOURCE_DIR}/lodepng_util.cpp
	${lodepng_SOURCE_DIR}/pngdetail.cpp
	)

	target_compile_features(lodepng PUBLIC cxx_std_17)
	target_compile_options(lodepng PUBLIC -Wall -Wextra -Wno-unused $<$<CONFIG:RELEASE>:-O2 -flto>)

	set_target_properties(lodepng PROPERTIES POSITION_INDEPENDENT_CODE ON)

	target_include_directories(lodepng
	PUBLIC
		$<INSTALL_INTERFACE:${lodepng_SOURCE_DIR}/>
	PRIVATE
		$<BUILD_INTERFACE:${lodepng_SOURCE_DIR}/>
	)
endif()
