find_package(spdlog QUIET)
if(${spdlog_FOUND})
	message(STATUS "Found spdlog ${spdlog_DIR}")
else()
	include(FetchContent)
	FetchContent_Declare(
		spdlog
		GIT_REPOSITORY https://github.com/gabime/spdlog.git
		GIT_TAG        v1.10.0 # This is kept on v1.10 as there is a linker conflict with libtorch in versions >= v1.11
		GIT_SHALLOW		 TRUE
	)
	set(SPDLOG_SYSTEM_INCLUDES ON)
	FetchContent_MakeAvailable(spdlog)
	# Add an alias to use the same target name as find_package mode when linking
	add_library(spdlog::spdlog ALIAS spdlog)
endif()
