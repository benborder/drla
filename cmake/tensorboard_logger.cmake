find_package(tensorboard_logger QUIET)
if(${tensorboard_logger_FOUND})
	message(STATUS "Found tensorboard_logger ${tensorboard_logger_DIR}")
else()
	include(FetchContent)
	FetchContent_Declare(
		tensorboard_logger
		GIT_REPOSITORY https://github.com/RustingSword/tensorboard_logger.git
		GIT_TAG        c92ede4b3b7d9571696e384bf7ec948b0f3eb77d
	)
	FetchContent_MakeAvailable(tensorboard_logger)
	# Add an alias to use the same target name as find_package mode when linking
	add_library(tensorboard_logger::tensorboard_logger ALIAS tensorboard_logger)
endif()
