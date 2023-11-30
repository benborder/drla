find_package(GifEncoder QUIET)
if(${GifEncoder_FOUND})
	message(STATUS "Found GifEncoder ${GifEncoder_DIR}")
else()
	include(FetchContent)
	FetchContent_Declare(
		GifEncoder
		GIT_REPOSITORY https://github.com/xiaozhuai/GifEncoder.git
		GIT_TAG        ea3b353b00e6268d7e9cb6650ec8d21369370f02
	)
	FetchContent_MakeAvailable(GifEncoder)
	# Exclude from the ALL target
	set_target_properties(egif_demo PROPERTIES EXCLUDE_FROM_ALL TRUE)
	set_target_properties(egif_test PROPERTIES EXCLUDE_FROM_ALL TRUE)
endif()
