find_package(nlohmann_json QUIET)
if(${nlohmann_json_FOUND})
	message(STATUS "Found nlohmann_json ${nlohmann_json_DIR}")
else()
	include(FetchContent)
	FetchContent_Declare(
		nlohmann_json
		GIT_REPOSITORY https://github.com/nlohmann/json.git
		GIT_TAG        v3.10.5
	)
	FetchContent_MakeAvailable(nlohmann_json)
endif()
