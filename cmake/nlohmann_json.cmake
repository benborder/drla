find_package(nlohmann_json QUIET)
if(${nlohmann_json_FOUND})
	message(STATUS "Found nlohmann_json ${nlohmann_json_DIR}")
else()
	include(FetchContent)
	FetchContent_Declare(
		nlohmann_json
		GIT_REPOSITORY https://github.com/nlohmann/json.git
		GIT_TAG        v3.11.3
		GIT_SHALLOW		 TRUE
	)
	set(JSON_SystemInclude ON)
	FetchContent_MakeAvailable(nlohmann_json)
endif()
