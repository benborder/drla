# DRLA - Deep Reinforcement Learning Agent Library

The DRLA library is a C++ Deep Reinforcement Learning Agent based on libtorch (pytorch C++ API). The motivation for this project is to provide general DRL agents with common interfaces for use in C++ applications and environments, allowing training and running solely within the C++ runtime.

*Note: This library is still in development and not yet considered stable, with features, potential optimisations, design improvements and bug fixes ongoing.*

## Features

- Agent interfaces to integrate with environments and your project
- Multi threaded async environment functionality
- Model-free On-Policy rollout based algorithms (PPO, A2C)
- Model-free Off-Policy replay based algorithms (DQN, SAC)
- Model-based Monte-Carlo tree search algorithms (MuZero)
- Hybrid Model-free and Model-based algorithms (Dreamer)

## Example

See the following repositories for examples on how to use this library:

- [drla-atari](https://github.com/benborder/drla-atari)

## Dependencies

The library has been designed to have minimal dependencies with the only dependencies for the core library:

- [libtorch](https://github.com/pytorch/pytorch) (build and runtime)
- [spdlog](https://github.com/gabime/spdlog) (build)
- Compiler with C++17 support
- CMake 3.14 or newer

Additional auxillary functionality such as config json serialisation and helper classes require the following libraries:

- [nlohmann_json](https://github.com/nlohmann/json) (build)

## Installing

Install libtorch at `/usr/local/libtorch` and ensure cmake is also installed. There are two methods for including the library in your cmake project.

1. Installing and using cmake `find_package`.
2. Using FetchContent to obtain the library as a subproject.

For both options make sure to add `drla::drla` to your projects `target_link_libraries`. To use the auxillary functionality also add the `drla::aux` target.

### 1. Install and find_package

Build and install DRLA

```bash
cmake --preset release
cmake --build --preset release --target install --parallel 8
```

Include in your cmake project via find package `find_package(drla)`.

### 2. FetchContent subproject

Add the following to your cmake project:

```cmake
FetchContent_Declare(
	drla
	GIT_REPOSITORY https://github.com:benborder/drla.git
	GIT_TAG        master
)
FetchContent_MakeAvailable(drla)
```
