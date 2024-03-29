#pragma once

#include <ATen/core/Tensor.h>
#include <torch/types.h>

#include <filesystem>
#include <string>

namespace drla
{

/// @brief Loads a tensor from the given filename. Throws if an error occurs.
/// @param filename The full filename to attempt to load the tensor from
/// @return The loaded tensor on the CPU
torch::Tensor load_tensor(const std::string& filename);

/// @brief Loads a vector of tensors from the given filename. Throws if an error occurs.
/// @param path The full path to attempt to load the tensor from
/// @return The loaded tensor on the CPU
std::vector<torch::Tensor> load_tensor_vector(const std::filesystem::path& path, const std::string& name);

/// @brief Saves a tensor to file
/// @param tensor The tensor to save
/// @param filename The full filename to save the tensor to disk
void save_tensor(const torch::Tensor& tensor, const std::string& filename);

/// @brief Reads a vector from a file
/// @param filename The full filename to read from
/// @return The vector read from file
std::vector<int> load_vector(const std::string& filename);

/// @brief Writes a vector to a file
/// @param vec The vector to write to file
/// @param filename The full filename to save the vector to disk
void save_vector(const std::vector<int>& vec, const std::string& filename);

} // namespace drla
