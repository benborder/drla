#include "tensor_storage.h"

#include <spdlog/spdlog.h>

#include <cassert>
#include <filesystem>
#include <fstream>
#include <vector>

torch::Tensor drla::load_tensor(const std::string& filename)
{
	std::ifstream in_file(filename, std::ios::binary);
	if (!in_file.is_open())
	{
		spdlog::error("Unable to open file '{}' for reading", filename);
		return {};
	}
	// [ndims, type, dims... , data...]
	int ndims;
	c10::ScalarType type;
	in_file.read(reinterpret_cast<char*>(&ndims), sizeof(ndims));
	in_file.read(reinterpret_cast<char*>(&type), sizeof(type));
	std::vector<int64_t> dims;
	dims.reserve(ndims);
	for (int d = 0; d < ndims; ++d)
	{
		int dim;
		in_file.read(reinterpret_cast<char*>(&dim), sizeof(dim));
		dims.push_back(dim);
	}

	// Create empty tensor
	auto tensor = torch::empty(dims, type);
	size_t size = tensor.numel() * tensor.element_size();
	// Read data from file into tensor
	in_file.read(reinterpret_cast<char*>(tensor.data_ptr()), size);
	in_file.close();
	return tensor;
}

std::vector<torch::Tensor> drla::load_tensor_vector(const std::filesystem::path& path, const std::string& name)
{
	std::vector<std::string> filenames;
	for (const auto& entry : std::filesystem::directory_iterator(path))
	{
		if (entry.is_regular_file() && entry.path().extension() == ".bin")
		{
			std::string filename = entry.path().filename().string();
			if (filename.compare(0, name.size(), name) == 0)
			{
				filenames.push_back(entry.path());
			}
		}
	}
	// Sort the filenames as the order is unspecified in the std::filesystem::directory_iterator spec
	std::sort(filenames.begin(), filenames.end());
	std::vector<torch::Tensor> tensors;
	for (auto& filename : filenames) { tensors.push_back(load_tensor(filename)); }
	return tensors;
}

void drla::save_tensor(const torch::Tensor& tensor, const std::string& filename)
{
	std::ofstream out_file(filename, std::ios::binary);
	if (!out_file.is_open())
	{
		spdlog::error("Unable to open file '{}' for writing", filename);
		return;
	}
	// [ndims, type, dims... , data...]
	int ndims = tensor.dim();
	auto dims = tensor.sizes();
	size_t size = tensor.numel() * tensor.element_size();
	c10::ScalarType type = tensor.scalar_type();
	out_file.write(reinterpret_cast<const char*>(&ndims), sizeof(int));
	out_file.write(reinterpret_cast<const char*>(&type), sizeof(type));
	for (int dim : dims) { out_file.write(reinterpret_cast<const char*>(&dim), sizeof(int)); }
	out_file.write(reinterpret_cast<const char*>(tensor.data_ptr()), size);
	out_file.close();
}

std::vector<int> drla::load_vector(const std::string& filename)
{
	std::vector<int> vec;
	std::ifstream in_file(filename, std::ios::binary);
	if (!in_file.is_open())
	{
		spdlog::error("Unable to open file '{}' for writing", filename);
		return vec;
	}
	size_t size;
	in_file.read(reinterpret_cast<char*>(&size), sizeof(size_t));
	vec.resize(size);
	in_file.read(reinterpret_cast<char*>(vec.data()), size * sizeof(int));
	in_file.close();
	return vec;
}

void drla::save_vector(const std::vector<int>& vec, const std::string& filename)
{
	std::ofstream out_file(filename, std::ios::binary);
	if (!out_file.is_open())
	{
		spdlog::error("Unable to open file '{}' for writing", filename);
		return;
	}
	size_t size = vec.size();
	out_file.write(reinterpret_cast<const char*>(&size), sizeof(size_t));
	out_file.write(reinterpret_cast<const char*>(vec.data()), size * sizeof(int));
	out_file.close();
}
