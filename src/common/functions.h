#pragma once

#include <ATen/core/Tensor.h>

#include <vector>

namespace drla
{

// Vector helpers

template <int start, int end = 0, typename T>
inline std::vector<T> slice(const std::vector<T>& vec)
{
	if constexpr (start >= 0)
	{
		if constexpr (end > 0)
		{
			return std::vector<T>(vec.begin() + start, vec.begin() + end);
		}
		else
		{
			return std::vector<T>(vec.begin() + start, vec.end() + end);
		}
	}
	else if constexpr (end > 0)
	{
		return std::vector<T>(vec.end() + start, vec.begin() + end);
	}
	else
	{
		return std::vector<T>(vec.end() + start, vec.end() + end);
	}
}

template <typename T>
inline std::vector<T> slice(const std::vector<T>& vec, int start, int end = 0)
{
	if (start >= 0)
	{
		if (end > 0)
		{
			return std::vector<T>(vec.begin() + start, vec.begin() + end);
		}
		else
		{
			return std::vector<T>(vec.begin() + start, vec.end() + end);
		}
	}
	else if (end > 0)
	{
		return std::vector<T>(vec.end() + start, vec.begin() + end);
	}
	else
	{
		return std::vector<T>(vec.end() + start, vec.end() + end);
	}
}

template <typename T>
inline std::vector<T> operator+(const std::vector<T>& a, const std::vector<T>& b)
{
	std::vector<T> c(a);
	c.insert(c.end(), b.begin(), b.end());
	return c;
}

template <typename T>
inline std::vector<T> operator+(std::vector<T>&& a, std::vector<T>&& b)
{
	a.insert(a.end(), std::make_move_iterator(b.begin()), std::make_move_iterator(b.end()));
	return std::move(a);
}

template <typename T>
inline void operator<<(std::vector<T>& a, const std::vector<T>& b)
{
	a.insert(a.end(), b.begin(), b.end());
}

template <typename T>
inline void operator<<(std::vector<T>& a, std::vector<T>&& b)
{
	a.insert(a.end(), std::make_move_iterator(b.begin()), std::make_move_iterator(b.end()));
}

// Tensor helpers

inline torch::Tensor make_tensor(float x, c10::TensorOptions options = {})
{
	return torch::from_blob(&x, 1, options).clone();
}

inline torch::Tensor make_tensor(const std::vector<float>& x, at::IntArrayRef size, c10::TensorOptions options = {})
{
	return torch::from_blob(const_cast<float*>(x.data()), size, options).clone();
}

#define TENSOR_SIZE(tensor)                                     \
	do {                                                          \
		std::cout << #tensor ": " << (tensor).sizes() << std::endl; \
	}                                                             \
	while (0)

#define PRINT_TENSOR(tensor)                            \
	do {                                                  \
		std::cout << #tensor ": " << (tensor) << std::endl; \
	}                                                     \
	while (0)

} // namespace drla
