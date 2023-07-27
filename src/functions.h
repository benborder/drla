#pragma once

#include <vector>

namespace drla
{

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

} // namespace drla
