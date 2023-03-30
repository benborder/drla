#pragma once

#include <math.h>

#include <limits>

namespace drla
{

/// @brief Stats class for calculating running mean, variance, stdev, min/max.
/// @tparam T The underlying type. Defaults to double, should not be an integral.
template <typename T = double>
class Stats
{
public:
	/// @brief Gets the moving mean
	[[nodiscard]] T get_mean() const { return mean_; }

	/// @brief Gets the approximated variance
	[[nodiscard]] T get_var() const { return var_; }

	/// @brief Gets the approximated stdev
	[[nodiscard]] T get_stdev() const { return std::sqrt(var_); }

	/// @brief Gets the all time max
	[[nodiscard]] T get_max() const { return max_; }

	/// @brief Gets the all time min
	[[nodiscard]] T get_min() const { return min_; }

	/// @brief Gets the total number of updates
	[[nodiscard]] size_t get_count() const { return count_; }

	/// @brief Sets the smoothing factor.
	/// @param ratio The smoothing ratio, valid over the domain [0,0.9999). 0 equals no smoothing, 0.9999 equals max
	/// smoothing.
	void set_ratio(double ratio) { ratio_ = 1.0 - std::clamp(ratio, 0.0, 0.9999); }

	/// @brief Updates the stats with a new value
	void update(const T& val)
	{
		if (count_ == 0)
		{
			mean_ = val;
			count_ = 1;
			max_ = val;
			min_ = val;
			return;
		}

		count_++;
		auto r = std::max(1.0 / (double)count_, ratio_);
		auto ir = (1.0 - r);
		auto new_mean = mean_ * ir + r * val;
		var_ = var_ * ir + r * (val - mean_) * (val - new_mean);
		mean_ = new_mean;
		max_ = std::max(max_, val);
		min_ = std::min(min_, val);
	}

private:
	T mean_ = 0;
	T var_ = 0;
	T max_ = -std::numeric_limits<T>::max();
	T min_ = std::numeric_limits<T>::max();
	double ratio_ = 0.01;
	size_t count_ = 0;
};

} // namespace drla
