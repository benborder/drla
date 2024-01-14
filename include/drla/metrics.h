#pragma once

#include "types.h"

#include <map>
#include <vector>

namespace drla
{

/// @brief A simple class to store training metrics from an algorithms update step.
class Metrics
{
public:
	/// @brief Adds a training update metric
	/// @param update_result The training update result to add.
	void add(UpdateResult&& update_result) { update_results_.emplace_back(update_result); }

	/// @brief Ads a single tensor metric
	/// @param name A unique name to referene/index the tensor with. This can be used in display/logging.
	/// @param data The tensor data to store
	void add(std::string name, const torch::Tensor& data)
	{
		data_metrics_.emplace(std::move(name), std::vector<torch::Tensor>{data});
	}

	/// @brief Ads a sequence of tensor metrics. These a grouped together.
	/// @param name A unique name to referene/index the tensor group with. This can be used in display/logging.
	/// @param data The vector of tensor data to store
	void add(std::string name, const std::vector<torch::Tensor>& data) { data_metrics_.emplace(std::move(name), data); }

	/// @brief Gets a reference to the algorithms update step results vector.
	/// @return An immutable reference of the update results vector
	const std::vector<UpdateResult>& get_update_results() const { return update_results_; }

	/// @brief Gets a reference to the algorithms update step stored tensor data metrics.
	/// @return An immutable reference of the tensor data metrics
	const std::map<std::string, std::vector<torch::Tensor>>& get_data() const { return data_metrics_; }

private:
	std::vector<UpdateResult> update_results_;
	std::map<std::string, std::vector<torch::Tensor>> data_metrics_;
};

} // namespace drla
