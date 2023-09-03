#pragma once

#include "types.h"

#include <map>
#include <vector>

namespace drla
{

class Metrics
{
public:
	void add(UpdateResult&& update_result) { update_results_.emplace_back(update_result); }

	void add(std::string name, const torch::Tensor& data)
	{
		data_metrics_.emplace(std::move(name), std::vector<torch::Tensor>{data});
	}

	void add(std::string name, const std::vector<torch::Tensor>& data) { data_metrics_.emplace(std::move(name), data); }

	const std::vector<UpdateResult> get_update_results() const { return update_results_; }

	const std::map<std::string, std::vector<torch::Tensor>> get_data() const { return data_metrics_; }

private:
	std::vector<UpdateResult> update_results_;
	std::map<std::string, std::vector<torch::Tensor>> data_metrics_;
};

} // namespace drla
