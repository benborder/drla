#pragma once

#include <random>
#include <vector>

namespace drla
{

class dirichlet_distribution
{
public:
	dirichlet_distribution(double alpha, size_t size)
	{
		std::vector<double> alpha_params;
		alpha_params.resize(size, alpha);
		set_params(alpha_params);
	}

	dirichlet_distribution(const std::vector<double>& alpha) { set_params(alpha); }

	void set_params(const std::vector<double>& alpha)
	{
		gamma_.clear();
		gamma_.reserve(alpha.size());
		for (auto& a : alpha) { gamma_.emplace_back(a, 1.0); }
	}

	template <class RNG>
	std::vector<double> operator()(RNG& generator)
	{
		std::vector<double> x(gamma_.size());
		double sum = 0.0;
		for (size_t i = 0; i < gamma_.size(); ++i) { sum += x[i] = gamma_[i](generator); }
		for (double& xi : x) { xi /= sum; }
		return x;
	}

private:
	std::vector<std::gamma_distribution<>> gamma_;
};

} // namespace drla
