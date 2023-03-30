#pragma once

#include "configuration.h"
#include "model.h"
#include "types.h"

#include <torch/torch.h>

#include <optional>
#include <random>
#include <vector>

namespace drla
{

class SearchNode
{
	friend class MCTS;

public:
	SearchNode();
	SearchNode(int action, float prior);

	bool is_expanded() const;

	std::optional<torch::Tensor> get_value() const;

	void expand(const ActionSet& legal_actions, int turn_index, const PredictOutput& prediction);

	void add_exploration_noise(std::mt19937& gen, float dirichlet_alpha, float exploration_fraction);

	PredictOutput get_prediction() const;

	const std::vector<SearchNode>& get_children() const;

	int get_visit_count() const;

	int get_action() const;

private:
	std::vector<SearchNode> child_nodes_;
	torch::Tensor reward_;
	std::vector<torch::Tensor> state_;
	torch::Tensor value_sum_;
	const int action_;
	double prior_;
	int turn_index_ = -1;
	int visit_count_ = 0;
};

struct MinMaxStats
{
	void update(double value)
	{
		if (value > max)
		{
			max = value;
		}
		if (value < min)
		{
			min = value;
		}
	}

	double normalise(double value)
	{
		if (min >= max)
		{
			return value;
		}
		else
		{
			return (value - min) / (max - min);
		}
	}

	double max = -std::numeric_limits<double>::max();
	double min = std::numeric_limits<double>::max();
};

struct MCTSInput
{
	Observations observation;
	ActionSet legal_actions;
	int turn_index;
	bool add_exploration_noise;
};

struct MCTSResult
{
	SearchNode root;
	int max_tree_depth = 0;
	int root_predicted_value = 0;
};

// TODO: Add support for multi objective MCTS
class MCTS
{
public:
	MCTS(const Config::MCTSAgent& config, const ActionSet& action_set, int num_actors);

	MCTSResult search(MCTSModelInterface* model, const MCTSInput& input);

protected:
	SearchNode* select_child(SearchNode* node);
	double ucb_score(SearchNode* parent, SearchNode* child);
	void backpropagate(const std::vector<SearchNode*>& search_path, torch::Tensor value, int turn_index);

private:
	const Config::MCTSAgent& config_;
	const ActionSet action_set_;
	const int num_actors_;

	MinMaxStats stats_;

	std::mt19937 gen_;
};

} // namespace drla
