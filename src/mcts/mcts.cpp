#include "mcts.h"

#include "dirichlet.h"

using namespace drla;

SearchNode::SearchNode() : action_(-1), prior_(0)
{
}

SearchNode::SearchNode(int action, float prior) : action_(action), prior_(prior)
{
}

bool SearchNode::is_expanded() const
{
	return !child_nodes_.empty();
}

std::optional<torch::Tensor> SearchNode::get_value() const
{
	return visit_count_ > 0 ? std::make_optional(value_sum_.div(visit_count_)) : std::nullopt;
}

void SearchNode::expand(const ActionSet& legal_actions, int turn_index, const PredictOutput& prediction)
{
	turn_index_ = turn_index;
	reward_ = prediction.reward[0];
	value_sum_ = torch::zeros_like(prediction.values[0]);
	state_ = prediction.state;
	torch::Tensor policy_values = torch::empty({int(legal_actions.size())}, prediction.policy.device());
	for (size_t i = 0; i < legal_actions.size(); ++i) { policy_values[i] = prediction.policy[0][legal_actions[i]]; }
	policy_values = torch::softmax(policy_values, 0);
	child_nodes_.reserve(legal_actions.size());
	for (size_t i = 0; i < legal_actions.size(); ++i)
	{
		child_nodes_.emplace_back(legal_actions[i], policy_values[i].item<float>());
	}
}

void SearchNode::add_exploration_noise(std::mt19937& gen, float dirichlet_alpha, float exploration_fraction)
{
	size_t size = child_nodes_.size();
	auto noise = dirichlet_distribution(dirichlet_alpha, size)(gen);
	for (size_t i = 0; i < size; ++i)
	{
		auto& node = child_nodes_[i];
		node.prior_ = node.prior_ * (1 - exploration_fraction) + noise[i] * exploration_fraction;
	}
}

PredictOutput SearchNode::get_prediction() const
{
	PredictOutput prediction;
	prediction.state = state_;
	prediction.reward = reward_;
	prediction.action = torch::empty({1}, state_.front().device());
	prediction.action[0] = action_;
	auto value = get_value();
	if (value)
	{
		prediction.values = *value;
	}
	else
	{
		prediction.values = torch::zeros_like(reward_);
	}
	return prediction;
}

const std::vector<SearchNode>& SearchNode::get_children() const
{
	return child_nodes_;
}

int SearchNode::get_visit_count() const
{
	return visit_count_;
}

int SearchNode::get_action() const
{
	return action_;
}

MCTS::MCTS(const Config::MCTSAgent& config, const ActionSet& action_set, int num_actors)
		: config_(config), action_set_(action_set), num_actors_(num_actors), gen_(std::random_device{}())
{
}

MCTSResult MCTS::search(MCTSModelInterface* model, const MCTSInput& input)
{
	if (input.legal_actions.empty())
	{
		throw std::runtime_error("Legal actions must be non zero");
	}

	MCTSResult res;
	// Make sure the input observations are on the same deivce as the model
	auto device = model->parameters().front().device();
	Observations observation;
	for (auto& obs : input.observation) { observation.push_back(obs.to(device)); }

	auto predict_output = static_cast<Model*>(model)->predict({observation});
	predict_output.reward = model->support_to_scalar(predict_output.reward);
	predict_output.values = model->support_to_scalar(predict_output.values);

	res.root.expand(input.legal_actions, input.turn_index, predict_output);

	if (input.add_exploration_noise)
	{
		res.root.add_exploration_noise(gen_, config_.root_dirichlet_alpha, config_.root_exploration_fraction);
	}

	stats_ = {};

	for (int sim_count = 0; sim_count < config_.num_simulations; ++sim_count)
	{
		auto sim_turn_index = input.turn_index;
		auto* node = &res.root;
		std::vector<SearchNode*> search_path = {node};
		int current_tree_depth = 0;

		SearchNode* parent = nullptr;
		while (node->is_expanded())
		{
			++current_tree_depth;
			parent = node;
			node = select_child(node);
			search_path.push_back(node);

			sim_turn_index = (sim_turn_index + 1) % num_actors_;
		}

		auto pred = parent->get_prediction();
		pred.action[0] = node->action_;
		pred.action.unsqueeze_(0);
		auto node_predict_output = model->predict(pred);
		node_predict_output.reward = model->support_to_scalar(node_predict_output.reward);
		node_predict_output.values = model->support_to_scalar(node_predict_output.values);
		node->expand(action_set_, sim_turn_index, node_predict_output);

		backpropagate(search_path, node_predict_output.values[0], sim_turn_index);

		if (current_tree_depth > res.max_tree_depth)
		{
			res.max_tree_depth = current_tree_depth;
		}
	}

	return res;
}

SearchNode* MCTS::select_child(SearchNode* node)
{
	auto max_ucb = -std::numeric_limits<double>::max();
	std::vector<double> ucb_vals;
	for (auto& child : node->child_nodes_)
	{
		auto score = ucb_score(node, &child);
		if (score > max_ucb)
		{
			max_ucb = score;
		}
		ucb_vals.push_back(score);
	}

	std::vector<SearchNode*> nodes;
	for (size_t i = 0, ilen = node->child_nodes_.size(); i < ilen; ++i)
	{
		if (ucb_vals[i] == max_ucb)
		{
			nodes.push_back(&node->child_nodes_.at(i));
		}
	}

	assert(!nodes.empty());

	if (nodes.size() == 1)
	{
		return nodes.back();
	}
	else
	{
		std::uniform_int_distribution<> dist(0, nodes.size() - 1);
		return nodes.at(dist(gen_));
	}
}

double MCTS::ucb_score(SearchNode* parent, SearchNode* child)
{
	auto pb_c = std::log((parent->visit_count_ + config_.pb_c_base + 1) / config_.pb_c_base) + config_.pb_c_init;
	pb_c *= std::sqrt(parent->visit_count_) / (child->visit_count_ + 1);
	auto prior_score = pb_c * child->prior_;
	double value_score = 0;
	if (child->visit_count_ > 0)
	{
		auto value = num_actors_ == 1 ? *child->get_value() : -*child->get_value();
		value_score = stats_.normalise((child->reward_ + config_.gamma[0] * value).sum().item<float>());
	}
	return value_score + prior_score;
}

void MCTS::backpropagate(const std::vector<SearchNode*>& search_path, torch::Tensor value, int turn_index)
{
	if (num_actors_ == 1)
	{
		for (auto node_iter = search_path.rbegin(); node_iter != search_path.rend(); ++node_iter)
		{
			auto node = *node_iter;
			node->value_sum_ += value;
			++node->visit_count_;
			torch::Tensor node_value;
			auto node_val = node->get_value();
			if (node_val)
			{
				node_value = *node_val;
			}
			else
			{
				node_value = torch::zeros_like(node->reward_);
			}
			stats_.update((node->reward_ + config_.gamma[0] * node_value).sum().item<float>());

			value = node->reward_ + config_.gamma[0] * value;
		}
	}
	else if (num_actors_ == 2)
	{
		for (auto node_iter = search_path.rbegin(); node_iter != search_path.rend(); ++node_iter)
		{
			auto node = *node_iter;
			node->value_sum_ += node->turn_index_ == turn_index ? value : -value;
			++node->visit_count_;
			torch::Tensor node_value;
			auto node_val = node->get_value();
			if (node_val)
			{
				node_value = *node_val;
			}
			else
			{
				node_value = torch::zeros_like(node->reward_);
			}
			stats_.update((node->reward_ + config_.gamma[0] * -node_value).sum().item<float>());

			value = (node->turn_index_ == turn_index ? -node->reward_ : node->reward_) + config_.gamma[0] * value;
		}
	}
	else
	{
		throw std::runtime_error("More than 2 actors is not supported yet");
	}
}
