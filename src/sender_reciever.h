#pragma once

#include <condition_variable>
#include <deque>
#include <optional>
#include <vector>

namespace drla
{

/// @brief The overflow behaviour options.
enum class OverflowBehaviour
{
	/// @brief When the queue overflows the head of the queue is dropped (popping the front)
	kDropHead,
	/// @brief When the queue overflows the tail of the queue is dropped (the new data to add is dropped)
	kDropTail,
};

template <class>
class Sender;

/// @brief Recieves data from an associated sender
/// @tparam T The type of the data to send. Must be movable.
template <typename T>
class Reciever
{
	friend class Sender<T>;

public:
	Reciever() = default;

	/// @brief Constructor
	/// @param max_queue_size The maximum size of the queue to store sent data.
	/// @param overflow_behaviour The overflow behaviour of the queue. See @see @ref OverflowBehaviour.
	Reciever(size_t max_queue_size, OverflowBehaviour overflow_behaviour)
			: overflow_behaviour_(overflow_behaviour), max_queue_size_(max_queue_size)
	{
	}

	~Reciever() { stop(); }

	/// @brief Request data to be sent, waits indefinately for data to be sent.
	/// @return The data that was sent.
	[[nodiscard]] T request()
	{
		T data;
		while (run_)
		{
			std::unique_lock lock(m_cv_);
			request_data_ = true;
			cv_.wait(lock, [this]() { return !queue_.empty(); });
			if (queue_.empty())
			{
				continue;
			}
			data = std::move(queue_.front());
			queue_.pop_front();
			request_data_ = false;
			break;
		}
		return data;
	}

	/// @brief Requests and waits for data to be sent, will stop waiting based on the criteria function
	/// @param criteria A function which returns a boolean. True if waiting should stop, false if it should continue.
	/// @return The data that was sent.
	template <typename Func>
	[[nodiscard]] std::optional<T> wait(Func criteria)
	{
		std::unique_lock lock(m_cv_);
		request_data_ = true;
		cv_.wait(lock, [&]() { return !queue_.empty() && criteria(); });
		if (queue_.empty())
		{
			return std::nullopt;
		}
		auto data = std::move(queue_.front());
		queue_.pop_front();
		request_data_ = false;

		return data;
	}

	/// @brief Checks if there is data available on the queue and returns it, otherwise returns nullopt.
	/// @return The data from the queue or nullopt if no data is available.
	[[nodiscard]] std::optional<T> check()
	{
		std::unique_lock lock(m_cv_);
		if (queue_.empty())
		{
			return std::nullopt;
		}
		auto data = std::move(queue_.front());
		queue_.pop_front();
		return data;
	}

	/// @brief Clears the queue
	void clear()
	{
		std::lock_guard lock(m_cv_);
		queue_.clear();
	}

private:
	/// @brief Adds the data to the recievers queue, following the queue behaviour assigned at construction.
	/// @param data The data to add to the queue. Must be moveable.
	void enqueue(const T& data)
	{
		std::lock_guard lock(m_cv_);
		request_data_ = false;
		if (overflow_behaviour_ == OverflowBehaviour::kDropHead)
		{
			queue_.push_back(data);
			while (queue_.size() > max_queue_size_) { queue_.pop_front(); }
		}
		else if (queue_.size() < max_queue_size_) // DropTail
		{
			queue_.push_back(data);
		}
		cv_.notify_all();
	}

	/// @brief Stops the reciever if currently waiting, clearing the queue.
	void stop()
	{
		run_ = false;
		clear();
		cv_.notify_all();
	}

	/// @brief Indicates if the reciever has an unanswered request for new data
	/// @return True if there is an unanswered request, false otherwise
	bool is_data_requested() const { return request_data_; }

private:
	const OverflowBehaviour overflow_behaviour_;
	const size_t max_queue_size_ = 1;

	bool run_ = true;
	bool request_data_ = false;
	std::deque<T> queue_;
	std::mutex m_cv_;
	std::condition_variable cv_;
};

/// @brief Sends data to recievers, typically on different threads
/// @tparam T The type of the data to send. Must be movable.
template <typename T>
class Sender
{
public:
	/// @brief Creates recievers that this sender can send to
	/// @param queue_size The size of the reciever queue.
	/// @param overflow_behaviour If the queue size is exceeded, the queue is clamped according to the overflow behaviour,
	/// where either the head or tail are dropped.
	/// @return The reciever object
	[[nodiscard]] std::shared_ptr<Reciever<T>>
	create_reciever(size_t queue_size = 1, OverflowBehaviour overflow_behaviour = OverflowBehaviour::kDropHead)
	{
		std::lock_guard lock(m_recievers_);
		auto reciever = std::make_shared<Reciever<T>>(queue_size, overflow_behaviour);
		recievers_.push_back(reciever);
		return reciever;
	}

	/// @brief Sends the data to registered recievers, creating coppies of the data for each reciever.
	/// @param data The data to send to recievers.
	/// @param send_all Sends to all recievers regardless if they requested an update or not
	void send(const T& data, bool send_all = true)
	{
		std::lock_guard lock(m_recievers_);
		for (auto reciever_iter = recievers_.begin(); reciever_iter != recievers_.end();)
		{
			if (reciever_iter->expired())
			{
				reciever_iter = recievers_.erase(reciever_iter);
			}
			else
			{
				auto recv = reciever_iter->lock();
				if (send_all || recv->is_data_requested())
				{
					recv->enqueue(data);
				}
				++reciever_iter;
			}
		}
	}

	/// @brief Returns true if there are any pending requests.
	/// @return True if there are pending requests false otherwise.
	bool has_requests() const
	{
		for (const auto& reciever : recievers_)
		{
			if (!reciever.expired())
			{
				auto recv = reciever.lock();
				if (recv->is_data_requested())
				{
					return true;
				}
			}
		}
		return false;
	}

public:
	std::vector<std::weak_ptr<Reciever<T>>> recievers_;
	std::mutex m_recievers_;
};

} // namespace drla
