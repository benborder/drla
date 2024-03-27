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

/// @brief Receives data from an associated sender
/// @tparam T The type of the data to send. Must be movable.
template <typename T>
class Receiver
{
	friend class Sender<T>;

public:
	Receiver() = default;

	/// @brief Constructor
	/// @param max_queue_size The maximum size of the queue to store sent data.
	/// @param overflow_behaviour The overflow behaviour of the queue. See @see @ref OverflowBehaviour.
	Receiver(
		size_t max_queue_size, OverflowBehaviour overflow_behaviour, std::weak_ptr<std::condition_variable> cv_request)
			: overflow_behaviour_(overflow_behaviour), max_queue_size_(max_queue_size), cv_request_(std::move(cv_request))
	{
	}

	~Receiver()
	{
		// Unblock senders if they are waiting for request data
		request_data();
		stop();
	}

	/// @brief Requests and waits for data to be sent, will stop waiting based on the criteria function
	/// @param criteria A function which returns a boolean. True if waiting should stop, false if it should continue.
	/// @return The data that was sent.
	template <typename Func>
	[[nodiscard]] std::optional<T> wait(Func criteria)
	{
		std::unique_lock lock(m_cv_queue_);
		request_data();
		cv_queue_.wait(lock, [&] { return (!queue_.empty() && criteria()) || !run_; });
		if (queue_.empty())
		{
			return std::nullopt;
		}
		auto data = std::move(queue_.front());
		queue_.pop_front();
		request_data_ = false;

		return data;
	}

	/// @brief Requests and waits for data to be sent, will wait indefinately or until the sender/receiver is stopped.
	/// @return The data that was sent.
	[[nodiscard]] std::optional<T> wait()
	{
		return wait([] { return true; });
	}

	/// @brief Checks if there is data available on the queue and returns it, otherwise returns nullopt.
	/// @return The data from the queue or nullopt if no data is available.
	[[nodiscard]] std::optional<T> check()
	{
		std::unique_lock lock(m_cv_queue_);
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
		std::lock_guard lock(m_cv_queue_);
		queue_.clear();
	}

private:
	/// @brief Adds the data to the receivers queue, following the queue behaviour assigned at construction.
	/// @param data The data to add to the queue. Must be moveable.
	void enqueue(T&& data)
	{
		std::lock_guard lock(m_cv_queue_);
		request_data_ = false;
		if (overflow_behaviour_ == OverflowBehaviour::kDropHead)
		{
			queue_.push_back(std::move(data));
			while (queue_.size() > max_queue_size_) { queue_.pop_front(); }
		}
		else if (queue_.size() < max_queue_size_) // DropTail
		{
			queue_.push_back(std::move(data));
		}
		cv_queue_.notify_all();
	}

	/// @brief Stops the receiver if currently waiting, clearing the queue.
	void stop()
	{
		run_ = false;
		clear();
		cv_queue_.notify_all();
	}

	/// @brief Makes the sender aware that data is requested
	void request_data()
	{
		if (cv_request_.expired())
		{
			stop();
			return;
		}
		request_data_ = true;
		cv_request_.lock()->notify_all();
	}

	/// @brief Indicates if the receiver has an unanswered request for new data
	/// @return True if there is an unanswered request, false otherwise
	bool is_data_requested() const { return request_data_; }

private:
	const OverflowBehaviour overflow_behaviour_;
	const size_t max_queue_size_ = 1;

	std::weak_ptr<std::condition_variable> cv_request_;
	bool run_ = true;
	bool request_data_ = false;
	std::deque<T> queue_;
	std::mutex m_cv_queue_;
	std::condition_variable cv_queue_;
};

/// @brief Sends data to receivers, typically on different threads
/// @tparam T The type of the data to send. Must be movable.
template <typename T>
class Sender
{
	static_assert(std::is_move_constructible<T>::value, "Type T must be move constructible");
	static_assert(std::is_move_assignable<T>::value, "Type T must be move assignable");

public:
	Sender() : cv_requests_(std::make_shared<std::condition_variable>()) {}

	/// @brief Creates receivers that this sender can send to
	/// @param queue_size The size of the receiver queue.
	/// @param overflow_behaviour If the queue size is exceeded, the queue is clamped according to the overflow
	/// behaviour, where either the head or tail are dropped.
	/// @return The receiver object
	[[nodiscard]] std::shared_ptr<Receiver<T>>
	create_receiver(size_t queue_size = 1, OverflowBehaviour overflow_behaviour = OverflowBehaviour::kDropHead)
	{
		std::lock_guard lock(m_receivers_);
		auto receiver = std::make_shared<Receiver<T>>(queue_size, overflow_behaviour, cv_requests_);
		receivers_.push_back(receiver);
		return receiver;
	}

	~Sender() { stop_receivers(); }

	/// @brief Sends the data to registered receivers, creating copies of the data for each receiver.
	/// @param data The data to send to receivers.
	void send(const T& data)
	{
		std::lock_guard lock(m_receivers_);
		for (auto receiver_iter = receivers_.begin(); receiver_iter != receivers_.end();)
		{
			if (receiver_iter->expired())
			{
				receiver_iter = receivers_.erase(receiver_iter);
			}
			else
			{
				auto recv = receiver_iter->lock();
				recv->enqueue(T{data}); // create copy
				++receiver_iter;
			}
		}
	}

	/// @brief Sends the data to first registered receiver that has requested data, moving the data to the receiver.
	/// @param data The data to send to receivers.
	/// @param block When true blocks until a receiver has requested data. If false and no receivers are available, drops
	/// the data
	void send_once(T&& data, bool block)
	{
		if (block)
		{
			std::unique_lock lock(m_requests_);
			cv_requests_->wait(lock, [&] { return has_requests(); });
		}

		std::lock_guard lock(m_receivers_);
		for (auto receiver_iter = receivers_.begin(); receiver_iter != receivers_.end();)
		{
			if (receiver_iter->expired())
			{
				receiver_iter = receivers_.erase(receiver_iter);
			}
			else
			{
				auto recv = receiver_iter->lock();
				if (recv->is_data_requested())
				{
					recv->enqueue(data);
					return;
				}
				++receiver_iter;
			}
		}
	}

	/// @brief Returns true if there are any pending requests.
	/// @return True if there are pending requests false otherwise.
	bool has_requests() const
	{
		std::lock_guard lock(m_receivers_);
		for (const auto& receiver : receivers_)
		{
			if (!receiver.expired())
			{
				auto recv = receiver.lock();
				if (recv->is_data_requested())
				{
					return true;
				}
			}
		}
		return false;
	}

private:
	void stop_receivers()
	{
		std::lock_guard lock(m_receivers_);
		for (const auto& receiver : receivers_)
		{
			if (!receiver.expired())
			{
				receiver.lock()->stop();
			}
		}
	}

private:
	std::mutex m_requests_;
	std::shared_ptr<std::condition_variable> cv_requests_;
	std::vector<std::weak_ptr<Receiver<T>>> receivers_;
	std::mutex m_receivers_;
};

} // namespace drla
