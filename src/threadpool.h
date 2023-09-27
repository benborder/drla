#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <thread>
#include <type_traits>
#include <vector>

namespace drla
{

/// @brief A minimal threadpool implementation
class ThreadPool
{
public:
	/// @brief Initialises the threadpool with the number of threads specified, blocking until all threads are created and
	/// waiting to run tasks.
	/// @param threads The number of threads use. If 0 the number threads will be set to the number of cores available.
	/// @param clamp_threads If set to true, the number of threads will be clamped to the number of cores available.
	/// @param max_queue_length The maximum allowed queue length. Adding further tasks to the queue will block
	ThreadPool(size_t threads = 0, bool clamp_threads = true, size_t max_queue_length = 0)
			: pending_(0), running_(false), max_queue_length_(max_queue_length)
	{
		if (threads == 0)
		{
			threads = std::thread::hardware_concurrency();
		}
		else if (clamp_threads)
		{
			threads = std::min<size_t>(std::thread::hardware_concurrency(), threads);
		}

		std::atomic_uint running_threads = 0;

		running_ = true;
		for (size_t i = 0; i < threads; ++i)
		{
			threads_.emplace_back([this, &running_threads]() {
				++running_threads;
				while (running_)
				{
					std::function<void()> task;
					// Wait for a task to become available in the queue
					{
						std::unique_lock lock(m_tasks_);
						cv_tasks_.wait(lock, [this]() { return !tasks_.empty() || !running_; });
						if (tasks_.empty() || !running_)
						{
							continue;
						}
						task = std::move(tasks_.front());
						tasks_.pop_front();
					}

					// Execute the task
					task();

					// Notify all threads waiting for tasks to be completed
					{
						std::unique_lock lock(m_signal_);
						--pending_;
						cv_signal_.notify_all();
					}
				}
			});
		}

		// block via spinlock while all threads are starting up
		while (running_threads < threads)
			;
	}

	/// @brief Stops all threads in the pool, clears the task queue and blocks until all threads have terminated.
	~ThreadPool()
	{
		running_ = false;
		clear_queued();
		for (auto& thread : threads_) { thread.join(); }
	}

	// Disable copy constructors and operators.
	ThreadPool(const ThreadPool&) = delete;
	ThreadPool(ThreadPool&) = delete;
	ThreadPool& operator=(const ThreadPool&) = delete;
	ThreadPool& operator=(ThreadPool&) = delete;

	/// @brief Dispatch a task to the queue and notify the pool a task is available. If enabled, this will block if the
	/// max queue length is exceeded.
	/// @param func The function to execute on a thread in the pool.
	template <typename Func, typename = std::enable_if_t<std::is_void_v<std::invoke_result_t<std::decay_t<Func>>>>>
	void queue_task(Func&& func)
	{
		std::unique_lock lock(m_tasks_);
		if (max_queue_length_ > 0 && tasks_.size() >= max_queue_length_)
		{
			cv_signal_.wait(lock, [&]() { return tasks_.size() < max_queue_length_; });
		}
		tasks_.push_back(std::move(func));
		++pending_;
		cv_tasks_.notify_one();
	}

	/// @brief Dispatch a task with a future return value to the queue and notify the pool a task is available. If
	/// enabled, this will block if the max queue length is exceeded.
	/// @param func The function to execute on a thread in the pool.
	/// @return A future of the of the return type of the task
	template <
		typename Func,
		typename Res = std::invoke_result_t<std::decay_t<Func>>,
		typename = std::enable_if_t<!std::is_void_v<Res>>>
	[[nodiscard]] std::future<Res> queue_task(Func&& func)
	{
		auto task_promise = std::make_shared<std::promise<Res>>();
		queue_task([func = std::move(func), task_promise] {
			try
			{
				task_promise->set_value(func());
			}
			catch (...)
			{
				task_promise->set_exception(std::current_exception());
			}
		});
		return task_promise->get_future();
	}

	/// @brief Wait until the task queue is empty.
	void wait_queue_empty()
	{
		std::unique_lock lock(m_signal_);
		cv_signal_.wait(lock, [&]() { return pending_ == 0; });
	}

	/// @brief Clears all tasks currently queued.
	void clear_queued()
	{
		std::lock_guard lock(m_tasks_);
		pending_ -= tasks_.size();
		tasks_.clear();
		cv_tasks_.notify_all();
	}

	/// @brief returns the number of threads currently in the pool.
	[[nodiscard]] size_t size() const { return threads_.size(); }

private:
	std::mutex m_tasks_;
	std::condition_variable cv_tasks_;
	std::deque<std::function<void()>> tasks_;

	std::mutex m_signal_;
	std::condition_variable cv_signal_;
	std::atomic_uint pending_;

	std::atomic_bool running_;
	std::vector<std::thread> threads_;

	const size_t max_queue_length_;
};

} // namespace drla
