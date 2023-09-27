#pragma once

#include <drla/callback.h>
#include <drla/stats.h>

#include <chrono>
#include <string>

class TensorBoardLogger;

namespace drla
{

/// @brief Logs various metrics for training to print to the terminal and save to tensorboard
class TrainingMetricsLogger
{
public:
	/// @brief Constructs the TrainingMetricsLogger
	/// @param path The path to create the tensorboard log file
	/// @param resume Attempt to resume logging to an existing tensorboard log file in the provided `path`.
	TrainingMetricsLogger(const std::filesystem::path& path, bool resume);
	~TrainingMetricsLogger();

	/// @brief Initialises the logger, resetting any existing metrics
	/// @param total_timesteps The total timesteps training will be conducted over.
	void init(int total_timesteps);

	/// @brief Updates the metrics with new timestep data, saving to tensorboard.
	/// @param timestep_data The data from a training timestep
	void update(const TrainUpdateData& timestep_data);

	/// @brief Prints the metrics to terminal
	/// @param timestep_data The data from a training timestep
	/// @param total_episode_count The total number of episodes completed in training
	void print(const TrainUpdateData& timestep_data, int total_episode_count) const;

	/// @brief Logs a scalar metric for the current timestep
	/// @param group The group in tensorboard
	/// @param name The name used to print to terminal and log in tensorboard
	/// @param value The scalar value to log
	void add_scalar(std::string group, std::string name, double value);

	/// @brief Logs an image to tensorboard
	/// @param group The group in tensorboard
	/// @param name The name in tensorboard
	/// @param image The image to save to tensorboard. Must have dimensions [Channel, Height, Width].
	void add_image(std::string group, std::string name, torch::Tensor image);

	/// @brief Creates a single animation from a sequence of images and logs them to tensorboard.
	/// @param group The group in tensorboard
	/// @param name The name in tensorboard
	/// @param images The sequence of images to create an animation from. Must have dimensions [Batch, Sequence, Channel,
	/// Height, Width]. If the batch size is greater than 1, the batch dim is tiled.
	/// @param speed The speed the animation runs at.
	void add_animation(std::string group, std::string name, torch::Tensor images, int speed = 10);

	/// @brief Creates a single animation from a sequence of images and logs them to tensorboard.
	/// @param group The group in tensorboard
	/// @param name The name in tensorboard
	/// @param images A vector of images to create an animation from. Each tensor in the vector must have dimensions
	/// [Height, Width, Channel].
	/// @param speed The speed the animation runs at.
	void add_animation(std::string group, std::string name, const std::vector<torch::Tensor>& images, int speed = 2);

protected:
	std::chrono::steady_clock::time_point start_time_;

	std::unique_ptr<TensorBoardLogger> tb_logger_;

	std::map<std::string, Stats<double>> metrics_;

	int current_timestep_ = 0;
	int total_timesteps_ = 0;
	Stats<double> fps_stats_;
	Stats<double> train_time_stats_;
	Stats<double> env_time_stats_;

	std::filesystem::path gif_path_;
};

} // namespace drla
