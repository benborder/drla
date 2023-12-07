#include "metrics_logger.h"

#include "tensor_to_image.h"

#include <GifEncoder.h>
#include <lodepng.h>
#include <spdlog/fmt/bundled/color.h>
#include <spdlog/fmt/chrono.h>
#include <spdlog/fmt/fmt.h>
#include <spdlog/spdlog.h>
#include <tensorboard_logger.h>

using namespace drla;
using namespace spdlog;

inline void print_stats_header(int margin = 15)
{
	fmt::print(
		fmt::bg(fmt::detail::color_type(fmt::rgb(50, 50, 50))) | fmt::emphasis::bold,
		"{:<{}}|{:^15}|{:^15}|{:^15}|{:^15}|",
		"",
		margin,
		"mean",
		"stdev",
		"max",
		"min");
	fmt::print("\n");
}

inline void print_stats(std::string name, const Stats<double>& stats, int margin = 15)
{
	fmt::print(
		"{:<{}}|{:>14g} |{:>14g} |{:>14g} |{:>14g} |\n",
		name,
		margin,
		stats.get_mean(),
		stats.get_stdev(),
		stats.get_max(),
		stats.get_min());
}

// The tensorboard file name must have 'tfevents' in it for tensorboard to be able to find and read the file.
TrainingMetricsLogger::TrainingMetricsLogger(const std::filesystem::path& path, bool resume)
		: tb_logger_(
				std::make_unique<TensorBoardLogger>((path / "tfevents.pb").c_str(), TensorBoardLoggerOptions{}.resume(resume)))
		, gif_path_()
{
	auto tmp_dir = std::filesystem::temp_directory_path();
	std::filesystem::create_directory(tmp_dir);
	gif_path_ = tmp_dir / "tmp.gif";
	if (std::filesystem::exists(gif_path_))
	{
		std::filesystem::remove(gif_path_);
	}
}

TrainingMetricsLogger::~TrainingMetricsLogger()
{
	google::protobuf::ShutdownProtobufLibrary();
}

void TrainingMetricsLogger::init(int total_timesteps)
{
	start_time_ = std::chrono::steady_clock::now();
	total_timesteps_ = total_timesteps;
	metrics_.clear();
}

void TrainingMetricsLogger::update(const TrainUpdateData& timestep_data)
{
	current_timestep_ = timestep_data.timestep;
	fps_stats_.update(timestep_data.fps);
	// Only update the stats if its greater than 0
	const auto train_time = std::chrono::duration_cast<std::chrono::milliseconds>(timestep_data.update_duration).count();
	if (train_time > 0)
	{
		train_time_stats_.update(train_time);
	}
	env_time_stats_.update(std::chrono::duration_cast<std::chrono::milliseconds>(timestep_data.env_duration).count());

	for (const auto& data : timestep_data.metrics.get_update_results())
	{
		std::string tag;
		if (data.type == TrainResultType::kLoss)
		{
			tag = "loss/";
		}
		else
		{
			tag = "train/";
		}
		tag += data.name;
		tb_logger_->add_scalar(tag, current_timestep_, data.value);
	}
}

void TrainingMetricsLogger::print(const TrainUpdateData& timestep_data, int total_episode_count) const
{
	double progress = 100 * static_cast<double>(current_timestep_ + 1) / static_cast<double>(total_timesteps_);

	auto elapsed_seconds = std::chrono::steady_clock::now() - start_time_;
	int total_hours = static_cast<int>(std::chrono::duration_cast<std::chrono::hours>(elapsed_seconds).count());
	int remaining_minutes =
		static_cast<int>(std::chrono::duration_cast<std::chrono::minutes>(elapsed_seconds % std::chrono::hours(1)).count());
	int remaining_seconds = static_cast<int>(
		std::chrono::duration_cast<std::chrono::seconds>(elapsed_seconds % std::chrono::minutes(1)).count());

	size_t max_len = 12;
	for (const auto& data : timestep_data.metrics.get_update_results()) { max_len = std::max(max_len, data.name.size()); }

	fmt::print("{:<{}}| {:g} [{:g}]\n", "env_fps", max_len, fps_stats_.get_mean(), timestep_data.fps_env);
	fmt::print("{:<{}}| {:g} ms\n", "env_time", max_len, env_time_stats_.get_mean());
	fmt::print("{:<{}}| {:g} ms\n", "train_time", max_len, train_time_stats_.get_mean());
	fmt::print("{:<{}}| {}:{:02}:{:02}\n", "elapsed_time", max_len, total_hours, remaining_minutes, remaining_seconds);
	fmt::print("{:<{}}| {} / {} [{:.2g}%]\n", "timesteps", max_len, current_timestep_ + 1, total_timesteps_, progress);
	fmt::print("{:<{}}| {}\n", "episodes", max_len, total_episode_count);
	fmt::print("{:<{}}| {}\n", "global_steps", max_len, timestep_data.global_steps);
	for (const auto& data : timestep_data.metrics.get_update_results())
	{
		fmt::print("{:<{}}| {}\n", data.name, max_len, data.value);
	}

	max_len = 15;
	for (auto& [name, stats] : metrics_) { max_len = std::max(max_len, name.size()); }

	print_stats_header(max_len);
	for (auto& [name, stats] : metrics_) { print_stats(name, stats, max_len); }
	fmt::print("{:=<{}}\n", "", 65 + max_len);
}

void TrainingMetricsLogger::add_scalar(std::string group, std::string name, double value)
{
	auto metric = metrics_.find(name);
	if (metric == metrics_.end())
	{
		metric = metrics_.emplace(name, Stats<double>{}).first;
	}
	metric->second.update(value);
	tb_logger_->add_scalar(group + "/" + name, current_timestep_, value);
}

void TrainingMetricsLogger::add_image(std::string group, std::string name, torch::Tensor image)
{
	if (image.dim() != 3)
	{
		spdlog::error("Invalid image dimensions. Expected 3 [C, H, W], got {}", image.dim());
		throw std::runtime_error("Invalid image dimensions.");
	}
	auto channels = std::min<int>(image.size(0), 3);
	auto obs_img = create_tensor_image(image.narrow(0, 0, channels));
	std::vector<unsigned char> png;
	unsigned error = lodepng::encode(png, obs_img.data, obs_img.width, obs_img.height, LCT_RGB);
	if (error != 0)
	{
		spdlog::error("png encoder error {}: {}", error, lodepng_error_text(error));
		return;
	}
	std::string img(std::make_move_iterator(png.begin()), std::make_move_iterator(png.end()));
	tb_logger_->add_image(group + "/" + name, current_timestep_, img, obs_img.height, obs_img.width, 3);
}

void TrainingMetricsLogger::add_animation(std::string group, std::string name, torch::Tensor images, int speed)
{
	// Dims should be [B, S, C, H, W]
	if (images.dim() < 5)
	{
		spdlog::error("Invalid image dimensions. Expected 5 [B, S, C, H, W], got {}", images.dim());
		throw std::runtime_error("Invalid image dimensions.");
	}

	GifEncoder gif_enc;
	const auto sz = images.sizes();
	const int b = sz[0];									 // batch size
	const int s = sz[1];									 // sequence size
	const int c = std::min<int>(sz[2], 3); // channels
	const int h = sz[3];									 // width
	const int w = sz[4];									 // height
	if (gif_enc.open(gif_path_, w, h, 10, true, 0, 3 * b * s * h * w))
	{
		auto batched_img_seq = images.cpu();
		for (int i = 0; i < sz[1]; ++i)
		{
			// Tile the batch dim
			using namespace torch::indexing;
			auto img = tile_tensor_images(batched_img_seq.index({Slice(), i, Slice(0, c)}));
			gif_enc.push(GifEncoder::PIXEL_FORMAT_RGB, img.data.data(), img.width, img.height, speed);
		}
		gif_enc.close();
		// This is a hacky method of writing the gif to a file then reading it back as the library does not support writing
		// to a buffer in memory.
		std::ifstream fin(gif_path_, std::ios::binary);
		std::ostringstream ss;
		ss << fin.rdbuf();
		std::string img = ss.str();
		ss.str("");
		fin.close();
		tb_logger_->add_image(group + "/" + name, current_timestep_, img, h, w, 3);
		std::filesystem::remove(gif_path_);
	}
	else
	{
		spdlog::error("gif encoder error: could not create temporary file '{}'\n", gif_path_.string());
	}
}

void TrainingMetricsLogger::add_animation(
	std::string group, std::string name, const std::vector<torch::Tensor>& images, int speed)
{
	GifEncoder gif_enc;
	const auto sz = images.front().sizes();
	const int w = sz[1];
	const int h = sz[0];
	if (gif_enc.open(gif_path_, w, h, 10, true, 0, 3 * w * h * images.size()))
	{
		for (auto& image : images)
		{
			auto img = create_tensor_image(image.cpu());
			gif_enc.push(GifEncoder::PIXEL_FORMAT_RGB, img.data.data(), img.width, img.height, speed);
		}
		gif_enc.close();
		// This is a hacky method of writing the gif to a file then reading it back as the library does not support writing
		// to a buffer in memory.
		std::ifstream fin(gif_path_, std::ios::binary);
		std::ostringstream ss;
		ss << fin.rdbuf();
		std::string img = ss.str();
		ss.str("");
		fin.close();
		tb_logger_->add_image(group + "/" + name, current_timestep_, img, h, w, 3);
		std::filesystem::remove(gif_path_);
	}
	else
	{
		fmt::print("gif encoder error: could not create file '{}'\n", gif_path_.string());
	}
}
