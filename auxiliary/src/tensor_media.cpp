#include "tensor_media.h"

#include <GifEncoder.h>
#include <spdlog/spdlog.h>

using namespace drla;

TensorImage drla::create_tensor_image(const torch::Tensor& tensor_image, bool invert)
{
	if (tensor_image.numel() == 0)
	{
		return {};
	}

	torch::Tensor rgb_tensor;
	if (tensor_image.dim() == 3)
	{
		if (tensor_image.size(0) == 1)
		{
			rgb_tensor =
				torch::cat({tensor_image[0].unsqueeze(2), tensor_image[0].unsqueeze(2), tensor_image[0].unsqueeze(2)}, -1)
					.detach();
		}
		else if (tensor_image.size(0) == 2)
		{
			rgb_tensor = torch::cat(
										 {tensor_image[0].unsqueeze(2),
											tensor_image[1].unsqueeze(2),
											((tensor_image[0] + tensor_image[1]) / 2).unsqueeze(2)},
										 -1)
										 .detach();
		}
		else if (tensor_image.size(0) == 3)
		{
			rgb_tensor =
				torch::cat({tensor_image[0].unsqueeze(2), tensor_image[1].unsqueeze(2), tensor_image[2].unsqueeze(2)}, -1)
					.detach();
		}
		else
		{
			rgb_tensor = tensor_image;
		}
	}
	else if (tensor_image.dim() == 2)
	{
		rgb_tensor =
			torch::cat({tensor_image.unsqueeze(2), tensor_image.unsqueeze(2), tensor_image.unsqueeze(2)}, -1).detach();
	}

	assert(rgb_tensor.is_contiguous());

	if (rgb_tensor.is_floating_point())
	{
		rgb_tensor = rgb_tensor.clamp(0.0, 1.0).mul_(std::numeric_limits<unsigned char>::max()).to(torch::kUInt8);
	}

	// Invert to make it easier to see
	if (invert)
	{
		rgb_tensor = std::numeric_limits<unsigned char>::max() - rgb_tensor;
	}

	auto ptr = rgb_tensor.data_ptr<unsigned char>();
	return {
		std::vector<unsigned char>(ptr, ptr + (size_t)rgb_tensor.numel()),
		static_cast<int>(rgb_tensor.size(0)),
		static_cast<int>(rgb_tensor.size(1)),
		static_cast<int>(rgb_tensor.size(2))};
}

TensorImage drla::tile_tensor_images(const torch::Tensor& tensor_image)
{
	using namespace torch::indexing;
	assert(tensor_image.dim() == 4);
	auto sz = tensor_image.sizes();
	// make a tile of [C, sqrt(B) * H, sqrt(B) * W]
	int tile_len = std::ceil(std::sqrt(sz[0]));
	auto image = torch::zeros({sz[1], tile_len * sz[2], tile_len * sz[3]});
	for (int i = 0; i < sz[0]; ++i)
	{
		int y = (i / tile_len) * sz[2];
		int x = (i % tile_len) * sz[3];
		image.index({Ellipsis, Slice(y, y + sz[2]), Slice(x, x + sz[3])}) = tensor_image[i];
	}
	return create_tensor_image(image);
}

bool drla::save_gif_animation(const std::filesystem::path& path, const torch::Tensor& image_sequence, int speed)
{
	// Dims should be [B, S, C, H, W]
	if (image_sequence.dim() < 5)
	{
		spdlog::error("Invalid image dimensions. Expected 5 [B, S, C, H, W], got {}", image_sequence.dim());
		throw std::runtime_error("Invalid image dimensions.");
	}

	GifEncoder gif_enc;
	const auto sz = image_sequence.sizes();
	const int b = sz[0];									 // batch size
	const int s = sz[1];									 // sequence size
	const int c = std::min<int>(sz[2], 3); // channels
	const int h = sz[3];									 // height
	const int w = sz[4];									 // width
	if (gif_enc.open(path, w, h, 10, true, 0, 3 * b * s * h * w))
	{
		auto batched_img_seq = image_sequence.cpu();
		for (int i = 0; i < sz[1]; ++i)
		{
			// Tile the batch dim
			using namespace torch::indexing;
			auto img = tile_tensor_images(batched_img_seq.index({Slice(), i, Slice(0, c)}));
			gif_enc.push(GifEncoder::PIXEL_FORMAT_RGB, img.data.data(), img.width, img.height, speed);
		}
		gif_enc.close();
		return true;
	}
	else
	{
		spdlog::error("gif encoder error: could not create temporary file '{}'\n", path.string());
		return false;
	}
}

bool drla::save_gif_animation(const std::filesystem::path& path, const std::vector<torch::Tensor>& images, int speed)
{
	// Dims for should be [C, H, W]
	if (!images.empty() && images.front().dim() < 3)
	{
		spdlog::error("Invalid image dimensions. Expected 3 [C, H, W], got {}", images.front().dim());
		throw std::runtime_error("Invalid image dimensions.");
	}

	GifEncoder gif_enc;
	const auto sz = images.front().sizes();
	const int w = sz[1];
	const int h = sz[0];
	if (gif_enc.open(path, w, h, 10, true, 0, 3 * w * h * images.size()))
	{
		for (auto& image : images)
		{
			auto img = create_tensor_image(image.cpu());
			gif_enc.push(GifEncoder::PIXEL_FORMAT_RGB, img.data.data(), img.width, img.height, speed);
		}
		gif_enc.close();
		return true;
	}
	else
	{
		spdlog::error("gif encoder error: could not create file '{}'\n", path.string());
		return false;
	}
}
