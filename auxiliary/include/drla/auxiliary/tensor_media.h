#pragma once

#include <ATen/core/Tensor.h>
#include <torch/types.h>

#include <filesystem>
#include <vector>

namespace drla
{

struct TensorImage
{
	std::vector<unsigned char> data;
	int height;
	int width;
	int channels;
};

/// @brief Converts a tensor to an 8bit RGB image. Supports 1-3 channels and automatically handles channel first or last
/// dim orders.
/// @param tensor The tensor to convert to an image
/// @param invert Inverts the image (i.e. 255 - x)
/// @return A tensor image object containing the raw 8bit RGB data
TensorImage create_tensor_image(const torch::Tensor& tensor, bool invert = false);

/// @brief Tile the batch dim and create a single image
/// @param tensor_image It is assumed the dims are [B, C, H, W]
/// @return An output image of [C, sqrt(B) * H, sqrt(B) * W]
TensorImage tile_tensor_images(const torch::Tensor& tensor);

/// @brief Creates a single animation from a sequence of images and saves them to a gif file.
/// @param images The sequence of images to create an animation from. Must have dimensions [Batch, Sequence, Channel,
/// Height, Width]. If the batch size is greater than 1, the batch dim is tiled.
/// @param speed The speed the animation runs at.
/// @return true if the animation was succesfully saved to file, false otherwise.
bool save_gif_animation(const std::filesystem::path& path, const torch::Tensor& image_sequence, int speed);

/// @brief Creates a single animation from a sequence of images and saves them to a gif file.
/// @param images A vector of images to create an animation from. Each tensor in the vector must have dimensions
/// [Height, Width, Channel].
/// @param speed The speed the animation runs at.
/// @return true if the animation was succesfully saved to file, false otherwise.
bool save_gif_animation(const std::filesystem::path& path, const std::vector<torch::Tensor>& images, int speed);

} // namespace drla
