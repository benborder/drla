#pragma once

#include <torch/torch.h>

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

} // namespace drla
