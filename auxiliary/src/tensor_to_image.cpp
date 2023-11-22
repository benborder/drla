#include "tensor_to_image.h"

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
