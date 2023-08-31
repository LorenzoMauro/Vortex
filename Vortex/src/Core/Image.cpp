#include "Image.h"
#include "stb_image.h"
#include "stb_image_write.h"
#include <cstring>

namespace vtx
{
	Image::Image(const std::string& path)
	{
		load(path);
	}

	Image::Image(const float* _data, const int _width, const int _height, const int _channels)
	{
		load(_data, _width, _height, _channels);
	}

	void Image::load(const std::string& path)
	{
		stbi_set_flip_vertically_on_load(1);
		float* tmpData = stbi_loadf(path.c_str(), &width, &height, &channels, 0);
		data.resize(width * height * channels);
		data.assign(tmpData, tmpData + width * height * channels);
	}

	void Image::load(const float* _data, const int _width, const int _height, const int _channels)
	{
		width = _width;
		height = _height;
		channels = _channels;
		data.resize(width * height * channels);
		data.assign(_data, _data + width * height * channels);
	}

	void Image::save(const std::string& path)
	{
		if (const std::string extension = utl::getFileExtension(path); extension == "jpg")
			saveJpeg(path);
		else if (extension == "bmp")
			saveBmp(path);
		else if (extension == "png")
			savePng(path);
		else if (extension == "hdr")
			saveHdr(path);
		else
			VTX_ERROR("Image::save: Unknown extension: " + extension);
		return;
	}
	int Image::getWidth() const
	{
		return width;
	}
	int Image::getHeight() const
	{
		return height;
	}
	int Image::getChannels() const
	{
		return channels;
	}
	math::vec2f Image::getSize() const
	{
		return math::vec2f(width, height);
	}

	float* Image::getData()
	{
		if (!data.empty())
		{
			return data.data();
		}
		return nullptr;
	}

	void Image::saveJpeg(const std::string& path)
	{
		const std::vector<float> flippedImage = flipImageVertically();
		std::vector<uint8_t> uiData;
		convertToUint8(flippedImage, uiData);
		stbi_write_jpg(path.c_str(), width, height, channels, uiData.data(), 100);
	}
	void Image::saveBmp(const std::string& path)
	{
		const std::vector<float> flippedImage = flipImageVertically();
		std::vector<uint8_t> uiData;
		convertToUint8(flippedImage, uiData);
		stbi_write_bmp(path.c_str(), width, height, channels, uiData.data());
	}
	void Image::savePng(const std::string& path)
	{
		const std::vector<float> flippedImage = flipImageVertically();
		std::vector<uint8_t> uiData;
		convertToUint8(flippedImage, uiData);
		stbi_write_png(path.c_str(), width, height, channels, uiData.data(), width * channels);
	}
	void Image::saveHdr(const std::string& path)
	{
		const std::vector<float> flippedImage = flipImageVertically();
		stbi_write_hdr(path.c_str(), width, height, channels, flippedImage.data());
	}
	void Image::convertToUint8(const std::vector<float>& image, std::vector<uint8_t>& uiData)
	{
		// Create a vector to hold the 8-bit data
		// Convert the floating point data to 8-bit data
		if (uiData.size() != image.size())
		{
			uiData.resize(image.size());
		}
		for (int i = 0; i < image.size(); ++i) {
			uiData[i] = static_cast<uint8_t>(image[i] * 255.0f);
		}
	}
	std::vector<float> Image::flipImageVertically() {

		std::vector<float> flippedData(width * height * channels);

		for (int y = 0; y < height; ++y) {
			for (int x = 0; x < width; ++x) {
				for (int c = 0; c < channels; ++c) {
					const int oldIndex = (y * width + x) * channels + c;
					const int newIndex = ((height - y - 1) * width + x) * channels + c;
					flippedData[newIndex] = data[oldIndex];
				}
			}
		}

		return flippedData;
	}
}
