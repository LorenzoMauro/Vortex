#pragma once
#include <string>
#include "Core/Utils.h"
#include "Core/Math.h"

namespace vtx
{
	class Image
	{
	public:
		Image() = default;

		Image(const std::string& path);

		Image(const float* _data, const int _width, const int _height, const int _channels);

		void load(const std::string& path);

		void load(const float* _data, const int _width, const int _height, const int _channels);

		void save(const std::string& path);

		int getWidth() const;

		int getHeight() const;

		int getChannels() const;

		math::vec2f getSize() const;

		float* getData();

	private:
		bool saveJpeg(const std::string& path);

		bool saveBmp(const std::string& path);

		bool savePng(const std::string& path);

		bool saveHdr(const std::string& path);

		static void convertToUint8(const std::vector<float>& image, std::vector<uint8_t>& uiData);

		std::vector<float> flipImageVertically();

		std::vector<float> data ={};
		int width = 0;
		int height = 0;
		int channels = 0;
	};
}