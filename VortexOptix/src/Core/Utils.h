#pragma once
#include <string>
#include <filesystem>
#include <fstream>
#include "Core/Log.h"
#include <sstream>

namespace utl{
	static std::string absolutePath(const std::string& relative_path) {
		std::filesystem::path path = relative_path;
		std::filesystem::path absPath = std::filesystem::absolute(path);
		return absPath.string();
	}

	static std::vector<char> readData(const std::string& FilePath) {
		std::ifstream inputData(FilePath, std::ios::binary);

		if (inputData.fail())
		{
			VTX_ERROR("ERROR: readData() Failed to open file {}", FilePath);
			return std::vector<char>();
		}

		// Copy the input buffer to a char vector.
		std::vector<char> data(std::istreambuf_iterator<char>(inputData), {});

		if (inputData.fail())
		{
			VTX_ERROR("ERROR: readData() Failed to read file {}", FilePath);
			return std::vector<char>();
		}

		return data;
	}

	bool static saveString(const std::string& filename, const std::string& text)
	{
		std::ofstream outputStream(filename);

		if (!outputStream)
		{
			std::cerr << "ERROR: saveString() Failed to open file " << filename << '\n';
			return false;
		}

		outputStream << text;

		if (outputStream.fail())
		{
			std::cerr << "ERROR: saveString() Failed to write file " << filename << '\n';
			return false;
		}

		return true;
	}

	static std::string getDateTime()
	{
		SYSTEMTIME time;
		GetLocalTime(&time);

		std::ostringstream oss;

		oss << time.wYear;
		if (time.wMonth < 10)
		{
			oss << '0';
		}
		oss << time.wMonth;
		if (time.wDay < 10)
		{
			oss << '0';
		}
		oss << time.wDay << '_';
		if (time.wHour < 10)
		{
			oss << '0';
		}
		oss << time.wHour;
		if (time.wMinute < 10)
		{
			oss << '0';
		}
		oss << time.wMinute;
		if (time.wSecond < 10)
		{
			oss << '0';
		}
		oss << time.wSecond << '_';
		if (time.wMilliseconds < 100)
		{
			oss << '0';
		}
		if (time.wMilliseconds < 10)
		{
			oss << '0';
		}
		oss << time.wMilliseconds;

		return oss.str();
	}

}