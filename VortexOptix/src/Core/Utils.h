#pragma once
#include <string>
#include <filesystem>
#include <fstream>
#include "Core/Log.h"


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
}