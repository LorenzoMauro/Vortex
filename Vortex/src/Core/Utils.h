#pragma once
#include <string>
#include <filesystem>
#include <fstream>
#include "Core/Log.h"
#include <sstream>
#include <stack>

namespace utl{

	std::string replaceFunctionNameInPTX(const std::string& ptxCode,
		const std::string& oldFunctionName,
		const std::string& newFunctionName);

	std::vector<std::string> splitString(const std::string& input, const std::string separator);

	std::string absolutePath(const std::string& relativePath, const std::string& folderPath = "");

	std::string getFileName(const std::string& path);

	std::string getFileExtension(const std::string& path);

	std::vector<char> readData(const std::string& FilePath);

	bool saveString(const std::string& filename, const std::string& text);

	std::string getDateTime();

	std::string getFolder(const std::string& path);

	std::string         getFile(const std::string& path);

	bool binaryDump(void* data, const size_t& size, const std::string& filePath);



	
	template<typename T>
	std::vector<T> binaryLoad(const int& count, const std::string & filePath)
	{
		std::vector<T> data(count);

		std::ifstream inFile(filePath, std::ios::binary);
		if (!inFile)
		{
			VTX_ERROR("ERROR: binaryLoad() Failed to open file {}", filePath);
			return std::vector<T>();  // return an empty vector on failure
		}
		inFile.read(reinterpret_cast<char*>(data.data()), count * sizeof(T));

		if (!inFile.good())
		{
			VTX_ERROR("ERROR: binaryLoad() Failed to read from file {}", filePath);
			return std::vector<T>();  // return an empty vector on failure
		}

		return data;
	}
}
