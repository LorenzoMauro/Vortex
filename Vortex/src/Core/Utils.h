#pragma once
#include <string>
#include <filesystem>
#include <fstream>
#include "Core/Log.h"
#include <sstream>
#include <stack>

namespace utl{

	std::string absolutePath(const std::string& relativePath, const std::string& folderPath = "");

	std::string getFileName(const std::string& path);

	std::vector<char> readData(const std::string& FilePath);

	bool saveString(const std::string& filename, const std::string& text);

	std::string getDateTime();

	std::string getFolder(const std::string& path);

	std::string         getFile(const std::string& path);
}
