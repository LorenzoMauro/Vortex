#pragma once
#include <string>
#include <filesystem>
#include <fstream>
#include "Core/Log.h"
#include <sstream>
#include <stack>

namespace utl{

	std::string absolutePath(const std::string& relative_path);

	std::string getFileName(const std::string& path);

	std::vector<char> readData(const std::string& FilePath);

	bool saveString(const std::string& filename, const std::string& text);

	std::string getDateTime();
}
