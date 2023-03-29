#pragma once
#include <string>
#include <filesystem>


namespace utl{
	static std::string absolutePath(char* relative_path) {
		std::filesystem::path path = relative_path;
		std::filesystem::path absPath = std::filesystem::absolute(path);
		return absPath.string();
	}
}