#include "Utils.h"
#include <windows.h>
#include <filesystem>

namespace utl{


	std::string replaceFunctionNameInPTX(const std::string& ptxCode,
		const std::string& oldFunctionName,
		const std::string& newFunctionName) {
		std::string result = ptxCode;
		size_t pos = 0;
		while ((pos = result.find(oldFunctionName, pos)) != std::string::npos) {
			result.replace(pos, oldFunctionName.length(), newFunctionName);
			pos += newFunctionName.length();
		}
		return result;
	}

	std::vector<std::string> splitString(const std::string& input, const std::string separator) {
		std::vector<std::string> result;
		size_t current, previous = 0;
		current = input.find(separator);
		while (current != std::string::npos) {
			result.push_back(input.substr(previous, current - previous));
			previous = current + separator.length();
			current = input.find(separator, previous);
		}
		result.push_back(input.substr(previous, current - previous));
		return result;
	}

	std::string absolutePath(const std::string& relativePath, const std::string& folderPath) {

		const std::filesystem::path path = relativePath;
		std::filesystem::path absPath;
		if (path.is_relative())
		{
			if (folderPath.empty())
			{
				absPath = std::filesystem::absolute(path);
			}
			else
			{
				const std::filesystem::path folder(folderPath);
				absPath = folder / relativePath;
			}
		}
		else
		{
			absPath = std::filesystem::absolute(path);
		}
		return absPath.string();
	}

	std::string replacePercent20WithSpace(const std::string& inputPath) {
		std::string newPath = inputPath;
		size_t index = 0;
		while (true) {
			index = newPath.find("%20", index);
			if (index == std::string::npos) break;
			newPath.replace(index, 3, " ");
			index += 1;
		}
		return newPath;
	}

	std::string getFolder(const std::string& path)
	{
		std::string folder = absolutePath(path);
		const std::filesystem::path p(path);
		folder = p.parent_path().string();
		return folder;
	}

	std::string getFile(const std::string& path)
	{
		std::string fileName = absolutePath(path);
		// Remove the path
		const std::filesystem::path p(path);
		fileName = p.filename().string();

		std::string result = fileName;

		if (!result.empty() && (result.back() == '/' || result.back() == '\\'))
		{
			result.pop_back();
		}

		return result;
	}

	std::string getFileName(const std::string& path)
	{
		std::string fileName = getFile(path);

		// Remove the file extension
		const std::size_t last_dot = fileName.find_last_of('.');
		if (last_dot != std::string::npos) {
			fileName.erase(last_dot);
		}

		return fileName;
	}

	std::string getFileExtension(const std::string& path)
	{
		std::string fileName = getFile(path);
		// Remove the file name
		const std::size_t last_dot = fileName.find_last_of('.');
		if (last_dot != std::string::npos)
		{
			fileName.erase(0, last_dot + 1);
		}
		return fileName;
	}


	std::vector<char> readData(const std::string& FilePath) {
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

	bool saveString(const std::string& filename, const std::string& text)
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

	bool binaryDump(void* data, const size_t& size, const std::string& filePath)
	{
		std::ofstream outFile(filePath, std::ios::binary);

		if (!outFile) {
			VTX_ERROR("ERROR: binaryDump() Failed to open file {}", filePath);
			return false;
		}

		outFile.write((char*)data, size);

		outFile.close();

		return true;
	}

	void createDirectory(const std::string& path)
	{
		std::string dir = getFolder(path);
		if (!std::filesystem::exists(dir)) {
			std::filesystem::create_directories(dir);
		}
	}

	std::string getExecutablePath()
	{
		char buffer[MAX_PATH];
		GetModuleFileName(NULL, buffer, MAX_PATH);
		std::string::size_type pos = std::string(buffer).find_last_of("\\/");
		return std::string(buffer).substr(0, pos);
	}

	void copyFileToDestination(const std::string& source, const std::string& destination)
	{
		createDirectory(getFolder(source));
		try {
			std::filesystem::copy(source, destination, std::filesystem::copy_options::overwrite_existing);
			std::cout << "File copied successfully.\n";
		}
		catch (std::filesystem::filesystem_error& e) {
			std::cout << "Failed to copy file: " << e.what() << '\n';
		}
	}


	std::string getDateTime()
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

