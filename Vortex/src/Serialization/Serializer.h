#pragma once
#include <string>

namespace vtx::serializer
{
	bool deserialize(const std::string& filePath, bool importScene = false);

	void serialize(const std::string& filePath);
}
