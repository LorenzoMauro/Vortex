#pragma once
#include <yaml-cpp/yaml.h>
#include "Core/Math.h"

namespace YAML
{
	template<>
	struct convert<vtx::math::vec3f>
	{
		static Node encode(const vtx::math::vec3f& rhs)
		{
			Node node;
			node.push_back(rhs.x);
			node.push_back(rhs.y);
			node.push_back(rhs.z);
			return node;
		}

		static bool decode(const Node& node, vtx::math::vec3f& rhs)
		{
			if (!node.IsSequence() || node.size() != 3)
			{
				return false;
			}
			rhs.x = node[0].as<float>();
			rhs.y = node[1].as<float>();
			rhs.z = node[2].as<float>();
			return true;
		}
	};
}