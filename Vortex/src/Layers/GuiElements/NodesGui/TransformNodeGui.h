#pragma once
#include <memory>

namespace vtx
{
	namespace graph
	{
		class Transform;
	}
}

namespace vtx::gui
{
	bool transformNodeGui(const std::shared_ptr<graph::Transform>& transformNode);
}
