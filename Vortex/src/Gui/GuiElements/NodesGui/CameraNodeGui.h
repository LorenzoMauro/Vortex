#pragma once
#include <memory>

namespace vtx
{
	namespace graph
	{
		class Camera;
	}
}

namespace vtx::gui
{
	bool cameraNodeGui(const std::shared_ptr<graph::Camera>& transformNode);
}
