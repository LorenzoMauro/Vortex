#pragma once
#include <memory>

namespace vtx::graph
{
	class Renderer;
}

namespace vtx::gui
{
	void rendererNodeGui(const std::shared_ptr<graph::Renderer>& renderNode);
}
