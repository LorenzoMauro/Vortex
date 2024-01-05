#pragma once
#include <string>

namespace vtx
{
	struct CUDABuffer;
}

namespace vtx::gui
{
	void popUpImageWindow(const std::string& nameIdentifier, CUDABuffer imageBuffer, int width, int height, int nChannels, bool forceRebuild);
}
