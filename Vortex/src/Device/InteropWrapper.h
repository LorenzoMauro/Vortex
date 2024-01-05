#pragma once
#include <cuda.h>
#include "glFrameBuffer.h"

namespace vtx
{
	enum InteropUsage
	{
		SingleThreaded,
		MultiThreaded
	};
	struct InteropWrapper
	{
	public:
		GlFrameBuffer glFrameBuffer;
		CUarray cuArray = nullptr;
		CUgraphicsResource cuGraphicResource = nullptr;
		InteropUsage usage = SingleThreaded;

		void prepare(const int width, const int height, const int nChannels, const InteropUsage newUsage);

		void copyToGlBuffer(const CUdeviceptr buffer, const int width, const int height);
	};
}
