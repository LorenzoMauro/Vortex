#include "InteropWrapper.h"
#include <cudaGL.h>
#include "CUDAChecks.h"
#include "OptixWrapper.h"

namespace vtx
{

	void InteropWrapper::prepare(const int width, const int height, const InteropUsage newUsage)
	{
		if (width != (int)glFrameBuffer.width || height != (int)glFrameBuffer.height || newUsage != usage)
		{
			const bool regenerateGlBuffer = (newUsage != usage) ? true : false;
			glFrameBuffer.setSize(width, height, regenerateGlBuffer);
			glFrameBuffer.bind();
			glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
			glClear(GL_COLOR_BUFFER_BIT);
			glFrameBuffer.unbind();

			(cuGraphicResource != nullptr) ? CU_CHECK_CONTINUE(cuGraphicsUnregisterResource(cuGraphicResource)) : void();
			const auto stream = optix::getState()->stream;
			CU_CHECK_CONTINUE(cuGraphicsGLRegisterImage(&cuGraphicResource, glFrameBuffer.colorAttachment, GL_TEXTURE_2D, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD));
			CU_CHECK_CONTINUE(cuGraphicsMapResources(1, &cuGraphicResource, stream));
			CU_CHECK_CONTINUE(cuGraphicsSubResourceGetMappedArray(&cuArray, cuGraphicResource, 0, 0));
			CU_CHECK_CONTINUE(cuGraphicsUnmapResources(1, &cuGraphicResource, stream)); // This is an implicit cuSynchronizeStream().
			usage = newUsage;
		}
	}

	void InteropWrapper::copyToGlBuffer(const CUdeviceptr buffer, const int width, const int height)
	{
		const auto stream = optix::getState()->stream;

		CU_CHECK_CONTINUE(cuGraphicsMapResources(1, &cuGraphicResource, stream));

		CUDA_MEMCPY3D params = {};
		params.srcMemoryType = CU_MEMORYTYPE_DEVICE;
		params.srcDevice = buffer;
		params.srcPitch = width * sizeof(math::vec4f);
		params.srcHeight = height;

		params.dstMemoryType = CU_MEMORYTYPE_ARRAY;
		params.dstArray = cuArray;
		params.WidthInBytes = width * sizeof(math::vec4f);
		params.Height = height;
		params.Depth = 1;

		CU_CHECK_CONTINUE(cuMemcpy3D(&params)); // Copy from linear to array layout.

		CU_CHECK_CONTINUE(cuGraphicsUnmapResources(1, &cuGraphicResource, stream)); // This is an implicit cuSynchronizeStream().

	}
}
