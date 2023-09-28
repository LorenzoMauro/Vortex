#include "UploadBuffers.h"

namespace vtx
{
	device::Buffers* vtx::device::Buffers::getInstance()
	{
		static Buffers buffersInstance;
		return &buffersInstance;
	}
	void device::Buffers::shutDown()
	{
		VTX_INFO("Shutting Down Buffers");
		//frameIdBuffer.free();
		//launchParamsBuffer.free();
		//rendererSettingsBuffer.free();
		//sbtProgramIdxBuffer.free();
		//instancesBuffer.free();
		//toneMapperSettingsBuffer.free();
	}
}
