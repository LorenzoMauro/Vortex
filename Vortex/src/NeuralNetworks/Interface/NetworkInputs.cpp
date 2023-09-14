#include "NetworkInputs.h"

namespace vtx
{
	NetworkInput* vtx::NetworkInput::upload(const int& maxSize, device::Buffers::NetworkInputBuffers& buffers)
	{
		const NetworkInput networkInput(maxSize, buffers);
		return buffers.networkStateStructBuffer.upload(networkInput);
	}

	NetworkInput* vtx::NetworkInput::getPreviouslyUploaded(const device::Buffers::NetworkInputBuffers& buffers)
	{
		return buffers.networkStateStructBuffer.castedPointer<NetworkInput>();
	}

	vtx::NetworkInput::NetworkInput(const int& maxSize, device::Buffers::NetworkInputBuffers& buffers)
	{
		position = buffers.positionBuffer.alloc<math::vec3f>(maxSize);
		wo = buffers.woBuffer.alloc<math::vec3f>(maxSize);
		normal = buffers.normalBuffer.alloc<math::vec3f>(maxSize);
	}
}

