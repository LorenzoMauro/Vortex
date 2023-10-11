#include "Paths.h"
#include "Device/UploadCode/UploadBuffers.h"

namespace vtx
{
	Paths* Paths::upload(device::PathsBuffer& buffers, const int& _maxDepth, const int& _numberOfPixels)
	{
		const Paths pathsStruct(buffers, _maxDepth, _numberOfPixels);
		CUDABuffer& pathsStructBuffer = buffers.pathStructBuffer;
		return pathsStructBuffer.upload(pathsStruct);
	}

	Paths* Paths::getPreviouslyUploaded(const device::PathsBuffer& buffers)
	{
		return buffers.pathStructBuffer.castedPointer<Paths>();
	}

	Paths::Paths(device::PathsBuffer& buffers, const int& _maxDepth, const int& _numberOfPixels)
	{
		//allocated all bounces for each pixel and max Depth

		maxAllocDepth = _maxDepth + 1; // +1 for miss "bounce"
		numberOfPaths = _numberOfPixels;
		auto* allBounces = buffers.bouncesBuffer.alloc<Bounce>(maxAllocDepth * numberOfPaths);
		//buffers.resetBouncesBuffer.alloc<Bounce>(maxAllocDepth * numberOfPaths);
		// Now select the proper pointer by considering the max depth and the index
		std::vector<Path> pathsVector;
		for (int i = 0; i < numberOfPaths; ++i)
		{
			Bounce* deviceBouncesPtr = allBounces + i * maxAllocDepth;
			pathsVector.emplace_back(deviceBouncesPtr, maxAllocDepth);
		}

		paths = buffers.pathsArrayBuffer.upload(pathsVector);
		validPixels = buffers.validPixelsBuffer.alloc<int>(numberOfPaths);
		validPixelsCount = 0;

		pixelsWithContribution = buffers.pixelsWithContributionBuffer.alloc<int>(numberOfPaths);
		pixelsWithContributionCount = 0;

		validLightSample = buffers.validLightSampleBuffer.alloc<math::vec3i>(numberOfPaths*(maxAllocDepth*2));
		validLightSampleCount = 0;

		pathsAccumulator = buffers.pathsAccumulatorBuffer.alloc<math::vec3f>(numberOfPaths);

	}

	Path::Path(Bounce* deviceBouncePtr, const int _maxAllocDepth)
	{
		maxDepth = 0;
		bounces = deviceBouncePtr;
		maxAllocDepth = _maxAllocDepth;
	}
}

