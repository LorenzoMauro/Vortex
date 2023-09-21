#include "Paths.h"
#include "Device/UploadCode/UploadBuffers.h"

namespace vtx
{
	Paths* Paths::upload(const int& _maxDepth, const int& _numberOfPixels)
	{
		const Paths pathsStruct(_maxDepth, _numberOfPixels);
		CUDABuffer& pathsStructBuffer = UPLOAD_BUFFERS->networkInterfaceBuffer.pathsBuffers.pathStructBuffer;
		return pathsStructBuffer.upload(pathsStruct);
	}

	Paths* Paths::getPreviouslyUploaded()
	{
		return UPLOAD_BUFFERS->networkInterfaceBuffer.pathsBuffers.pathStructBuffer.castedPointer<Paths>();
	}

	Paths::Paths(const int& _maxDepth, const int& _numberOfPixels)
	{
		//allocated all bounces for each pixel and max Depth

		maxAllocDepth = _maxDepth + 1; // +1 for miss "bounce"
		numberOfPaths = _numberOfPixels;
		auto* allBounces = UPLOAD_BUFFERS->networkInterfaceBuffer.pathsBuffers.bouncesBuffer.alloc<Bounce>(maxAllocDepth * numberOfPaths);
		UPLOAD_BUFFERS->networkInterfaceBuffer.pathsBuffers.resetBouncesBuffer.alloc<Bounce>(maxAllocDepth * numberOfPaths);
		// Now select the proper pointer by considering the max depth and the index
		std::vector<Path> pathsVector;
		for (int i = 0; i < numberOfPaths; ++i)
		{
			Bounce* deviceBouncesPtr = allBounces + i * maxAllocDepth;
			pathsVector.emplace_back(deviceBouncesPtr, maxAllocDepth);
		}

		paths = UPLOAD_BUFFERS->networkInterfaceBuffer.pathsBuffers.pathsArrayBuffer.upload(pathsVector);
		validPixels = UPLOAD_BUFFERS->networkInterfaceBuffer.pathsBuffers.validPixelsBuffer.alloc<int>(numberOfPaths);
		validPixelsCount = 0;

		pixelsWithContribution = UPLOAD_BUFFERS->networkInterfaceBuffer.pathsBuffers.pixelsWithContributionBuffer.alloc<int>(numberOfPaths);
		pixelsWithContributionCount = 0;

		validLightSample = UPLOAD_BUFFERS->networkInterfaceBuffer.pathsBuffers.validLightSampleBuffer.alloc<math::vec3i>(numberOfPaths*(maxAllocDepth*2));
		validLightSampleCount = 0;

		pathsAccumulator = UPLOAD_BUFFERS->networkInterfaceBuffer.pathsBuffers.pathsAccumulatorBuffer.alloc<math::vec3f>(numberOfPaths);

	}

	Path::Path(Bounce* deviceBouncePtr, const int _maxAllocDepth)
	{
		maxDepth = 0;
		bounces = deviceBouncePtr;
		maxAllocDepth = _maxAllocDepth;
	}
}

