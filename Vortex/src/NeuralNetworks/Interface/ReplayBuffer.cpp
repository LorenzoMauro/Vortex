#include "ReplayBuffer.h"
#include "Device/UploadCode/UploadBuffers.h"
#include "NetworkInputs.h"

namespace vtx
{
	ReplayBuffer* ReplayBuffer::upload(device::ReplayBufferBuffers& buffers, const int& maxReplayBufferSize)
	{
		const ReplayBuffer replayBuffer(buffers, maxReplayBufferSize);
		return buffers.replayBufferStructBuffer.upload(replayBuffer);
	}

	ReplayBuffer* ReplayBuffer::getPreviouslyUploaded(const device::ReplayBufferBuffers& buffers)
	{
		return buffers.replayBufferStructBuffer.castedPointer<ReplayBuffer>();
	}

	ReplayBuffer::ReplayBuffer(device::ReplayBufferBuffers& buffers, const int& maxReplayBufferSize)
	{
		action = buffers.actionBuffer.alloc<math::vec3f>(maxReplayBufferSize);
		reward = buffers.rewardBuffer.alloc<float>(maxReplayBufferSize);
		doneSignal = buffers.doneBuffer.alloc<int>(maxReplayBufferSize);
		state = NetworkInput::upload(maxReplayBufferSize, buffers.stateBuffer);
		nextState = NetworkInput::upload(maxReplayBufferSize, buffers.nextStatesBuffer);
		nAlloc = maxReplayBufferSize;
	}
}


