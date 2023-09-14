#include "ReplayBuffer.h"
#include "Device/UploadCode/UploadBuffers.h"
#include "NetworkInputs.h"

namespace vtx
{
	ReplayBuffer* ReplayBuffer::upload(const int& maxReplayBufferSize)
	{
		const ReplayBuffer replayBuffer(maxReplayBufferSize);
		return UPLOAD_BUFFERS->networkInterfaceBuffer.replayBufferBuffers.replayBufferStructBuffer.upload(replayBuffer);
	}

	ReplayBuffer* ReplayBuffer::getPreviouslyUploaded()
	{
		return UPLOAD_BUFFERS->networkInterfaceBuffer.replayBufferBuffers.replayBufferStructBuffer.castedPointer<ReplayBuffer>();
	}

	ReplayBuffer::ReplayBuffer(const int& maxReplayBufferSize)
	{
		device::Buffers::ReplayBufferBuffers& replayBufferBuffers = UPLOAD_BUFFERS->networkInterfaceBuffer.replayBufferBuffers;
		action = replayBufferBuffers.actionBuffer.alloc<math::vec3f>(maxReplayBufferSize);
		reward = replayBufferBuffers.rewardBuffer.alloc<float>(maxReplayBufferSize);
		doneSignal = replayBufferBuffers.doneBuffer.alloc<int>(maxReplayBufferSize);
		state = NetworkInput::upload(maxReplayBufferSize, replayBufferBuffers.stateBuffer);
		nextState = NetworkInput::upload(maxReplayBufferSize, replayBufferBuffers.nextStatesBuffer);
		nAlloc = maxReplayBufferSize;
	}
}


