#pragma once
#ifndef REPLAY_BUFFER_H
#define REPLAY_BUFFER_H
#include "Core/Math.h"

namespace vtx
{
	namespace device
	{
		struct ReplayBufferBuffers;
	}

	struct NetworkInput;

	struct ReplayBuffer
	{
		static ReplayBuffer* upload(device::ReplayBufferBuffers& buffers, const int& maxReplayBufferSize);

		static ReplayBuffer* getPreviouslyUploaded(const device::ReplayBufferBuffers& buffers);

	private:
		ReplayBuffer(device::ReplayBufferBuffers& buffers, const int& maxReplayBufferSize);
	public:

		NetworkInput* state;
		NetworkInput* nextState;
		math::vec3f* action;
		float* reward;
		int* doneSignal;
		int nAlloc;
	};
}
#endif