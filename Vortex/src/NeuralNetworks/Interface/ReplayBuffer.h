#pragma once
#ifndef REPLAY_BUFFER_H
#define REPLAY_BUFFER_H
#include "Core/Math.h"

namespace vtx
{
	struct NetworkInput;

	struct ReplayBuffer
	{
		static ReplayBuffer* upload(const int& maxReplayBufferSize);

		static ReplayBuffer* getPreviouslyUploaded();

	private:
		ReplayBuffer(const int& maxReplayBufferSize);
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