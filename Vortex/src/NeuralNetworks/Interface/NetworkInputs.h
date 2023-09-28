#pragma once
#ifndef NETWORK_INPUTS_H
#define NETWORK_INPUTS_H
#include "Core/Math.h"

namespace vtx
{
	namespace device
	{
		struct NetworkInputBuffers;
	}

	struct NetworkInput
	{
		static NetworkInput* upload(const int& maxSize, device::NetworkInputBuffers& buffers);

		static NetworkInput* getPreviouslyUploaded(const device::NetworkInputBuffers& buffers);

	private:
		NetworkInput(const int& maxSize, device::NetworkInputBuffers& buffers);

	public:
		__forceinline__ __device__ void addState(const int& index, const math::vec3f& _position, const math::vec3f& _wo, const math::vec3f& _normal) const
		{
			this->position[index] = _position;
			this->wo[index] = _wo;
			this->normal[index] = _normal;
		}
		math::vec3f* position = nullptr;
		math::vec3f* wo = nullptr;
		math::vec3f* normal = nullptr;
	};
}
#endif
