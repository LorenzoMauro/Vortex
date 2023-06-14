#pragma once

#include <tiny-cuda-nn/common_device.h>

#include <tiny-cuda-nn/config.h>
#include "Core/Math.h"
#include "Scene/Node.h"

using namespace tcnn;

namespace vtx::graph
{
	struct NetworkOptions
	{
		int batchSize = 10; // in thousand
	};

	class Network : public Node
	{
		Network() : Node(NT_NETWORK)
		{
			trainingTarget = GPUMatrix<math::vec3f>(inputDimension, batchSize);
			trainingInput = GPUMatrix<math::vec3f>(inputDimension, batchSize);
		}

		void test()
		{
			trainingInput.data();
		}

	public:
		GPUMatrix<math::vec3f> trainingTarget;
		GPUMatrix<math::vec3f> trainingInput;
		json config;
		constexpr int inputDimension = 9;
	};
};