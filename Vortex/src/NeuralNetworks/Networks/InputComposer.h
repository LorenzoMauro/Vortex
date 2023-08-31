#ifndef NETWORK_INPUTCOMPOSER_H
#define NETWORK_INPUTCOMPOSER_H

#pragma once
#include "NeuralNetworks/NetworkSettings.h"
#include "Device/UploadCode/UploadBuffers.h"
#include <torch/torch.h>

namespace vtx::network
{
	class InputComposer
	{
	public:
		InputComposer() = default;

		InputComposer(torch::Device _device, InputSettings* _settings);

		torch::Tensor& getInput();

		void setFromDevice(float* positionPtr, float* woPtr, float* normalPtr, const int _inputCount);

		void setFromBuffer(const device::Buffers::NetworkInputBuffers& inputBuffer, const int inputCount, const int startIdx);

		int dimension();
	private:
		void prepareInput();

		int encodingDim(const EncodingSettings& encSettings);

		void computeInputDimension();

		torch::Tensor encode(float* input, int inputSize, CUDABuffer& encodedBuffer, const EncodingSettings& encodingSettings);

		float* position = nullptr;
		torch::Tensor positionTensor;
		float* wo = nullptr;
		torch::Tensor woTensor;
		float* normal = nullptr;
		torch::Tensor normalTensor;
		int inputCount = 0;
		int inputDimension = 0;

		torch::Tensor inputTensor;
		bool inputPrepared = false;

		InputSettings* settings = nullptr;
		CUDABuffer encodedPositionBuffer;
		CUDABuffer encodedWoBuffer;
		CUDABuffer encodedNormalBuffer;

		torch::Device device = torch::kCPU;
	};
}

#endif
