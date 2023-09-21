#include "InputComposer.h"
#include "Device/CudaFunctions/cudaFunctions.h"
#include "Core/Log.h"

namespace vtx::network
{
	InputComposer::InputComposer(const torch::Device _device, InputSettings* _settings) :
		device(_device)
	{
		settings = _settings;
	}

	torch::Tensor& InputComposer::getInput()
	{
		if (!inputPrepared)
		{
			prepareInput();
		}
		return inputTensor;
	}

	void InputComposer::setFromDevice(float* positionPtr, float* woPtr, float* normalPtr, const int _inputCount)
	{
		position = positionPtr;
		wo = woPtr;
		normal = normalPtr;
		inputCount = _inputCount;
		inputPrepared = false;
	}

	void InputComposer::setFromBuffer(const device::Buffers::NetworkInputBuffers& inputBuffer, const int inputCount, const int startIdx)
	{
		const auto positionPtr = inputBuffer.positionBuffer.castedPointer<float>();
		const auto woPtr = inputBuffer.woBuffer.castedPointer<float>();
		const auto normalPtr = inputBuffer.normalBuffer.castedPointer<float>();

		float* positionBatchPtr = positionPtr + startIdx * inputCount * 3;
		float* woBatchPtr = woPtr + startIdx * inputCount * 3;
		float* normalBatchPtr = normalPtr + startIdx * inputCount * 3;
		setFromDevice(positionBatchPtr, woBatchPtr, normalBatchPtr, inputCount);
	}

	int InputComposer::dimension()
	{
		if (inputDimension == 0)
		{
			computeInputDimension();
		}
		return inputDimension;
	}

	void InputComposer::prepareInput()
	{
		if (!position || !wo || !normal)
		{
			VTX_ERROR("Input not set");
		}
		positionTensor = encode(position, 3, encodedPositionBuffer, settings->positionEncoding);
		woTensor = encode(wo, 3, encodedWoBuffer, settings->woEncoding);
		normalTensor = encode(normal, 3, encodedNormalBuffer, settings->normalEncoding);

		inputTensor = torch::cat({ positionTensor, woTensor, normalTensor }, -1);
		inputPrepared = true;
	}

	int InputComposer::encodingDim(const EncodingSettings& encSettings)
	{

		switch (encSettings.type)
		{
		case E_NONE:
		{
			return 3;
		}
		case E_TRIANGLE_WAVE:
		{
			return encSettings.features * 3;
		}
		default:
		{
			VTX_ERROR("Encoding type not supported");
			return 0;
		}
		}
	}

	void InputComposer::computeInputDimension()
	{
		inputDimension = 0;
		inputDimension += encodingDim(settings->positionEncoding);
		inputDimension += encodingDim(settings->woEncoding);
		inputDimension += encodingDim(settings->normalEncoding);
	}

	torch::Tensor InputComposer::encode(float* input, int inputSize, CUDABuffer& encodedBuffer, const EncodingSettings& encodingSettings)
	{
		switch (encodingSettings.type)
		{
		case E_NONE:
		{
			return torch::from_blob(input, { inputCount, inputSize }, torch::TensorOptions().device(device).dtype(torch::kFloat));
		}
		case E_TRIANGLE_WAVE:
		{
			float* encodedPositionPtr = cuda::triangleWaveEncoding(encodedBuffer, input, inputCount, inputSize, encodingSettings.features);
			return torch::from_blob(encodedPositionPtr, { inputCount, encodingSettings.features * inputSize }, torch::TensorOptions().device(device).dtype(torch::kFloat));
		}
		default:
		{
			VTX_ERROR("Encoding type not supported");
			return {};
		}
		}
	}

}
