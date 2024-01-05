#include "NetworkImplementation.h"
#include "Device/DevicePrograms/LaunchParams.h"
#include "Device/UploadCode/DeviceDataCoordinator.h"
#include "Device/Wrappers/KernelLaunch.h"
#include "Device/Wrappers/KernelTimings.h"
#include "Interface/NetworkInterface.h"

namespace vtx::network
{
	void NetworkImplementation::prepareDataset()
	{
		const LaunchParams* deviceParams = onDeviceData->launchParamsData.getDeviceImage();
		const LaunchParams& hostParams = onDeviceData->launchParamsData.getHostImage();
		const math::vec2ui screenSize = hostParams.frameBuffer.frameSize;
		const int maxBounces = hostParams.settings.renderer.maxBounces;
		const int               nPixels = screenSize.x * screenSize.y;

		gpuParallelFor(eventNames[N_FILL_PATH],
			nPixels,
			[deviceParams, screenSize] __device__(const int id)
		{
			deviceParams->networkInterface->finalizePath(id, screenSize.x, screenSize.y, deviceParams->settings.neural, true);
			if (id == 0)
			{
				deviceParams->networkInterface->trainingData->reset();
			}
		});

		const int               maxDatasetSize = maxBounces * nPixels * 2;
		gpuParallelFor(eventNames[N_PREPARE_DATASET],
			maxDatasetSize,
			[deviceParams] __device__(const int id)
		{
			const Samples* samples = deviceParams->networkInterface->samples;
			TrainingData* trainingData = deviceParams->networkInterface->trainingData;

			trainingData->buildTrainingData(id, samples);
		});

	}
}

