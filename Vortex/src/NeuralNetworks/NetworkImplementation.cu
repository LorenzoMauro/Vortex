#include "NetworkImplementation.h"
#include "Device/DevicePrograms/LaunchParams.h"
#include "Device/Wrappers/KernelLaunch.h"
#include "Device/Wrappers/KernelTimings.h"
#include "Interface/NetworkInterface.h"

namespace vtx::network
{
	void NetworkImplementation::shuffleDataset(LaunchParams* params) const
	{
		const int datasetSize = settings->batchSize * settings->maxTrainingStepPerFrame;
		LaunchParams* paramsCopy = params;
		NetworkSettings settingsCopy = *settings;
		gpuParallelFor(eventNames[N_SHUFFLE_DATASET],
			datasetSize,
			[paramsCopy, settingsCopy] __device__(const int id)
		{
			paramsCopy->networkInterface->shuffleDataset(id, settingsCopy);
		});
	}
}

