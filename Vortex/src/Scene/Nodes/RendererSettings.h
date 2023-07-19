#pragma once
#include "Device/DevicePrograms/LaunchParams.h"


namespace vtx::graph
{
	struct RendererSettings
	{
		int                                       iteration;
		int                                       maxBounces;
		int                                       maxSamples;
		bool                                      accumulate;
		SamplingTechnique samplingTechnique;
		DisplayBuffer     displayBuffer;
		bool                                      isUpdated;
		float                                     minClamp;
		float                                     maxClamp;
		int                                       noiseKernelSize;
		bool                                      adaptiveSampling;
		int                                       minAdaptiveSamples;
		int                                       minPixelSamples;
		int                                       maxPixelSamples;
		float                                     albedoNormalNoiseInfluence;
		float                                     noiseCutOff;
		int                                       fireflyKernelSize;
		float                                     fireflyThreshold;
		bool                                      removeFireflies;
		bool                                      enableDenoiser;
		int                                       denoiserStart;
		float                                     denoiserBlend;
		bool                                      useWavefront;
		bool                                      useRussianRoulette;
		bool                                      fitWavefront;
		bool                                      optixShade;
		bool                                      parallelShade;
		bool                                      runOnSeparateThread;
		float                                     longPathPercentage;
		bool                                      useLongPathKernel;
		bool                                      useNetwork;
	};
}
