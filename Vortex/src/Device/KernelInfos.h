#pragma once
#ifndef KERNEL_INFOS_H
#define KERNEL_INFOS_H
#include "Wrappers/dWrapper.h"

namespace vtx
{
	

	enum EventType
	{
		//WAVEFRONT CUDA EVENTS
		K_SET_QUEUE_COUNTERS,
		K_GEN_CAMERA_RAY,
		K_TRACE_RADIANCE_RAY,
		K_SHADE_RAY,
		K_HANDLE_ESCAPED_RAY,
		K_ACCUMULATE_RAY,
		K_RETRIEVE_QUEUE_SIZE,
		K_RESET,

		//RENDERER CUDA EVENTS
		R_NOISE_COMPUTATION,
		R_TRACE,
		R_POSTPROCESSING,
		R_DISPLAY,
		R_TONE_MAP_RADIANCE,

		//NN CUDA EVENTS
		N_SHUFFLE_DATASET,
		N_TRAIN,
		N_INFER,
		N_TRIANGLE_WAVE_ENCODING,

		K_COUNT
	};

	inline static const char* eventNames[] = {
		"setQueueCounters",
		"genCameraRay",
		"traceRadianceRay",
		"shadeRay",
		"handleEscapedRay",
		"accumulateRay",
		"retrieveQueueSize",
		"reset",

		"Rendering Noise Computation",
		"Rendering Trace",
		"Rendering Post Processing",
		"Rendering Display",
		"Rendering Tone Mapping Radiance",

		"NN Shuffle Dataset",
		"NN Train",
		"NN Infer",
		"NN Triangle Wave Encoding"

	};

	struct CudaEventTimes
	{
		float genCameraRay = 0.0f;
		float traceRadianceRay = 0.0f;
		float reset = 0.0f;
		float shadeRay = 0.0f;
		float handleEscapedRay = 0.0f;
		float accumulateRay = 0.0f;
		float fetchQueueSize = 0.0f;
		float setQueueCounters = 0.0f;
		float noiseComputation = 0.0f;
		float trace = 0.0f;
		float postProcessing = 0.0f;
		float display = 0.0f;
		float toneMapRadiance = 0.0f;
		float nnShuffleDataset = 0.0f;
		float nnTrain = 0.0f;
		float nnInfer = 0.0f;
	};

	static CudaEventTimes& getCudaEventTimes()
	{
		CudaEventTimes times;
		times.setQueueCounters = GetKernelTimeMS(eventNames[K_SET_QUEUE_COUNTERS]);
		times.genCameraRay = GetKernelTimeMS(eventNames[K_GEN_CAMERA_RAY]);
		times.traceRadianceRay = GetKernelTimeMS(eventNames[K_TRACE_RADIANCE_RAY]);
		times.reset = GetKernelTimeMS(eventNames[K_RESET]);
		times.shadeRay = GetKernelTimeMS(eventNames[K_SHADE_RAY]);
		times.handleEscapedRay = GetKernelTimeMS(eventNames[K_HANDLE_ESCAPED_RAY]);
		times.accumulateRay = GetKernelTimeMS(eventNames[K_ACCUMULATE_RAY]);
		times.fetchQueueSize = GetKernelTimeMS(eventNames[K_RETRIEVE_QUEUE_SIZE]);

		times.noiseComputation = GetKernelTimeMS(eventNames[R_NOISE_COMPUTATION]);
		times.trace = GetKernelTimeMS(eventNames[R_TRACE]);
		times.postProcessing = GetKernelTimeMS(eventNames[R_POSTPROCESSING]);
		times.display = GetKernelTimeMS(eventNames[R_DISPLAY]);
		times.toneMapRadiance = GetKernelTimeMS(eventNames[R_TONE_MAP_RADIANCE]);

		times.nnShuffleDataset = GetKernelTimeMS(eventNames[N_SHUFFLE_DATASET]);
		times.nnTrain = GetKernelTimeMS(eventNames[N_TRAIN]);
		times.nnInfer = GetKernelTimeMS(eventNames[N_INFER]);
		return times;
	}

	static int getLaunches()
	{
		return GetKernelLaunches(eventNames[R_TRACE]);
	}
}
#endif
