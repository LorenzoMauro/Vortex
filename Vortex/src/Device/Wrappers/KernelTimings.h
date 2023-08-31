#pragma once
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <utility>

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

		//EXPERIMENT
		K_MAPE,
		K_MSE,

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
		"NN Triangle Wave Encoding",

		"MAPE",
		"MSE"

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

	CudaEventTimes getCudaEventTimes();

	int getLaunches();

    std::pair<cudaEvent_t, cudaEvent_t> GetProfilerEvents(const char* description);

    float GetKernelTimeMS(const char* description);

    int GetKernelLaunches(const char* description);

    void resetKernelStats();

}
