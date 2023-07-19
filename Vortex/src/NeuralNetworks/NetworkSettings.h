#pragma once
#ifndef NETWORK_SETTINGS_H
#define NETWORK_SETTINGS_H

namespace vtx::network
{
	enum NetworkType
	{
		NT_SAC,
		NT_NGP,

		NT_COUNT
	};

	inline static const char* networkNames[] = {
		"Soft Actor Critic", "Neural Path Guiding"
	};
	
	struct NetworkSettings
	{
		int   batchSize;
		int   maxTrainingStepPerFrame;
		bool  doInference;
		int   inferenceIterationStart;
		bool  clearOnInferenceStart;

	};

	struct SacSettings : public NetworkSettings
	{
		float polyakFactor;
		float logAlphaStart;
		float gamma;
		float neuralSampleFraction;
		float policyLr;
		float qLr;
		float alphaLr;
		float autoencoderLr;
	};

	struct NgpSettings : public NetworkSettings
	{
		
	};
	
}


#endif