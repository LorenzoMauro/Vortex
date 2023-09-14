#pragma once
#ifndef NETWORK_SETTINGS_H
#define NETWORK_SETTINGS_H

namespace vtx::network
{
	enum LossType
	{
		L_KL_DIV,
		L_KL_DIV_MC_ESTIMATION,
		L_PEARSON_DIV,
		L_PEARSON_DIV_MC_ESTIMATION,
		L_COUNT
	};

	inline static const char* lossNames[] = {
		"KL Divergence",
		"KL Divergence MC Estimation",
		"Pearson Divergence",
		"Pearson Divergence MC Estimation"
	};

	enum NetworkType
	{
		NT_SAC,
		NT_NGP,

		NT_COUNT
	};

	inline static const char* networkNames[] = {
		"Soft Actor Critic",
		"Neural Path Guiding"
	};

	enum SamplingStrategy
	{
		SS_ALL,
		SS_PATHS_WITH_CONTRIBUTION,
		SS_LIGHT_SAMPLES,

		SS_COUNT
	};

	inline static const char* samplingStrategyNames[] = {
		"All",
		"Paths with Contribution",
		"Light Samples"
	};

	struct TrainingBatchGenerationSettings
	{
		SamplingStrategy strategy;
		bool             weightByMis;
		float            lightSamplingProb;
		bool              isUpdated;
	};

	enum EncodingType
	{
		E_NONE,
		E_TRIANGLE_WAVE,

		E_COUNT
	};

	inline static const char* encodingNames[] = {
		"None", "Triangle Wave"
	};

	enum DistributionType
	{
		D_SPHERICAL_GAUSSIAN,
		D_NASG_TRIG,
		D_NASG_ANGLE,
		D_NASG_AXIS_ANGLE,

		D_COUNT
	};

	inline static const char* distributionNames[] = {
			"Spherical Gaussian",
			"NASG Trigonometric",
			"NASG Angle",
			"NASG Axis Angle"
	};

	struct EncodingSettings
	{
		EncodingType type = E_NONE;
		int features = 3;
		bool isUpdated = true;

	};

	struct InputSettings
	{
		EncodingSettings positionEncoding;
		EncodingSettings woEncoding;
		EncodingSettings normalEncoding;
		bool isUpdated = true;
	};

	struct PathGuidingNetworkSettings
	{
		int hiddenDim = 64;
		int numHiddenLayers = 4;
		DistributionType distributionType = D_SPHERICAL_GAUSSIAN;
		int mixtureSize = 1;
		bool produceSamplingFraction = false;
		bool isUpdated = true;
	};

	struct SacSettings
	{
		float polyakFactor;
		float logAlphaStart;
		float gamma;
		float neuralSampleFraction;
		float policyLr;
		float qLr;
		float alphaLr;
		bool isUpdated = true;
	};

	struct NpgSettings
	{
		float    learningRate;
		float    e;
		bool     constantBlendFactor;
		bool     samplingFractionBlend;
		LossType lossType  = L_KL_DIV;
		bool     isUpdated = true;
		bool     absoluteLoss;
		bool      meanLoss;
	};

	struct NetworkSettings
	{
		bool                            active;
		int                             batchSize;
		int                             maxTrainingStepPerFrame;
		bool                            doTraining;
		int                             maxTrainingSteps;
		bool                            doInference;
		int                             inferenceIterationStart;
		bool                            clearOnInferenceStart;
		NetworkType                     type;
		TrainingBatchGenerationSettings trainingBatchGenerationSettings;
		InputSettings                   inputSettings;
		PathGuidingNetworkSettings      pathGuidingSettings;
		SacSettings                     sac;
		NpgSettings                     npg;
		bool                            isUpdated            = true;
		bool                            isDatasetSizeUpdated = true;
		int                             depthToDebug         = 0;

		void resetUpdate()
		{
			isUpdated = false;
			isDatasetSizeUpdated = false;
			pathGuidingSettings.isUpdated = false;
			inputSettings.isUpdated = false;
			inputSettings.positionEncoding.isUpdated = false;
			inputSettings.woEncoding.isUpdated = false;
			inputSettings.normalEncoding.isUpdated = false;
			sac.isUpdated = false;
			npg.isUpdated = false;
			trainingBatchGenerationSettings.isUpdated = false;
		}

		bool isAnyUpdated()
		{
			return (
				isUpdated || pathGuidingSettings.isUpdated ||
				inputSettings.isUpdated || inputSettings.positionEncoding.isUpdated ||
				inputSettings.woEncoding.isUpdated || inputSettings.normalEncoding.isUpdated || sac.isUpdated || npg.isUpdated ||
				trainingBatchGenerationSettings.isUpdated
				);
		}
	};

	
}


#endif