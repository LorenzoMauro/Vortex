#pragma once
#ifndef NETWORK_SETTINGS_H
#define NETWORK_SETTINGS_H
#include <map>
#include <string>

#include "Networks/TcnnSettings.h"

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

	inline static std::map<std::string, LossType> lossNameToEnum =
	{
			{"KL Divergence", L_KL_DIV},
			{"KL Divergence MC Estimation", L_KL_DIV_MC_ESTIMATION},
			{"Pearson Divergence", L_PEARSON_DIV},
			{"Pearson Divergence MC Estimation", L_PEARSON_DIV_MC_ESTIMATION}
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

	inline static std::map<std::string, NetworkType> networkNameToEnum =
	{
		{"Soft Actor Critic", NT_SAC},
		{"Neural Path Guiding", NT_NGP}
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

	inline static std::map<std::string, SamplingStrategy> samplingStrategyNameToEnum =
	{
		{"All", SS_ALL},
		{"Paths with Contribution", SS_PATHS_WITH_CONTRIBUTION},
		{"Light Samples", SS_LIGHT_SAMPLES}
	};

	struct TrainingBatchGenerationSettings
	{
		SamplingStrategy strategy;
		bool             weightByMis;
		float            lightSamplingProb;
		bool              isUpdated = true;
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

	inline static std::map<std::string, EncodingType> encodingNameToEnum =
	{
		{"None", E_NONE},
		{"Triangle Wave", E_TRIANGLE_WAVE}
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

	inline static std::map<std::string, DistributionType> distributionNameToEnum =
	{
			{"Spherical Gaussian", D_SPHERICAL_GAUSSIAN},
			{"NASG Trigonometric", D_NASG_TRIG},
			{"NASG Angle", D_NASG_ANGLE},
			{"NASG Axis Angle", D_NASG_AXIS_ANGLE}
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
		TcnnCompositeEncodingConfig tcnnCompositeEncodingConfig;
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
		bool							wasActive = active;	
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
		int                             depthToDebug         = 0;

		void resetUpdate()
		{
			isUpdated = false;
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