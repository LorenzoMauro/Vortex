#pragma once
#ifndef NETWORK_SETTINGS_H
#define NETWORK_SETTINGS_H
#include <map>
#include <string>

#include "DistributionConfig.h"
#include "EncodingConfig.h"
#include "LossConfig.h"
#include "TrainingBatchConfig.h"

namespace vtx::network::config
{
	enum ActivationType
	{
		AT_RELU,
		AT_TANH,
		AT_SIGMOID,
		AT_SOFTMAX,
		AT_NONE
	};

	struct MlpSettings
	{
		int				inputDim;
		int				outputDim;
		int				hiddenDim;
		int				numHiddenLayers;
		ActivationType	activationType = AT_RELU;
	};


	struct MainNetEncodingConfig
	{
		bool                    normalizePosition = false;
		EncodingConfig position = {};
		EncodingConfig wo = {};
		EncodingConfig normal = {};
	};

	struct AuxNetEncodingConfig
	{
		EncodingConfig wi = {};
	};

	struct NetworkSettings
	{
		// GENERAL SETTINGS
		bool active       = true;
		bool wasActive    = active;
		bool doTraining   = true;
		bool doInference  = true;
		bool plotGraphs   = true;
		bool isUpdated    = true;
		int  depthToDebug = 0;

		int   maxTrainingSteps        = 1000;
		int   batchSize               = 1;
		float learningRate            = 0.001f;
		int   inferenceIterationStart = 1;
		bool  clearOnInferenceStart   = false;

		// BATCH GENERATION
		BatchGenerationConfig trainingBatchGenerationSettings;

		//MAIN NETWORK SETTINGS
		MlpSettings mainNetSettings;
		MainNetEncodingConfig inputSettings;

		// DISTRIBUTION SETTINGS
		DistributionType distributionType = D_SPHERICAL_GAUSSIAN;
		int              mixtureSize      = 1;

		//LOSS SETTINGS
		LossType      lossType              = L_KL_DIV_MC_ESTIMATION;
		LossReduction lossReduction         = MEAN;
		bool          constantBlendFactor   = false;
		float         blendFactor           = 0.9f;
		float         targetScale			= 1.0f;
		bool          samplingFractionBlend = false;
		bool		  scaleBySampleProb    = false;

		// ENTROPY LOSS SETTINGS
		bool          useEntropyLoss = false;
		float         entropyWeight = 1.0f;
		float         targetEntropy = 3.0f;

		// AUXILIARY LOSS SETTINGS
		bool                 useAuxiliaryNetwork = false;
		MlpSettings          auxiliaryNetSettings;
		AuxNetEncodingConfig auxiliaryInputSettings;
		int                  totAuxInputSize             = 64;
		float                inRadianceLossFactor        = 1.0f;
		float                outRadianceLossFactor       = 1.0f;
		float                throughputLossFactor        = 1.0f;
		float                auxiliaryWeight             = 1.0f;
		float                radianceTargetScaleFactor   = 1.0f;
		float                throughputTargetScaleFactor = 1.0f;

		bool useMaterialId = false;
		EncodingConfig materialIdEncodingConfig = {};
		bool useTriangleId = false;
		EncodingConfig triangleIdEncodingConfig = {};
		bool useInstanceId = false;
		EncodingConfig instanceIdEncodingConfig = {};

		bool  scaleLossBlendedQ            = false;
		bool  clampBsdfProb                = false;
		float fractionBlendTrainPercentage = 0.2;

		bool learnInputRadiance = false;
		float  lossClamp = 100.0f;

		void resetUpdate()
		{
			isUpdated                                 = false;
			trainingBatchGenerationSettings.isUpdated = false;
		}

		bool isAnyUpdated()
		{
			return isUpdated || trainingBatchGenerationSettings.isUpdated;
		}
	};

}


#endif