#pragma once
#include "TcnnSettings.h"
#include "NeuralNetworks/NetworkSettings.h"

namespace vtx::network
{
	

	struct PGNetSettings
	{
		int maxTrainingSteps = 1000;
		int batchSize = 1;
		float learningRate = 0.001f;


		// INPUT ENCODING SETTINGS
		bool normalizePosition = false;
		TcnnCompositeEncodingConfig inputSettings;

		// DISTRIBUTION SETTINGS
		DistributionType distributionType = D_SPHERICAL_GAUSSIAN;
		int mixtureSize = 1;

		bool          doTraining    = true;
		LossType      lossType      = L_KL_DIV_MC_ESTIMATION;
		LossReduction lossReduction = LossReduction::MEAN;
		bool          constantBlendFactor;
		float         blendFactor;
		bool          useEntropyLoss = false;
		float         entropyWeight  = 1.0f;
		float         targetEntropy  = 3.0f;
		bool          plotGraphs     = true;
		bool           samplingFractionBlend = false;
	};
}
