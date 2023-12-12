#include "Experiment.h"

#include <queue>
#include <random>
#include <unordered_set>

#include "Core/Application.h"
#include "Device/UploadCode/DeviceDataCoordinator.h"
#include "Gui/Windows/ExperimentsWindow.h"
#include "Scene/Nodes/Renderer.h"
#include "Device/CudaFunctions/cudaFunctions.h"
#include "Serialization/Serializer.h"

static std::random_device rd;  // Obtain a random number from hardware
static std::mt19937 g(rd());   // Seed the generator

namespace vtx
{

#define stringHash(x) #x << "|" << std::to_string(x)
	std::string toStringHash(const network::config::BatchGenerationConfig& config)
	{
		std::stringstream hash;
		hash << stringHash(config.strategy);
		hash << stringHash(config.weightByMis);
		hash << stringHash(config.lightSamplingProb);
		hash << stringHash(config.isUpdated);
		return hash.str();
	}

	std::string toStringHash(const network::config::FrequencyEncoding& config) {
		std::stringstream hash;
		hash << stringHash((int)config.otype);
		hash << stringHash(config.n_frequencies);
		return hash.str();
	};

	std::string toStringHash(const network::config::GridEncoding& config) {
		std::stringstream hash;
		hash << stringHash((int)config.otype);
		hash << stringHash((int)config.type);
		hash << stringHash(config.n_levels);
		hash << stringHash(config.n_features_per_level);
		hash << stringHash(config.log2_hashmap_size);
		hash << stringHash(config.base_resolution);
		hash << stringHash(config.per_level_scale);
		hash << stringHash((int)config.interpolation);
		return hash.str();
	};

	std::string toStringHash(const network::config::IdentityEncoding& config) {
		std::stringstream hash;
		hash << stringHash((int)config.otype);
		hash << stringHash(config.scale);
		hash << stringHash(config.offset);
		return hash.str();
	};

	std::string toStringHash(const network::config::OneBlobEncoding& config) {
		std::stringstream hash;
		hash << stringHash((int)config.otype);
		hash << stringHash(config.n_bins);
		return hash.str();
	};

	std::string toStringHash(const network::config::SphericalHarmonicsEncoding& config) {
		std::stringstream hash;
		hash << stringHash((int)config.otype);
		hash << stringHash(config.degree);
		return hash.str();
	};

	std::string toStringHash(const network::config::TriangleWaveEncoding& config) {
		std::stringstream hash;
		hash << stringHash((int)config.otype);
		hash << stringHash(config.n_frequencies);
		return hash.str();
	};

	std::string toStringHash(const network::config::EncodingConfig& config)
	{
		std::stringstream hash;
		hash << stringHash((int)config.otype);
		hash << toStringHash(config.frequencyEncoding);
		hash << toStringHash(config.gridEncoding);
		hash << toStringHash(config.identityEncoding);
		hash << toStringHash(config.oneBlobEncoding);
		hash << toStringHash(config.sphericalHarmonicsEncoding);
		hash << toStringHash(config.triangleWaveEncoding);
		return hash.str();
	};

	std::string toStringHash(const network::config::MainNetEncodingConfig& config)
	{
		std::stringstream hash;
		hash << stringHash(config.normalizePosition);
		hash << toStringHash(config.position);
		hash << toStringHash(config.wo);
		hash << toStringHash(config.normal);
		return hash.str();
	};

	std::string toStringHash(const network::config::MlpSettings& config)
	{
		std::stringstream hash;
		hash << stringHash(config.inputDim);
		hash<<stringHash(config.outputDim);
		hash<<stringHash(config.hiddenDim);
		hash<<stringHash(config.numHiddenLayers);
		hash<<stringHash(config.activationType );
		return hash.str();
	};

	std::string toStringHash(const network::config::AuxNetEncodingConfig& config)
	{
		std::stringstream hash;
		hash << toStringHash(config.wi);
		return hash.str();
	};

	std::string toStringHash(const network::config::NetworkSettings& settings)
	{
		std::stringstream hash;

		//hash << stringHash(settings.active);
		//hash << stringHash(settings.wasActive);
		hash << stringHash(settings.doTraining);
		hash << stringHash(settings.doInference);
		//hash << stringHash(settings.plotGraphs);
		//hash << stringHash(settings.isUpdated);
		//hash << stringHash(settings.depthToDebug);
		hash << stringHash(settings.maxTrainingSteps);
		hash << stringHash(settings.batchSize);
		hash << stringHash(settings.learningRate);
		hash << stringHash(settings.inferenceIterationStart);
		//hash << stringHash(settings.clearOnInferenceStart);
		hash << toStringHash(settings.trainingBatchGenerationSettings);
		hash << toStringHash(settings.mainNetSettings);
		hash << toStringHash(settings.inputSettings);
		hash << stringHash(settings.distributionType);
		hash << stringHash(settings.mixtureSize);
		hash << stringHash(settings.lossType);
		hash << stringHash(settings.lossReduction);
		hash << stringHash(settings.constantBlendFactor);
		if(settings.constantBlendFactor)
		{
			hash << stringHash(settings.blendFactor);
		}
		hash << stringHash(settings.samplingFractionBlend);
		hash << stringHash(settings.useEntropyLoss);
		if(settings.useEntropyLoss)
		{
			hash << stringHash(settings.entropyWeight);
			hash << stringHash(settings.targetEntropy);
		}
		hash << stringHash(settings.useAuxiliaryNetwork);
		if (settings.useAuxiliaryNetwork)
		{
			hash << toStringHash(settings.auxiliaryNetSettings);
			hash << toStringHash(settings.auxiliaryInputSettings);
			hash << stringHash(settings.totAuxInputSize);
			hash << stringHash(settings.inRadianceLossFactor);
			hash << stringHash(settings.outRadianceLossFactor);
			hash << stringHash(settings.throughputLossFactor);
			hash << stringHash(settings.auxiliaryWeight);
			hash << stringHash(settings.radianceTargetScaleFactor);
		}

		hash << stringHash(settings.useTriangleId);
		if(settings.useTriangleId)
		{
			hash << toStringHash(settings.triangleIdEncodingConfig);
		}
		hash << stringHash(settings.useInstanceId);
		if(settings.useInstanceId)
		{
			hash << toStringHash(settings.instanceIdEncodingConfig);
		}
		hash << stringHash(settings.useMaterialId);
		if(settings.useMaterialId)
		{
			hash << toStringHash(settings.materialIdEncodingConfig);
		}
		hash << stringHash(settings.scaleLossBlendedQ);
		hash << stringHash(settings.clampBsdfProb);

		return hash.str();
	}

	std::string toStringHash(const Experiment& exp)
	{
		std::stringstream hash;
		hash << stringHash(exp.rendererSettings.samplingTechnique);
		if(exp.networkSettings.active)
		{
			hash << toStringHash(exp.networkSettings);
		}
		return hash.str();
	}

	void Experiment::constructName(const int experimentNumber)
	{
		name = "Experiment_" + std::to_string(experimentNumber);
	}

	std::string Experiment::getStringHashKey()
	{
		return	toStringHash(*this);
	}

	void ExperimentsManager::loadGroundTruth(const std::string& filePath)
	{
		Image image;
		image.load(filePath);
		float* hostImage = image.getData();
		width = image.getWidth();
		height = image.getHeight();

		groundTruthBuffer.resize(image.getWidth() * image.getHeight() * image.getChannels() * sizeof(float));
		groundTruthBuffer.upload(hostImage, image.getWidth() * image.getHeight() * image.getChannels());
		groundTruth = groundTruthBuffer.castedPointer<math::vec3f>();

		isGroundTruthReady = true;
	}

	void ExperimentsManager::saveGroundTruth(const std::string& filePath)
	{
		std::vector<math::vec3f> hostImage(width * height);
		math::vec3f* hostImagePtr = hostImage.data();
		groundTruthBuffer.download(hostImagePtr);
		Image image;
		image.load((float*)hostImagePtr, width, height, 3);
		image.save(filePath);
	}

	using namespace network::config;
	static std::vector<int> batchSizes = { 64000, 128000, 256000, 512000 };
	static std::vector<float> learningRates = { 0.001f, 0.005f, 0.01f, 0.02f };// these might be problematic , 0.05f, 0.1f
	static std::vector<SamplingStrategy> samplings = { SS_ALL, SS_PATHS_WITH_CONTRIBUTION };// , SS_LIGHT_SAMPLES
	static std::vector<bool> weightByMiss = { true, false };
	static std::vector<float> lightSamplingProbs = { 0.0f, 0.25f, 0.5f, 0.75f, 1.0f };
	static std::vector<bool> normalizePositions = { true, false };
	static std::vector<int> networkHiddenDims = { 32, 64, 128 };
	static std::vector<int> networkLayers = { 3, 4, 5, 8 };
	static std::vector<DistributionType> distributionTypes = { D_SPHERICAL_GAUSSIAN, D_NASG_TRIG, D_NASG_ANGLE,D_NASG_AXIS_ANGLE};
	static std::vector<int> mixtureSizes = { 1, 3, 5 };
	static std::vector<LossType> lossTypes = {L_KL_DIV, L_KL_DIV_MC_ESTIMATION, L_PEARSON_DIV, L_PEARSON_DIV_MC_ESTIMATION};
	static std::vector<EncodingType> positionEncodingType = { EncodingType::Frequency, EncodingType::Grid, EncodingType::Identity,EncodingType::OneBlob, EncodingType::TriangleWave };
	static std::vector<EncodingType> directionEncodingType = { EncodingType::Frequency,EncodingType::Grid,EncodingType::Identity,EncodingType::OneBlob,EncodingType::SphericalHarmonics,EncodingType::TriangleWave };

	// GRID OPTIONS
	static std::vector<GridType> gridTypes = { GridType::Hash,GridType::Tiled };// , GridType::Dense};
	static std::vector<InterpolationType> interpolationTypes = {InterpolationType::Nearest,InterpolationType::Linear, InterpolationType::Smoothstep};
	static std::vector<int> levelCounts = { 8, 16, 32 };
	static std::vector<int> featuresPerLevel = { 1, 2, 4 };
	static std::vector<int> log2HashmapSizes = { 16, 19, 22 };
	static std::vector<int> baseResolutions = { 8, 16, 32 };
	static std::vector<float> perLevelScales = { 1.0, 2.0, 4.0 };

	static std::vector<int> oneBlobBins = { 8, 16, 32 };
	static std::vector<int> sphericalHarmonicsDegrees = { 2, 4, 8 };
	static std::vector<int> triangleWaveFrequencies = { 6, 12, 24 };
	static std::vector<int> frequencyFrequencies = { 6, 12, 24 };

//#define MUTATE(setting, options, option, mutations) \
//	for (auto& x : options) \
//	{\
//		if(setting.option == x) \
//			continue; \
//		mutations.push_back(setting); \
//		mutations.back().option = x; \
//	}\

	template<typename U, typename S>
	void mutate(const S& setting, const std::vector<U>& options, U S::* option, std::vector<S>& mutations) {
		for (const auto& x : options) {
			if (setting.*option == x)
				continue;
			mutations.push_back(setting);
			mutations.back().*option = x;
		}
	}

	template<typename U, typename V, typename J>
	void mutateNested(const J& setting, const std::vector<U>& options, U V::* option, V J::* nestedOption, std::vector<J>& mutations) {
		for (const auto& x : options) {
			if ((setting.*nestedOption).*option == x)
				continue;
			mutations.push_back(setting);
			(mutations.back().*nestedOption).*option = x;
		}
	}

	void mutateEncodingConfig(const EncodingConfig& encodingConfig, std::vector<EncodingConfig>& configs)
	{
		std::vector<FrequencyEncoding> frequencyConfigs;
		std::vector<GridEncoding> gridConfigs;
		std::vector<OneBlobEncoding> oneBlobConfigs;
		std::vector<SphericalHarmonicsEncoding> sphericalHarmonicsConfigs;
		std::vector<TriangleWaveEncoding> triangleWaveConfigs;
		switch(encodingConfig.otype)
		{
		case EncodingType::Frequency: 
			mutate(encodingConfig.frequencyEncoding, frequencyFrequencies, &FrequencyEncoding::n_frequencies, frequencyConfigs);
			mutate(encodingConfig, frequencyConfigs, &EncodingConfig::frequencyEncoding, configs);
			break;
		case EncodingType::Grid: 
			mutate(encodingConfig.gridEncoding, gridTypes, &GridEncoding::type, gridConfigs);
			mutate(encodingConfig.gridEncoding, interpolationTypes, &GridEncoding::interpolation, gridConfigs);
			mutate(encodingConfig.gridEncoding, levelCounts, &GridEncoding::n_levels, gridConfigs);
			mutate(encodingConfig.gridEncoding, featuresPerLevel, &GridEncoding::n_features_per_level, gridConfigs);
			mutate(encodingConfig.gridEncoding, log2HashmapSizes, &GridEncoding::log2_hashmap_size, gridConfigs);
			mutate(encodingConfig.gridEncoding, baseResolutions, &GridEncoding::base_resolution, gridConfigs);
			mutate(encodingConfig.gridEncoding, perLevelScales, &GridEncoding::per_level_scale, gridConfigs);
			mutate(encodingConfig, gridConfigs, &EncodingConfig::gridEncoding, configs);
			break;
		case EncodingType::Identity: 
			break;
		case EncodingType::OneBlob: 
			mutate(encodingConfig.oneBlobEncoding, oneBlobBins, &OneBlobEncoding::n_bins, oneBlobConfigs);
			mutate(encodingConfig, oneBlobConfigs, &EncodingConfig::oneBlobEncoding, configs);
			break;
		case EncodingType::SphericalHarmonics:
			mutate(encodingConfig.sphericalHarmonicsEncoding, sphericalHarmonicsDegrees, &SphericalHarmonicsEncoding::degree, sphericalHarmonicsConfigs);
			mutate(encodingConfig, sphericalHarmonicsConfigs, &EncodingConfig::sphericalHarmonicsEncoding, configs);
			break;
		case EncodingType::TriangleWave:
			mutate(encodingConfig.triangleWaveEncoding, triangleWaveFrequencies, &TriangleWaveEncoding::n_frequencies, triangleWaveConfigs);
			mutate(encodingConfig, triangleWaveConfigs, &EncodingConfig::triangleWaveEncoding, configs);
			break;
		default: ;
		}
	}
	std::vector<NetworkSettings> ExperimentsManager::generateNetworkSettingNeighbors(const NetworkSettings& setting)
	{
		std::vector<NetworkSettings> newNetworkSettings;

		mutateNested(setting, samplings, &BatchGenerationConfig::strategy, &NetworkSettings::trainingBatchGenerationSettings, newNetworkSettings);
		mutateNested(setting, networkLayers, &MlpSettings::numHiddenLayers, &NetworkSettings::mainNetSettings, newNetworkSettings);
		mutateNested(setting, networkHiddenDims, &MlpSettings::hiddenDim, &NetworkSettings::mainNetSettings, newNetworkSettings);

		std::vector<EncodingConfig> normalConfigs;
		mutate(setting.inputSettings.normal, directionEncodingType, &EncodingConfig::otype, normalConfigs);
		mutateEncodingConfig(setting.inputSettings.normal, normalConfigs);
		mutateNested(setting, normalConfigs, &MainNetEncodingConfig::normal, &NetworkSettings::inputSettings, newNetworkSettings);

		std::vector<EncodingConfig> woConfigs;
		mutate(setting.inputSettings.wo, directionEncodingType, &EncodingConfig::otype, woConfigs);
		mutateEncodingConfig(setting.inputSettings.wo, woConfigs);
		mutateNested(setting, woConfigs, &MainNetEncodingConfig::wo, &NetworkSettings::inputSettings, newNetworkSettings);


		std::vector<EncodingConfig> positionConfigs;
		mutate(setting.inputSettings.position, positionEncodingType, &EncodingConfig::otype, positionConfigs);
		mutateEncodingConfig(setting.inputSettings.position, positionConfigs);
		mutateNested(setting, positionConfigs, &MainNetEncodingConfig::position, &NetworkSettings::inputSettings, newNetworkSettings);

		mutateNested(setting, normalizePositions, &MainNetEncodingConfig::normalizePosition , &NetworkSettings::inputSettings, newNetworkSettings);
		mutate(setting, mixtureSizes, &NetworkSettings::mixtureSize, newNetworkSettings);
		mutate(setting, distributionTypes, &NetworkSettings::distributionType, newNetworkSettings);
		mutate(setting, batchSizes, &NetworkSettings::batchSize, newNetworkSettings);
		mutate(setting, learningRates, &NetworkSettings::learningRate, newNetworkSettings);
		mutate(setting, lossTypes, &NetworkSettings::lossType, newNetworkSettings);
		mutateNested(setting, { false, true }, &BatchGenerationConfig::weightByMis, &NetworkSettings::trainingBatchGenerationSettings, newNetworkSettings);

		// ENTROPY LOSS MUTATION
		mutate(setting, { true, false }, &NetworkSettings::useEntropyLoss, newNetworkSettings);
		if(setting.useEntropyLoss)
		{
			mutate(setting, { 0.001f, 0.01f, 0.1f, 1.0f }, &NetworkSettings::entropyWeight, newNetworkSettings);
		}

		// AUXILIARY NETWORK MUTATION
		mutate(setting, { true, false }, &NetworkSettings::useAuxiliaryNetwork, newNetworkSettings);
		if (setting.useAuxiliaryNetwork)
		{
			mutateNested(setting, { 2, 3, 4, 5, 8 }, &MlpSettings::numHiddenLayers, &NetworkSettings::auxiliaryNetSettings, newNetworkSettings);
			mutateNested(setting, { 32, 64, 128 }, &MlpSettings::hiddenDim, &NetworkSettings::auxiliaryNetSettings, newNetworkSettings);
			mutate(setting, {0.01f, 0.1f, 1.0f, 10.f}, &NetworkSettings::inRadianceLossFactor, newNetworkSettings);
			mutate(setting, {0.01f, 0.1f, 1.0f, 10.f}, &NetworkSettings::outRadianceLossFactor, newNetworkSettings);
			mutate(setting, {0.01f, 0.1f, 1.0f, 10.f}, &NetworkSettings::throughputLossFactor, newNetworkSettings);
			mutate(setting, {1.0f, 10.f }, &NetworkSettings::radianceTargetScaleFactor, newNetworkSettings);
			mutate(setting, {16, 32, 64}, &NetworkSettings::totAuxInputSize, newNetworkSettings);
			mutate(setting, { 0.001f, 0.01f, 0.1f, 1.0f }, &NetworkSettings::auxiliaryWeight, newNetworkSettings);
		}
		

		mutate(setting, { true, false }, &NetworkSettings::useTriangleId, newNetworkSettings);
		mutate(setting, { true, false }, &NetworkSettings::useMaterialId, newNetworkSettings);


		mutate(setting, { true, false }, &NetworkSettings::useInstanceId, newNetworkSettings);
		if(setting.useInstanceId)
		{
			std::vector<EncodingConfig> instIdConfigs;
			mutate(setting.instanceIdEncodingConfig, directionEncodingType, &EncodingConfig::otype, instIdConfigs);
			mutateEncodingConfig(setting.instanceIdEncodingConfig, instIdConfigs);
			mutate(setting, instIdConfigs, &NetworkSettings::instanceIdEncodingConfig, newNetworkSettings);
		}
		

		mutate(setting, { true, false }, &NetworkSettings::useMaterialId, newNetworkSettings);
		if (setting.useMaterialId)
		{
			std::vector<EncodingConfig> matIdConfigs;
			mutate(setting.materialIdEncodingConfig, directionEncodingType, &EncodingConfig::otype, matIdConfigs);
			mutateEncodingConfig(setting.materialIdEncodingConfig, matIdConfigs);
			mutate(setting, matIdConfigs, &NetworkSettings::materialIdEncodingConfig, newNetworkSettings);
		}

		mutate(setting, { true, false }, &NetworkSettings::useTriangleId, newNetworkSettings);
		if (setting.useTriangleId)
		{
			std::vector<EncodingConfig> triIdConfigs;
			mutate(setting.triangleIdEncodingConfig, directionEncodingType, &EncodingConfig::otype, triIdConfigs);
			mutateEncodingConfig(setting.triangleIdEncodingConfig, triIdConfigs);
			mutate(setting, triIdConfigs, &NetworkSettings::triangleIdEncodingConfig, newNetworkSettings);
		}

		mutate(setting, { true, false }, &NetworkSettings::scaleLossBlendedQ, newNetworkSettings);
		mutate(setting, { true, false }, &NetworkSettings::clampBsdfProb, newNetworkSettings);
		mutate(setting, { true, false }, &NetworkSettings::samplingFractionBlend, newNetworkSettings);
		mutate(setting, { true, false }, &NetworkSettings::constantBlendFactor, newNetworkSettings);
		if(setting.constantBlendFactor)
		{
			mutate(setting, { 1.0f, 0.99f, 0.9f, 0.8f, 0.5f }, &NetworkSettings::blendFactor, newNetworkSettings);
		}
		mutate(setting, { true, false }, &NetworkSettings::constantBlendFactor, newNetworkSettings);

		//mutateNested(setting, {0.0f, 0.5f, 0.75f, 1.0f}, &BatchGenerationConfig::lightSamplingProb, & NetworkSettings::trainingBatchGenerationSettings, newNetworkSettings);

		std::shuffle(newNetworkSettings.begin(), newNetworkSettings.end(), g);
		return newNetworkSettings;
	}

	std::vector<Experiment> ExperimentsManager::generateExperimentNeighbors(const Experiment& experiment)
	{
		std::vector<Experiment>            newExperiments;
		const std::vector<NetworkSettings> networkSettings = generateNetworkSettingNeighbors(experiment.networkSettings);
		for(const auto& networkSetting : networkSettings)
		{
			Experiment newExperiment = experiment;
			newExperiment.networkSettings = networkSetting;
			//newExperiment.rendererSettings.samplingTechnique = S_MIS;
			//newExperiments.push_back(newExperiment);
			newExperiment.rendererSettings.samplingTechnique = S_BSDF;
			newExperiments.push_back(newExperiment);
		}
		return newExperiments;
	}

	NetworkSettings ExperimentsManager::getBestGuess()
	{
		NetworkSettings bestGuessNetworkSettings;
		bestGuessNetworkSettings.active = true;
		bestGuessNetworkSettings.doTraining = true;
		bestGuessNetworkSettings.doInference = true;
		bestGuessNetworkSettings.plotGraphs = true;
		bestGuessNetworkSettings.isUpdated = true;
		bestGuessNetworkSettings.depthToDebug = 0;
		bestGuessNetworkSettings.maxTrainingSteps = 1000;
		bestGuessNetworkSettings.inferenceIterationStart = 1;
		bestGuessNetworkSettings.clearOnInferenceStart = false;


		bestGuessNetworkSettings.useTriangleId = false;
		bestGuessNetworkSettings.useInstanceId = false;
		bestGuessNetworkSettings.useMaterialId = true;

		bestGuessNetworkSettings.mainNetSettings.hiddenDim = 128;
		bestGuessNetworkSettings.mainNetSettings.numHiddenLayers = 3;

		bestGuessNetworkSettings.inputSettings.normalizePosition = true;
		bestGuessNetworkSettings.inputSettings.position.otype = EncodingType::Frequency;
		bestGuessNetworkSettings.inputSettings.position.frequencyEncoding.n_frequencies = 6;

		bestGuessNetworkSettings.inputSettings.normal.otype = EncodingType::Grid;
		bestGuessNetworkSettings.inputSettings.normal.gridEncoding.type = GridType::Tiled;
		bestGuessNetworkSettings.inputSettings.normal.gridEncoding.n_levels = 16;
		bestGuessNetworkSettings.inputSettings.normal.gridEncoding.n_features_per_level = 1;
		bestGuessNetworkSettings.inputSettings.normal.gridEncoding.log2_hashmap_size = 19;
		bestGuessNetworkSettings.inputSettings.normal.gridEncoding.base_resolution = 16;
		bestGuessNetworkSettings.inputSettings.normal.gridEncoding.per_level_scale = 4.0f;
		bestGuessNetworkSettings.inputSettings.normal.gridEncoding.interpolation = InterpolationType::Smoothstep;


		bestGuessNetworkSettings.inputSettings.wo.otype = EncodingType::Frequency;
		bestGuessNetworkSettings.inputSettings.wo.frequencyEncoding.n_frequencies = 12;

		bestGuessNetworkSettings.inputSettings.wo.gridEncoding.type = GridType::Hash;
		bestGuessNetworkSettings.inputSettings.wo.gridEncoding.n_levels = 32;
		bestGuessNetworkSettings.inputSettings.wo.gridEncoding.n_features_per_level = 4;
		bestGuessNetworkSettings.inputSettings.wo.gridEncoding.log2_hashmap_size = 19;
		bestGuessNetworkSettings.inputSettings.wo.gridEncoding.base_resolution = 32;
		bestGuessNetworkSettings.inputSettings.wo.gridEncoding.per_level_scale = 2.0f;
		bestGuessNetworkSettings.inputSettings.wo.gridEncoding.interpolation = InterpolationType::Nearest;


		bestGuessNetworkSettings.distributionType = D_NASG_AXIS_ANGLE;
		bestGuessNetworkSettings.mixtureSize = 3;// 5;// 3;

		//AUXILIARY NETWORK SETTINGS

		bestGuessNetworkSettings.useAuxiliaryNetwork = false;// false;// false;
		bestGuessNetworkSettings.totAuxInputSize = 64;
		bestGuessNetworkSettings.auxiliaryNetSettings.hiddenDim = 64;
		bestGuessNetworkSettings.auxiliaryNetSettings.numHiddenLayers = 3;

		bestGuessNetworkSettings.auxiliaryInputSettings.wi.otype = EncodingType::Grid;
		bestGuessNetworkSettings.auxiliaryInputSettings.wi.gridEncoding.type = GridType::Hash;
		bestGuessNetworkSettings.auxiliaryInputSettings.wi.gridEncoding.n_levels = 16;
		bestGuessNetworkSettings.auxiliaryInputSettings.wi.gridEncoding.n_features_per_level = 2;
		bestGuessNetworkSettings.auxiliaryInputSettings.wi.gridEncoding.log2_hashmap_size = 19;
		bestGuessNetworkSettings.auxiliaryInputSettings.wi.gridEncoding.base_resolution = 16;
		bestGuessNetworkSettings.auxiliaryInputSettings.wi.gridEncoding.per_level_scale = 2.0f;
		bestGuessNetworkSettings.auxiliaryInputSettings.wi.gridEncoding.interpolation = InterpolationType::Linear;

		bestGuessNetworkSettings.inRadianceLossFactor = 1.0f;
		bestGuessNetworkSettings.outRadianceLossFactor = 1.0f;
		bestGuessNetworkSettings.throughputLossFactor = 1.0f;
		bestGuessNetworkSettings.auxiliaryWeight = 0.001f;

		// TRAINING SETTINGS
		bestGuessNetworkSettings.learningRate = 0.005f;
		bestGuessNetworkSettings.batchSize = 256000;// 64000;
		bestGuessNetworkSettings.trainingBatchGenerationSettings.strategy = SS_ALL;
		bestGuessNetworkSettings.trainingBatchGenerationSettings.lightSamplingProb = 0.0f;
		bestGuessNetworkSettings.trainingBatchGenerationSettings.weightByMis = false;
		bestGuessNetworkSettings.clampBsdfProb = false;
		bestGuessNetworkSettings.scaleLossBlendedQ = false;
		bestGuessNetworkSettings.blendFactor = 1.0f;
		bestGuessNetworkSettings.constantBlendFactor = false;
		bestGuessNetworkSettings.samplingFractionBlend = false;
		bestGuessNetworkSettings.lossType = L_KL_DIV_MC_ESTIMATION;
		bestGuessNetworkSettings.lossReduction = MEAN;

		bestGuessNetworkSettings.useEntropyLoss = true;
		bestGuessNetworkSettings.entropyWeight = 0.001f;
		bestGuessNetworkSettings.targetEntropy = 3.0f;



		return bestGuessNetworkSettings;
	}
	std::tuple<Experiment, Experiment, Experiment, Experiment, Experiment>
	ExperimentsManager::startingConfigExperiments(const std::shared_ptr<graph::Renderer>& renderer)
	{
		Experiment groundTruthExp;
		groundTruthExp.rendererSettings = renderer->settings;
		groundTruthExp.rendererSettings.samplingTechnique = S_MIS;
		groundTruthExp.rendererSettings.maxBounces = 8;
		groundTruthExp.rendererSettings.maxSamples = gtSamples;
		groundTruthExp.wavefrontSettings.active = true;
		groundTruthExp.networkSettings = getBestGuess();
		groundTruthExp.networkSettings.active = false;

		Experiment bsdfExperiment = groundTruthExp;
		bsdfExperiment.rendererSettings.samplingTechnique = S_BSDF;
		bsdfExperiment.rendererSettings.maxSamples = testSamples;

		Experiment misExperiment = groundTruthExp;
		misExperiment.rendererSettings.maxSamples = testSamples;

		Experiment bestGuessBsdf = bsdfExperiment;
		bestGuessBsdf.networkSettings.active = true;

		Experiment bestGuessMis = misExperiment;
		bestGuessBsdf.networkSettings.active = true;

		return { groundTruthExp, misExperiment, bsdfExperiment, bestGuessBsdf, bestGuessMis };
	}

#define COMBINE_NET_PARAM(options, option, newVector, oldVector) \
	for(auto& setting: (oldVector)) \
	{\
		for (auto& x : options) \
		{\
			setting.option = x;\
			newVector.push_back(setting);\
		}\
	} \
	std::swap(oldVector, newVector); \
	newVector.clear(); \

	std::vector<NetworkSettings> ExperimentsManager::generateNetworkSettingsCombination()
	{
		NetworkSettings networkSettings;
		networkSettings.active = true;
		std::vector<NetworkSettings> oldNetworkSettings = { networkSettings };
		std::vector<NetworkSettings> newNetworkSettings;

		COMBINE_NET_PARAM(batchSizes, batchSize, newNetworkSettings, oldNetworkSettings);
		COMBINE_NET_PARAM(learningRates, learningRate, newNetworkSettings, oldNetworkSettings);
		//COMBINE_NET_PARAM(samplings, trainingBatchGenerationSettings.strategy, newNetworkSettings, oldNetworkSettings);
		//COMBINE_NET_PARAM(weightByMiss, trainingBatchGenerationSettings.weightByMis, newNetworkSettings, oldNetworkSettings);
		//COMBINE_NET_PARAM(lightSamplingProbs, trainingBatchGenerationSettings.lightSamplingProb, newNetworkSettings, oldNetworkSettings);
		//COMBINE_NET_PARAM(networkHiddenDims, hiddenDim, newNetworkSettings, oldNetworkSettings);
		//COMBINE_NET_PARAM(networkLayers, numHiddenLayers, newNetworkSettings, oldNetworkSettings);
		COMBINE_NET_PARAM(distributionTypes, distributionType, newNetworkSettings, oldNetworkSettings);
		COMBINE_NET_PARAM(mixtureSizes, mixtureSize, newNetworkSettings, oldNetworkSettings);
		//COMBINE_NET_PARAM(lossTypes, lossType, newNetworkSettings, oldNetworkSettings);
		COMBINE_NET_PARAM(directionEncodingType, inputSettings.normal.otype, newNetworkSettings, oldNetworkSettings);
		COMBINE_NET_PARAM(directionEncodingType, inputSettings.wo.otype, newNetworkSettings, oldNetworkSettings);

		//COMBINE_NET_PARAM(positionEncodingType, inputSettings.position.otype, newNetworkSettings, oldNetworkSettings);
		//COMBINE_NET_PARAM(normalizePositions, normalizePosition, newNetworkSettings, oldNetworkSettings);

		for (auto& setting : (oldNetworkSettings))
		{
			for (auto& x : positionEncodingType)
			{
				setting.inputSettings.position.otype = x;
				setting.inputSettings.normalizePosition = true;
				newNetworkSettings.push_back(setting);
				setting.inputSettings.normalizePosition = false;
				newNetworkSettings.push_back(setting);
			}
		};
		//std::swap(oldNetworkSettings, newNetworkSettings);
		//newNetworkSettings.clear();

		return newNetworkSettings;
	}

	std::pair<Experiment, std::vector<Experiment>> ExperimentsManager::generateExperiments(const std::shared_ptr<graph::Renderer>& renderer)
	{
		Experiment groundTruthExp;
		groundTruthExp.rendererSettings = renderer->settings;
		groundTruthExp.rendererSettings.samplingTechnique = S_MIS;
		groundTruthExp.rendererSettings.maxBounces = 8;
		groundTruthExp.rendererSettings.maxSamples = gtSamples;
		groundTruthExp.wavefrontSettings.active = true;
		groundTruthExp.networkSettings.active = false;

		Experiment bsdfExperiment = groundTruthExp;
		bsdfExperiment.rendererSettings.samplingTechnique = S_BSDF;
		bsdfExperiment.rendererSettings.maxSamples = testSamples;

		Experiment misExperiment = groundTruthExp;
		misExperiment.rendererSettings.maxSamples = testSamples;

		const std::vector<NetworkSettings> newNetworkSettings = generateNetworkSettingsCombination();
		std::vector<Experiment> experimentList;

		for (auto exp : {bsdfExperiment, misExperiment })
		{
			exp.constructName(experimentList.size());
			experimentList.push_back(exp);
			for (auto& networkSetting : newNetworkSettings)
			{
				Experiment experiment = exp;
				experiment.networkSettings = networkSetting;
				experiment.constructName(experimentList.size());
				experimentList.push_back(experiment);
			}
		}

		return std::make_pair(groundTruthExp, experimentList);
	}


	void ExperimentsManager::setupNewExperiment(const Experiment& experiment, const std::shared_ptr<graph::Renderer>& renderer)
	{
		const auto tmpDisplayBuffer = renderer->settings.displayBuffer;
		renderer->settings = experiment.rendererSettings;
		renderer->settings.displayBuffer = tmpDisplayBuffer;
		renderer->settings.isUpdated = true;
		renderer->waveFrontIntegrator.settings = experiment.wavefrontSettings;
		renderer->waveFrontIntegrator.settings.isUpdated = true;
		renderer->waveFrontIntegrator.network.settings = experiment.networkSettings;
		renderer->waveFrontIntegrator.network.settings.isUpdated = true;
		renderer->waveFrontIntegrator.network.settings.isUpdated = true;
		renderer->waveFrontIntegrator.network.settings.maxTrainingSteps = testSamples;
		renderer->waveFrontIntegrator.network.reset();
		renderer->restart();
	}

	void ExperimentsManager::generateGroundTruth(const Experiment& gtExperiment, Application* app, const std::shared_ptr<graph::Renderer>& renderer)
	{
		// GROUND TRUTH GENERATION
		VTX_WARN("GROUND TRUTH GENERATION");
		setupNewExperiment(gtExperiment, renderer);
		for (int i = 0; i < renderer->settings.maxSamples; i++)
		{
			app->batchExperimentAppLoopBody(i, renderer);
		}
		{
			const size_t imageSize = (size_t)width * (size_t)height * sizeof(math::vec3f);

			groundTruthBuffer.resize(imageSize);
			groundTruth = groundTruthBuffer.castedPointer<math::vec3f>();
			const void* groundTruthRendered = onDeviceData->launchParamsData.getHostImage().frameBuffer.tmRadiance;

			cudaError_t        error = cudaMemcpy((void*)groundTruth, groundTruthRendered, imageSize, cudaMemcpyDeviceToDevice);
			groundTruthHost = std::vector<float>((size_t)width * (size_t)height * 3);
			onDeviceData->frameBufferData.resourceBuffers.tmRadiance.download(groundTruthHost.data());
			isGroundTruthReady = true;

			currentExperimentStep = 0;
		}
	}

	bool ExperimentsManager::performExperiment(Experiment& experiment, Application* app, const std::shared_ptr<graph::Renderer>& renderer) {

		constexpr int      maxRetries = 3;
		const std::string& name       = experiment.name;
		constexpr int      nTries     = 3;
		int                tryCount   = 0;
		while (tryCount < nTries)
		{
			try
			{
				VTX_WARN("EXPERIMENT {} Try {}", experiment.name, tryCount);
				experiment.mape.clear();
				setupNewExperiment(experiment, renderer);

				float mapeSum = 0.f;
				for (int i = 0; i < renderer->settings.maxSamples; i++)
				{
					app->batchExperimentAppLoopBody(i, renderer);
					const float mape = cuda::computeMape(groundTruth, onDeviceData->launchParamsData.getHostImage().frameBuffer.tmRadiance, width, height);
					experiment.mape.push_back(mape);
					mapeSum += mape;
					if (glfwWindowShouldClose(app->glfwWindow))
						return false;
				}
				experiment.averageMape = mapeSum / (float)renderer->settings.maxSamples;
				return true;
			}
			catch (std::exception& e)
			{
				VTX_ERROR("Standard exception in experiment {}: {}", name, e.what());
			}
			catch (...)
			{
				VTX_ERROR("Unknown exception in experiment {}", name);
			}
			tryCount++;
		}
		
		return false;
	}

	void ExperimentsManager::refillExperimentQueue()
	{
		bool firstRun = true;
		while ((firstRun || experimentQueue.empty()) && !experimentMinHeap.empty())
		{
			firstRun = false;
			int bestIndex = -1;
			while (!experimentMinHeap.empty())
			{
				const int minIndex = experimentMinHeap.top().second; experimentMinHeap.pop();
				if (experiments[minIndex].networkSettings.active && experiments[minIndex].rendererSettings.samplingTechnique != S_MIS)
				{
					bestIndex = minIndex;
					break;
				}
			}
			if (bestIndex == -1)
			{
				VTX_WARN("No more experiments to run");
				break;
			}
			Experiment best = experiments[bestIndex];
			auto variations = generateExperimentNeighbors(best);

			VTX_INFO("Adding {} variations to the queue front", variations.size());
			for (auto& neighbor : variations)
			{
				if (experimentSet.count(neighbor.getStringHashKey()) == 0)
				{
					experimentQueue.push_front(neighbor);
				}
				else
				{
					VTX_WARN("SKIPPING BECAUSE ALREADY IN SET");
				}
			}
		}
	}

	void ExperimentsManager::BatchExperimentRun()
	{
		VTX_INFO("BATCH EXPERIMENT RUN");
		Application* app = Application::get();
		const graph::Scene* scene = graph::Scene::get();
		const std::shared_ptr<graph::Renderer>& renderer = scene->renderer;
		auto& net = renderer->waveFrontIntegrator.network;
		const std::shared_ptr<ExperimentsWindow> ew = app->windowManager->getWindow<ExperimentsWindow>();
		const LaunchParams& lPHost = onDeviceData->launchParamsData.getHostImage();
		const CUDABuffer& renderedImageBuffer = onDeviceData->frameBufferData.resourceBuffers.tmRadiance;

		renderer->isSizeLocked = false;
		renderer->resize(width, height);
		renderer->isSizeLocked = true;

		renderer->camera->lockCamera = false;
		renderer->camera->resize(width, height);
		renderer->camera->lockCamera = true;


		// for each Scene
		{
			auto [gtExperiment, misExperiment, bsdfExperiment, bestGuessBsdf, bestGuessMis] = startingConfigExperiments(renderer);
			//experimentQueue.push_back(misExperiment);
			//experimentQueue.push_back(bestGuessMis);
			if (experimentSet.count(bsdfExperiment.getStringHashKey()) == 0)
			{
				experimentQueue.push_back(bsdfExperiment);
			}
			if (experimentSet.count(bestGuessBsdf.getStringHashKey()) == 0)
			{
				experimentQueue.push_back(bestGuessBsdf);
			}

			if(!(isGroundTruthReady && groundTruthBuffer.bytesSize()!=0))
			{
				generateGroundTruth(gtExperiment, app, renderer);
				Image(renderedImageBuffer, lPHost.frameBuffer.frameSize.x, lPHost.frameBuffer.frameSize.y, 3).save(getImageSavePath("groundTruth.png"));
				experimentSet.insert(gtExperiment.getStringHashKey());
			}
			else
			{
				groundTruth = groundTruthBuffer.castedPointer<math::vec3f>();
			}

			if(experimentQueue.empty())
			{
				refillExperimentQueue();
			}

			while (!experimentQueue.empty() && !glfwWindowShouldClose(app->glfwWindow))
			{
				const int queueSize = experimentQueue.size();
				for (int i = 0; i < queueSize; i++)
				{
					experiments.push_back(experimentQueue.front()); experimentQueue.pop_front();
					auto& experiment = experiments.back();
					experiment.constructName(experiments.size());
					experiment.generatedByBatchExperiments = true;

					const bool success = performExperiment(experiment, app, renderer);

					if(success)
					{
						Image(renderedImageBuffer, lPHost.frameBuffer.frameSize.x, lPHost.frameBuffer.frameSize.y, 3).save(getImageSavePath(experiment.name));
						experiment.statistics = renderer->statistics;
						float mapeScore = experiment.averageMape + experiment.mape.back();
						experimentMinHeap.push({ mapeScore, experiments.size() - 1 });
						experimentSet.insert(experiment.getStringHashKey());
						experiment.completed = true;
						VTX_INFO("\tExperiment {} finished with MAPE {} Overall {} best Overall Mape is {}", experiment.name, experiment.averageMape, mapeScore, bestMapeScore);
						if(mapeScore < bestMapeScore && experiment.networkSettings.active)
						{
							VTX_INFO("\t\tNew best experiment {} with MAPE {}", experiment.name, experiment.averageMape);
							VTX_INFO("\t\tBreaking and Searching from this!");
							bestMapeScore = mapeScore;
							bestExperimentIndex = experiments.size() - 1;
							vtx::serializer::serializeBatchExperiments(saveFilePath);
							experiment.displayExperiment = true;
							break;
						}
						else if (experiment.networkSettings.active)
						{
							experiment.displayExperiment = false;
						}
					}
					else
					{
						VTX_ERROR("Experiment {} failed", experiment.name);
						experiment.completed = false;
						experiment.displayExperiment = false;
					}
					vtx::serializer::serializeBatchExperiments(saveFilePath);

					if (glfwWindowShouldClose(app->glfwWindow))
					{
						break;
					}
				}

				refillExperimentQueue();
			}
		}
		renderer->isSizeLocked = false;
		renderer->camera->lockCamera = false;
	}

	std::string ExperimentsManager::getImageSavePath(std::string experimentName)
	{
		std::string savePath = utl::getFolder(saveFilePath) + "/Images/Experiment_" + experimentName + ".png";
		return utl::absolutePath(savePath);
	}


}
