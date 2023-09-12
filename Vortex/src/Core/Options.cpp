#include "Options.h"
#include "Device/DevicePrograms/LaunchParams.h"
#include "NeuralNetworks/NetworkSettings.h"

namespace vtx {

	static Options options;

	void startOptions()
	{

		options.width = 2100;
		options.height = 900;
		options.windowName = "Vortex";
		options.dataFolder = "E:/Dev/VortexOptix/data/";
		options.imGuiIniFile = options.dataFolder + "ImGui.ini";
		options.dllPath = "./";
		options.clearColor[4] = (0.45f, 0.55f, 0.60f, 1.00f);
#ifdef NDEBUG
		options.isDebug = false;
#else
		options.isDebug = true;
#endif
		////////////////////////////////////////////////////////////////////////////////////
		/////////////////// Rendering Settings /////////////////////////////////////////////
		////////////////////////////////////////////////////////////////////////////////////

		options.rendererSettings.iteration = -1;
		options.rendererSettings.maxBounces = 10;
		options.rendererSettings.maxSamples = 100000;
		options.rendererSettings.accumulate = true;
		options.rendererSettings.samplingTechnique = S_MIS;
		options.rendererSettings.displayBuffer = FB_DEBUG_1;
		options.rendererSettings.minClamp = 0.0001f;
		options.rendererSettings.maxClamp = 1000.0f;
		options.rendererSettings.useRussianRoulette = true;
		options.rendererSettings.runOnSeparateThread = false;
		options.rendererSettings.isUpdated = false;

		options.rendererSettings.adaptiveSamplingSettings.active = false;
		options.rendererSettings.adaptiveSamplingSettings.noiseKernelSize = 3;
		options.rendererSettings.adaptiveSamplingSettings.minAdaptiveSamples = 100;
		options.rendererSettings.adaptiveSamplingSettings.minPixelSamples = 1;
		options.rendererSettings.adaptiveSamplingSettings.maxPixelSamples = 1;
		options.rendererSettings.adaptiveSamplingSettings.albedoNormalNoiseInfluence = 1.0f;
		options.rendererSettings.adaptiveSamplingSettings.noiseCutOff = 0.1f;
		options.rendererSettings.adaptiveSamplingSettings.isUpdated = false;

		options.rendererSettings.fireflySettings.kernelSize = 3;
		options.rendererSettings.fireflySettings.threshold = 2.0f;
		options.rendererSettings.fireflySettings.active = false;
		options.rendererSettings.fireflySettings.isUpdated = false;

		options.rendererSettings.denoiserSettings.active = false;
		options.rendererSettings.denoiserSettings.denoiserStart = 10;
		options.rendererSettings.denoiserSettings.denoiserBlend = 0.1f;
		options.rendererSettings.denoiserSettings.isUpdated = false;

		options.rendererSettings.toneMapperSettings.whitePoint = { 1.0f, 1.0f, 1.0f };
		options.rendererSettings.toneMapperSettings.invWhitePoint = { 1.0f, 1.0f, 1.0f };
		options.rendererSettings.toneMapperSettings.colorBalance = { 1.0f, 1.0f, 1.0f };
		options.rendererSettings.toneMapperSettings.burnHighlights = 0.0f;
		options.rendererSettings.toneMapperSettings.crushBlacks = 1.0f;
		options.rendererSettings.toneMapperSettings.saturation = 1.0f;
		options.rendererSettings.toneMapperSettings.gamma = 2.2f;
		options.rendererSettings.toneMapperSettings.invGamma = 1.0f / 2.2f;
		options.rendererSettings.toneMapperSettings.isUpdated = false;

		options.wavefrontSettings.active = true;
		options.wavefrontSettings.fitWavefront = false;
		options.wavefrontSettings.optixShade = false;
		options.wavefrontSettings.parallelShade = false;
		options.wavefrontSettings.longPathPercentage = 0.25f;
		options.wavefrontSettings.useLongPathKernel = false;
		options.wavefrontSettings.isUpdated  = false;

		////////////////////////////////////////////////////////////////////////////////////
		/////////////////// Neural Network Options /////////////////////////////////////////
		////////////////////////////////////////////////////////////////////////////////////
		options.networkSettings.active = false; // 10;//
		options.networkSettings.batchSize = 100000; // 10;//
		options.networkSettings.maxTrainingStepPerFrame = 1;
		options.networkSettings.doInference = true;
		options.networkSettings.doTraining = true;
		options.networkSettings.maxTrainingSteps = 1000;
		options.networkSettings.inferenceIterationStart = 1;
		options.networkSettings.clearOnInferenceStart = true;
		options.networkSettings.type = network::NetworkType::NT_NGP;

		options.networkSettings.trainingBatchGenerationSettings.lightSamplingProb = 0.0f;
		options.networkSettings.trainingBatchGenerationSettings.weightByMis= true;
		options.networkSettings.trainingBatchGenerationSettings.strategy = network::SS_PATHS_WITH_CONTRIBUTION;

		options.networkSettings.inputSettings.positionEncoding.type= network::E_NONE;
		options.networkSettings.inputSettings.positionEncoding.features = 3;
		options.networkSettings.inputSettings.woEncoding.type = network::E_NONE;
		options.networkSettings.inputSettings.woEncoding.features = 3;
		options.networkSettings.inputSettings.normalEncoding.type = network::E_NONE;
		options.networkSettings.inputSettings.normalEncoding.features   = 3;

		options.networkSettings.pathGuidingSettings.hiddenDim               = 64;
		options.networkSettings.pathGuidingSettings.numHiddenLayers         = 4;
		options.networkSettings.pathGuidingSettings.distributionType        = network::D_NASG_AXIS_ANGLE;
		options.networkSettings.pathGuidingSettings.produceSamplingFraction = false;
		options.networkSettings.pathGuidingSettings.mixtureSize				= 3;

		options.networkSettings.sac.polyakFactor = 0.0001f;
		options.networkSettings.sac.logAlphaStart = 0.0f;
		options.networkSettings.sac.gamma = 0.0f;
		options.networkSettings.sac.neuralSampleFraction = 0.9f;
		options.networkSettings.sac.policyLr = 0.00001f;
		options.networkSettings.sac.qLr = 0.00001f;
		options.networkSettings.sac.alphaLr = 0.00001f;

		options.networkSettings.npg.learningRate = 0.020f;
		options.networkSettings.npg.e = 0.8f;
		options.networkSettings.npg.constantBlendFactor = false;
		options.networkSettings.npg.samplingFractionBlend = false;
		options.networkSettings.npg.lossType = network::L_KL_DIV_MC_ESTIMATION;
		options.networkSettings.npg.meanLoss = false;
		options.networkSettings.npg.absoluteLoss = false;


		////////////////////////////////////////////////////////////////////////////////////
		/////////////////// Optix Options //////////////////////////////////////////////////
		////////////////////////////////////////////////////////////////////////////////////
		options.OptixVersion = OPTIX_VERSION;
		options.deviceID = 0;
		options.maxDcDepth = 2;
		options.maxTraversableGraphDepth = 2;
		options.LaunchParamName = "optixLaunchParams";
		options.enableCache = true;

		////////////////////////////////////////////////////////////////////////////////////
		/////////////////// MDL Options ////////////////////////////////////////////////////
		////////////////////////////////////////////////////////////////////////////////////
		options.mdlSearchPath = { options.dataFolder, "./" , "E:/Dev/OptixTut/bin/Debug-windows-x86_64/OptixApp/mdl" };
		options.numTextureSpaces = 1; //should be set to 1 for performance reasons If you do not use the hair BSDF.
		options.numTextureResults = 16;
		options.enable_derivatives = false;
		options.mdlOptLevel = "2";
		options.initialized = true;
		options.mdlCallType = MDL_CUDA;

		////////////////////////////////////////////////////////////////////////////////////
		/////////////////// Gui Options ////////////////////////////////////////////////////
		////////////////////////////////////////////////////////////////////////////////////
		options.nodeWidth = 300;
		options.fontPath = options.dataFolder + "fonts/LucidaGrande.ttf";
	}
	Options* getOptions()
	{
		if(!options.initialized)
		{
			startOptions();
		}
		return &options;
	}

}
