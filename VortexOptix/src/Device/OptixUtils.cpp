#include <algorithm>

#include "OptixUtils.h"
#include "Core/Log.h"
#include "CUDAChecks.h"
#include "Core/Options.h"
#include "ShadersDefinitions.h"
#include "Core/Utils.h"

#undef max;
#undef min;

namespace vtx::optix
{

	/*SBT Record Template*/
	struct SbtRecordHeader
	{
		__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	};

	static State state;

	static OptixRenderingPipelineData orp;

	State* getState()
	{
		if (state.isValid) {
			return &state;
		}
		else {
			init();
			return &state;
		}
	}

	OptixRenderingPipelineData* getRenderingPipeline()
	{
		if (state.isValid) {
			return &orp;
		}
		else {
			createRenderingPipeline();
			return &orp;
		}
	}

	void init()
	{
		if (state.isValid) return;
		startOptix();
		createContext();
		setModuleCompilersOptions();
		setPipelineCompilersOptions();
		setPipelineLinkOptions();
		state.isValid = true;
	}

	void startOptix()
	{
		VTX_INFO("Initializing optix");
		cudaFree(0);
		int numDevices;
		cudaGetDeviceCount(&numDevices);
		if (numDevices == 0)
		{
			VTX_ERROR("no CUDA capable devices found!");
			throw std::runtime_error("no CUDA capable devices found!");
		}
		VTX_INFO("found {} CUDA devices", numDevices);

		OPTIX_CHECK(optixInit());
		VTX_INFO("Optix Initialized");
	}

	void createContext()
	{
		VTX_INFO("Creating Optix Context");
		int deviceID = getOptions()->deviceID;
		checkCudaError(cudaSetDevice(deviceID));
		CUDA_CHECK(cudaStreamCreate(&state.stream));

		cudaGetDeviceProperties(&state.deviceProps, deviceID);
		VTX_INFO("running on device: {}", state.deviceProps.name);

		CUresult  cuRes = cuCtxGetCurrent(&state.cudaContext);
		if (cuRes != CUDA_SUCCESS)
			fprintf(stderr, "Error querying current context: error code %d\n", cuRes);

		OPTIX_CHECK(optixDeviceContextCreate(state.cudaContext, 0, &state.optixContext));
		OPTIX_CHECK(optixDeviceContextSetLogCallback(state.optixContext, context_log_cb, nullptr, 4));
	}

	void setModuleCompilersOptions()
	{
		VTX_INFO("Setting Module Compiler Options");
		// OptixModuleCompileOptions
		state.moduleCompileOptions = {};

		state.moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
		if (getOptions()->isDebug) {
			state.moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0; // No optimizations.
			state.moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;     // Full debug. Never profile kernels with this setting!
		}
		else {
			state.moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3; // All optimizations, is the default.
			state.moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
			//state.moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO; // To activate if OPTIX VERSION is < 70400
		}
	}

	void setPipelineCompilersOptions()
	{
		// OptixPipelineCompileOptions
		VTX_INFO("Setting Pipeline Compiler Options");
		state.pipelineCompileOptions = {};

		state.pipelineCompileOptions.usesMotionBlur = false;
		state.pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
		state.pipelineCompileOptions.numPayloadValues = 2;  // I need two to encode a 64-bit pointer to the per ray payload structure.
		state.pipelineCompileOptions.numAttributeValues = 2;  // The minimum is two for the triangle barycentrics.
		if (getOptions()->isDebug) {
			state.pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG;
			state.pipelineCompileOptions.exceptionFlags =
				OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW |
				OPTIX_EXCEPTION_FLAG_TRACE_DEPTH |
				OPTIX_EXCEPTION_FLAG_USER |
				OPTIX_EXCEPTION_FLAG_DEBUG;
		}
		else
			state.pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
		state.pipelineCompileOptions.pipelineLaunchParamsVariableName = LaunchParamName.c_str();
		if (getOptions()->OptixVersion != 70000)
			state.pipelineCompileOptions.usesPrimitiveTypeFlags =
				OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE |
				OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_CUBIC_BSPLINE;
	}

	void setPipelineLinkOptions()
	{
		// OptixPipelineLinkOptions
		VTX_INFO("Setting Pipeline Linker Options");
		state.pipelineLinkOptions = {};
		state.pipelineLinkOptions.maxTraceDepth = 2;
		//if (options.isDebug) {
		//	pipelineLinkOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL; // Full debug. Never /profile /kernels with this setting!
		//}
		//else {
		//	pipelineLinkOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
		//	//m_plo.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO; // Version < 7.4
		//}
	}

	void createRenderingPipeline()
	{
		if(orp.isValid) return;
		CreateModules();
		CreatePrograms();
		CreatePipeline();
		SetStackSize();
		CreateSBT();
		orp.isValid = true;
	}

	void CreateModules()
	{
		VTX_INFO("Creating Modules");
		// Each source file results in one OptixModule.
		orp.modules.resize(NUM_MODULE_IDENTIFIERS);

		// Create all orp.modules:
		for (int moduleID = 0; moduleID < NUM_MODULE_IDENTIFIERS; moduleID++) {

			std::string& modulePath = utl::absolutePath(MODULE_FILENAME[(ModuleIdentifier)moduleID]);
			std::vector<char> programData = utl::readData(modulePath);

			char log[2048];
			size_t sizeof_log = sizeof(log);
			OPTIX_CHECK(optixModuleCreate(state.optixContext,
				&state.moduleCompileOptions,
				&state.pipelineCompileOptions,
				programData.data(),
				programData.size(),
				log, &sizeof_log,
				&orp.modules[moduleID]));
			if (sizeof_log > 1) {
				VTX_WARN(log);
			}

		}
	}

	void CreatePrograms()
	{
		VTX_INFO("Creating Programs");
		OptixProgramGroupOptions pgOptions = {};

		std::vector<OptixProgramGroupDesc>	programGroupDescriptions(NUM_PROGRAM_GROUPS);
		orp.programGroups.resize(NUM_PROGRAM_GROUPS);
		OptixProgramGroupDesc* pgd;

		// Raygen program:
		pgd = &programGroupDescriptions[PROGRAMGROUP_ID_RAYGEN];
		pgd->kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
		pgd->raygen.module = orp.modules[funcPropMap[__RAYGEN__RENDERFRAME].module];
		pgd->raygen.entryFunctionName = funcPropMap[__RAYGEN__RENDERFRAME].name;


		pgd = &programGroupDescriptions[PROGRAMGROUP_ID_ECXEPTION];
		pgd->kind = OPTIX_PROGRAM_GROUP_KIND_EXCEPTION;
		pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
		pgd->exception.module = orp.modules[funcPropMap[__EXCEPTION__ALL].module];
		pgd->exception.entryFunctionName = funcPropMap[__EXCEPTION__ALL].name;

		// Miss program:
		pgd = &programGroupDescriptions[PROGRAMGROUP_ID_MISS];
		pgd->kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
		pgd->raygen.module = orp.modules[funcPropMap[__MISS__RADIANCE].module];
		pgd->raygen.entryFunctionName = funcPropMap[__MISS__RADIANCE].name;

		//Hit Program
		pgd = &programGroupDescriptions[PROGRAMGROUP_ID_HIT];
		pgd->kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
		pgd->hitgroup.moduleCH = orp.modules[funcPropMap[__CLOSESTHIT__RADIANCE].module];
		pgd->hitgroup.entryFunctionNameCH = funcPropMap[__CLOSESTHIT__RADIANCE].name;
		pgd->hitgroup.moduleAH = orp.modules[funcPropMap[__ANYHIT__RADIANCE].module];
		pgd->hitgroup.entryFunctionNameAH = funcPropMap[__ANYHIT__RADIANCE].name;


		//Hit Program
		pgd = &programGroupDescriptions[PROGRAMGROUP_ID_CAMERA];
		pgd->kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
		pgd->callables.moduleDC = orp.modules[funcPropMap[__CALLABLE__PINHOLECAMERA].module];
		pgd->callables.entryFunctionNameDC = funcPropMap[__CALLABLE__PINHOLECAMERA].name;

		// Callables
		/*if (false)
			{
				pgd = &programGroupDescriptions[PROGRAMGROUP_ID_CAMERA];
				pgd->kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
				pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
				pgd->callables.moduleDC = modules[funcPropMap[__CALLABLE__PINHOLECAMERA].module];
				pgd->callables.entryFunctionNameDC = funcPropMap[__CALLABLE__PINHOLECAMERA].name;
			}*/

		char log[2048];
		size_t sizeof_log = sizeof(log);
		OPTIX_CHECK(optixProgramGroupCreate(
			state.optixContext,
			programGroupDescriptions.data(),
			(unsigned int)programGroupDescriptions.size(),
			&pgOptions,
			log, &sizeof_log,
			orp.programGroups.data()
		));
		if (sizeof_log > 1) {
			VTX_WARN(log);
		}
	}

	void CreatePipeline()
	{
		VTX_INFO("Creating Pipeline");

		char log[2048];
		size_t sizeof_log = sizeof(log);
		OPTIX_CHECK(optixPipelineCreate(
			state.optixContext,
			&state.pipelineCompileOptions,
			&state.pipelineLinkOptions,
			orp.programGroups.data(),
			(int)orp.programGroups.size(),
			log, &sizeof_log,
			&orp.pipeline
		));
		if (sizeof_log > 1) {
			VTX_WARN(log);
		}
	}

	void SetStackSize()
	{
		VTX_INFO("Computing Stack Sizes");

		OptixStackSizes ssp = {}; // Whole Program Stack Size
		for (OptixProgramGroup pg : orp.programGroups) {
			OptixStackSizes ss;

			OPTIX_CHECK(optixProgramGroupGetStackSize(pg, &ss, orp.pipeline));

			ssp.cssRG = std::max(ssp.cssRG, ss.cssRG);
			ssp.cssMS = std::max(ssp.cssMS, ss.cssMS);
			ssp.cssCH = std::max(ssp.cssCH, ss.cssCH);
			ssp.cssAH = std::max(ssp.cssAH, ss.cssAH);
			ssp.cssIS = std::max(ssp.cssIS, ss.cssIS);
			ssp.cssCC = std::max(ssp.cssCC, ss.cssCC);
			ssp.dssDC = std::max(ssp.dssDC, ss.dssDC);
		}

		// Temporaries
		unsigned int cssCCTree = ssp.cssCC;
		unsigned int cssCHOrMSPlusCCTree = std::max(ssp.cssCH, ssp.cssMS) + cssCCTree;
		const unsigned int maxDCDepth = 2;

		// Arguments
		unsigned int directCallableStackSizeFromTraversal = ssp.dssDC * maxDCDepth; // FromTraversal: DC is invoked from IS or AH.    // Possible stack size optimizations.
		unsigned int directCallableStackSizeFromState = ssp.dssDC * maxDCDepth; // FromState:     DC is invoked from RG, MS, or CH. // Possible stack size optimizations.
		unsigned int continuationStackSize =
			ssp.cssRG +
			cssCCTree +
			cssCHOrMSPlusCCTree * (std::max(1u, state.pipelineLinkOptions.maxTraceDepth) - 1u) +
			std::min(1u, state.pipelineLinkOptions.maxTraceDepth) * std::max(cssCHOrMSPlusCCTree, ssp.cssAH + ssp.cssIS);
		unsigned int maxTraversableGraphDepth = 2;

		OPTIX_CHECK(optixPipelineSetStackSize(
			orp.pipeline,
			directCallableStackSizeFromTraversal,
			directCallableStackSizeFromState,
			continuationStackSize,
			maxTraversableGraphDepth));

	}

	void CreateSBT()
	{
		VTX_INFO("Filling Shader Table");

		const int numHeaders = static_cast<int>(orp.programGroups.size());

		std::vector<SbtRecordHeader> sbtRecordHeaders(numHeaders);

		for (int i = 0; i < numHeaders; ++i)
		{
			OPTIX_CHECK(optixSbtRecordPackHeader(orp.programGroups[PROGRAMGROUP_ID_RAYGEN + i], &sbtRecordHeaders[i]));
		}

		CUDABuffer sbtRecordHeadersBuffer;
		sbtRecordHeadersBuffer.alloc_and_upload(sbtRecordHeaders);

		orp.sbt = {};
		orp.sbt.raygenRecord = sbtRecordHeadersBuffer.d_pointer();
		orp.sbt.exceptionRecord = sbtRecordHeadersBuffer.d_pointer() + sizeof(SbtRecordHeader) * PROGRAMGROUP_ID_ECXEPTION;

		orp.sbt.missRecordBase = sbtRecordHeadersBuffer.d_pointer() + sizeof(SbtRecordHeader) * PROGRAMGROUP_ID_MISS;
		orp.sbt.missRecordStrideInBytes = (unsigned int)sizeof(SbtRecordHeader);
		orp.sbt.missRecordCount = NUM_MISS_PROGRAM;

		orp.sbt.hitgroupRecordBase = sbtRecordHeadersBuffer.d_pointer() + sizeof(SbtRecordHeader) * PROGRAMGROUP_ID_HIT;
		orp.sbt.hitgroupRecordStrideInBytes = (unsigned int)sizeof(SbtRecordHeader);
		orp.sbt.hitgroupRecordCount = NUM_HIT_PROGRAM;

		orp.sbt.callablesRecordBase = sbtRecordHeadersBuffer.d_pointer() + sizeof(SbtRecordHeader) * PROGRAMGROUP_ID_CAMERA;
		orp.sbt.callablesRecordStrideInBytes = (unsigned int)sizeof(SbtRecordHeader);
		orp.sbt.callablesRecordCount = NUM_CALLABLE_PROGRAM;

	}

}

