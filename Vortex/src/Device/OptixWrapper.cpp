#include <algorithm>
#include "OptixWrapper.h"

#include <set>

#include "Core/Log.h"
#include "CUDAChecks.h"
#include "Core/Options.h"
#include "Core/Utils.h"
#include "UploadCode/CUDAMap.h"
#include "UploadCode/UploadBuffers.h"
#include "UploadCode/DeviceDataCoordinator.h"
#include <optix_function_table_definition.h>

namespace vtx::optix
{
	static State state;

	static PipelineOptix pipeline;

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

	PipelineOptix* getRenderingPipeline()
	{
		return &pipeline;
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

	void shutDown()
	{
		VTX_INFO("ShuttingDown: Optix");
		state.denoiser.shutDown();
		CU_CHECK_CONTINUE(cuCtxSetCurrent(state.cudaContext)); // Activate this CUDA context. Not using activate() because this needs a no-throw check.
		CU_CHECK_CONTINUE(cuCtxSynchronize());

		OPTIX_CHECK_CONTINUE(optixPipelineDestroy(optix::getRenderingPipeline()->getPipeline()));
		OPTIX_CHECK_CONTINUE(optixDeviceContextDestroy(state.optixContext));

		CU_CHECK_CONTINUE(cuStreamDestroy(state.stream));
		//CU_CHECK_CONTINUE(cuCtxDestroy(state.cudaContext));
	}

	void startOptix()
	{
		VTX_INFO("Optix Wrapper: Initializing optix");
		cudaFree(0);
		int numDevices;
		cudaGetDeviceCount(&numDevices);
		VTX_ASSERT_CLOSE(numDevices != 0, "Optix Wrapper: no CUDA capable devices found!");
		VTX_INFO("Optix Wrapper: found {} CUDA devices", numDevices);

		OPTIX_CHECK(optixInit());
		VTX_INFO("Optix Wrapper : Optix Initialized");
	}

	void createContext()
	{
		VTX_INFO("Optix Wrapper : Creating Optix Context");
		const int deviceId = getOptions()->deviceID;
		CUDA_CHECK(cudaSetDevice(deviceId));
		CUDA_CHECK(cudaStreamCreate(&state.stream));

		VTX_INFO("DeviceProps pointer : {}", (void*)& state.deviceProps);
		CUDA_CHECK(cudaGetDeviceProperties(&state.deviceProps, deviceId));
		VTX_INFO("Optix Wrapper : running on device: {}", state.deviceProps.name);

		CUresult  cuRes = cuCtxGetCurrent(&state.cudaContext);
		VTX_ASSERT_CLOSE(cuRes == CUDA_SUCCESS, "Optix Wrapper : Error querying current context: error code {}", (int)cuRes);

		OptixDeviceContextOptions options = {};
		options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
		options.logCallbackFunction = &context_log_cb;
		options.logCallbackLevel = 4;
		OPTIX_CHECK(optixDeviceContextCreate(state.cudaContext, &options, &state.optixContext));
		OPTIX_CHECK(optixDeviceContextSetLogCallback(state.optixContext, context_log_cb, nullptr, 4));
		if((getOptions()->isDebug && !(getOptions()->enableCache)) || !getOptions()->enableCache)
		{
			OPTIX_CHECK(optixDeviceContextSetCacheEnabled(state.optixContext, 0));
		}
	}

	std::vector<cudaStream_t>& createMaterialStreamVector(const int size)
	{
		state.materialStreams = std::vector<CUstream>();
		state.materialStreams.resize(size);
		//state.materialStreams[0] = state.stream;
		for (int i = 0; i < size; i++)
		{
			CUDA_CHECK(cudaStreamCreate(&state.materialStreams[i]));
		}
		return state.materialStreams;
	}

	void setModuleCompilersOptions()
	{
		VTX_INFO("Optix Wrapper : Setting Module Compiler Options");
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
		VTX_INFO("Optix Wrapper : Setting Pipeline Compiler Options");
		state.pipelineCompileOptions = {};

		state.pipelineCompileOptions.usesMotionBlur = false;
		state.pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
		state.pipelineCompileOptions.numPayloadValues = 2;  // I need two to encode a 64-bit pointer to the per ray payload structure.
		state.pipelineCompileOptions.numAttributeValues = 2;  // The minimum is two for the triangle barycentrics.
		if (getOptions()->isDebug) {
			state.pipelineCompileOptions.exceptionFlags =
				OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW |
				OPTIX_EXCEPTION_FLAG_TRACE_DEPTH |
				OPTIX_EXCEPTION_FLAG_USER;
		}
		else
			state.pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
		state.pipelineCompileOptions.pipelineLaunchParamsVariableName = getOptions()->LaunchParamName.c_str();
		if (getOptions()->OptixVersion != 70000)
			state.pipelineCompileOptions.usesPrimitiveTypeFlags =
				OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE |
				OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_CUBIC_BSPLINE;
	}

	void setPipelineLinkOptions()
	{
		// OptixPipelineLinkOptions
		VTX_INFO("Optix Wrapper : Setting Pipeline Linker Options");
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

	std::string getStrippedProgramName(std::string fullFunctionName)
	{
		const std::string prefix = "__direct_callable__";
		VTX_ASSERT_CLOSE((fullFunctionName.substr(0, prefix.size()) == prefix), "The name of the mdl function {} is not matching what expected", fullFunctionName);
		return fullFunctionName.substr(prefix.size());
	}

	std::shared_ptr<ProgramOptix> createDcProgram(const std::shared_ptr<ModuleOptix>& module, const std::string& functionName, const vtxID id, const std::vector<std::string>& sbtName)
	{
		const auto function = std::make_shared<optix::FunctionOptix>();
		function->name = functionName;
		function->module = module;
		function->type = optix::OptixFunctionType::F_DirectCallable;

		auto program = std::make_shared<optix::ProgramOptix>();
		program->type = optix::OptixProgramType::P_DirectCallable;
		if(id ==0)
		{
			program->name = getStrippedProgramName(functionName);
		}
		else
		{
			program->name = getStrippedProgramName(functionName) + "_" + std::to_string(id);
		}
		program->directCallableFunction = function;

		if(sbtName.size() == 1 && sbtName[0].empty())
		{
			optix::getRenderingPipeline()->registerProgram(program);
		}
		else
		{
			optix::getRenderingPipeline()->registerProgram(program, sbtName);

		}
		return program;
	}
	
	const std::string& ModuleOptix::getCode()
	{
		if (code.empty())
		{
			VTX_ASSERT_CLOSE(!path.empty(), "Requesting module {} code but neither code or path is present", name);
			const std::string& modulePath = utl::absolutePath(path);
			std::vector<char> programData = utl::readData(modulePath);
			code = std::string(programData.begin(), programData.end());
		}

		return code;
	}

	void ModuleOptix::createModule()
	{
		VTX_INFO("Optix Wrapper: Creatig Optix Module {}", name);
		const OptixDeviceContext& context = optix::getState()->optixContext;
		const OptixModuleCompileOptions& moduleCompileOptions = optix::getState()->moduleCompileOptions;
		const OptixPipelineCompileOptions& pipelineCompileOptions = optix::getState()->pipelineCompileOptions;

		char log[2048];
		size_t logSize = sizeof(log);
		const OptixResult result = optixModuleCreate(context,
													 &moduleCompileOptions,
													 &pipelineCompileOptions,
													 getCode().data(),
													 getCode().size(),
													 log,
													 &logSize,
													 &module);
		//VTX_ASSERT_CONTINUE(logSize <= 1, log);
		OPTIX_CHECK(result);
	}

	OptixModule& ModuleOptix::getModule()
	{
		if (module == nullptr)
			createModule();
		return module;
	}

	void ProgramOptix::createProgramGroupDesc()
	{
		pgd.flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
		switch (type)
		{
			case P_Raygen:
			{
				VTX_ASSERT_CLOSE(raygenFunction, "Optix Wrapper: Requesting pgd for raygen program {} but raygen function is missing!", name);
				VTX_INFO("Optix Wrapper: Creating raygen program group desc for program {} with function {}", name, raygenFunction->name.data());
				pgd.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
				pgd.raygen.module = raygenFunction->module->getModule();
				pgd.raygen.entryFunctionName = raygenFunction->name.data();
			}break;
			case P_Exception:
			{
				VTX_ASSERT_CLOSE(exceptionFunction, "Optix Wrapper: Requesting pgd for program {} but exception function is missing!", name);
				VTX_INFO("Optix Wrapper: Creating exception program group desc for program {} with function {}", name, exceptionFunction->name.data());
				pgd.kind = OPTIX_PROGRAM_GROUP_KIND_EXCEPTION;
				pgd.exception.module = exceptionFunction->module->getModule();
				pgd.exception.entryFunctionName = exceptionFunction->name.data();
			} break;
			case P_Miss:
			{
				VTX_ASSERT_CLOSE(missFunction, "Optix Wrapper: Requesting pgd for program {} but miss function is missing!", name);
				VTX_INFO("Optix Wrapper: Creating Miss program group desc for program {} with function {}", name, missFunction->name.data());
				pgd.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
				pgd.miss.module = missFunction->module->getModule();
				pgd.miss.entryFunctionName = missFunction->name.data();
			} break;
			case P_Hit:
			{
				VTX_ASSERT_CLOSE(closestHitFunction, "Optix Wrapper: Requesting pgd for program {} but closest hit function is missing!", name);
				VTX_ASSERT_CLOSE(anyHitFunction, "Optix Wrapper: Requesting pgd for program {} but any hit function is missing!", name);
				VTX_INFO("Optix Wrapper: Creating Hit program group desc for program {} with function anyHit {}", name, anyHitFunction->name.data());
				VTX_INFO("Optix Wrapper: Creating Hit program group desc for program {} with function closestHit {}", name, closestHitFunction->name.data());
				pgd.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
				pgd.hitgroup.moduleCH = closestHitFunction->module->getModule();
				pgd.hitgroup.entryFunctionNameCH = closestHitFunction->name.data();
				pgd.hitgroup.moduleAH = anyHitFunction->module->getModule();
				pgd.hitgroup.entryFunctionNameAH = anyHitFunction->name.data();
			} break;
			case P_DirectCallable:
			{
				VTX_ASSERT_CLOSE(directCallableFunction, "Optix Wrapper: Requesting pgd for program {} but direct callable function is missing!", name);
				VTX_INFO("Optix Wrapper: Creating Direct Callable program group desc for program {} with function {}", name, directCallableFunction->name.data());
				pgd.kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
				pgd.callables.moduleDC = directCallableFunction->module->getModule();
				pgd.callables.entryFunctionNameDC = directCallableFunction->name.data();
			} break;
			case P_ContinuationCallable:
			{
				VTX_ASSERT_CLOSE(continuationCallableFunction, "Optix Wrapper: Requesting pgd for program {} but continuation callable function is missing!", name);
				VTX_INFO("Optix Wrapper: Creating Continuation Callable program group desc for program {} with function {}", name, continuationCallableFunction->name.data());
				pgd.kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
				pgd.callables.moduleCC = continuationCallableFunction->module->getModule();
				pgd.callables.entryFunctionNameCC = continuationCallableFunction->name.data();
			} break;
		}
	}

	void ProgramOptix::createProgramGroup()
	{
		VTX_INFO("Optix Wrapper: Creatig Optix Program Group {}", name);
		const OptixProgramGroupDesc& programGd = getProgramGroupDescription();
		const OptixDeviceContext& context = optix::getState()->optixContext;
		constexpr OptixProgramGroupOptions pgOptions = {};

		char log[2048];
		size_t logSize = sizeof(log);

		const OptixResult result = optixProgramGroupCreate(context,
														   &programGd,
														   1,
														   &pgOptions,
														   log,
														   &logSize,
														   &pg);
		//VTX_ASSERT_CONTINUE(logSize <= 1, log);
		OPTIX_CHECK(result);
	}

	const OptixProgramGroup& ProgramOptix::getProgramGroup()
	{
		if (pg == nullptr)
			createProgramGroup();
		return pg;
	}

	const OptixProgramGroupDesc& ProgramOptix::getProgramGroupDescription()
	{
		if (!pgdCreated)
		{
			createProgramGroupDesc();
			pgdCreated = true;
		}
		return pgd;
	}

	void SbtPipeline::registerProgram(const std::shared_ptr<ProgramOptix>& program)
	{
		programs[program->type].push_back(program);
		isDirty = true;
	}

	void SbtPipeline::initSbtMap()
	{
		createSbt();
		isDirty = false;
	}

	std::vector<OptixProgramGroup> SbtPipeline::getProgramGroups()
	{
		// ATTENTION : the order in which the program groups is added will define the SBT index
		if (isDirty || programGroups.empty())
		{
			programGroups = {};
			sbtMap = {};
			int globalSbtPosition = 0;
			int localSbtPosition = 0;
			for (int i = 0; i < P_NumberOfProgramType; ++i)
			{
				auto programType = static_cast<OptixProgramType>(i);
				if (programType != P_ContinuationCallable)
				{
					localSbtPosition = 0;
				}
				for (const std::shared_ptr<ProgramOptix>& program : programs[programType])
				{
					programGroups.push_back(program->getProgramGroup());
					uint32_t hash = stringHash(program->name.data());
					sbtMap[hash] = sbtPosition{globalSbtPosition,localSbtPosition};
					++globalSbtPosition;
					++localSbtPosition;
				}
			}
		}

		return programGroups;
	}


	void SbtPipeline::createSbt()
	{
		VTX_INFO("Optix Wrapper: Filling Shader Table");


		const std::vector<OptixProgramGroup>& pgs = getProgramGroups();
		//const OptixPipeline& pipe = getPipeline();

		const int numHeaders = static_cast<int>(pgs.size());

		std::vector<SbtRecordHeader> sbtRecordHeaders(numHeaders);

		for (int i = 0; i < numHeaders; ++i)
		{
			const OptixResult result = optixSbtRecordPackHeader(pgs[i], &sbtRecordHeaders[i]);
			OPTIX_CHECK(result);
		}

		sbtRecordHeadersBuffer.upload(sbtRecordHeaders);

		sbt = {};
		int offset = 0;
		sbt.raygenRecord = sbtRecordHeadersBuffer.dPointer();
		offset = 1;

		if(!programs[OptixProgramType::P_Exception].empty())
		{
			sbt.exceptionRecord = sbtRecordHeadersBuffer.dPointer() + sizeof(SbtRecordHeader) * offset;
			offset += 1;
		}

		if(!programs[OptixProgramType::P_Miss].empty())
		{
			sbt.missRecordBase = sbtRecordHeadersBuffer.dPointer() + sizeof(SbtRecordHeader) * offset;
			sbt.missRecordStrideInBytes = static_cast<unsigned int>(sizeof(SbtRecordHeader));
			sbt.missRecordCount = programs[OptixProgramType::P_Miss].size();
			offset += sbt.missRecordCount;
		}

		if (!programs[OptixProgramType::P_Hit].empty())
		{
			sbt.hitgroupRecordBase = sbtRecordHeadersBuffer.dPointer() + sizeof(SbtRecordHeader) * offset;
			sbt.hitgroupRecordStrideInBytes = static_cast<unsigned int>(sizeof(SbtRecordHeader));
			sbt.hitgroupRecordCount = programs[OptixProgramType::P_Hit].size();
			offset += sbt.hitgroupRecordCount;
		}

		if (!programs[OptixProgramType::P_DirectCallable].empty() || !programs[OptixProgramType::P_ContinuationCallable].empty())
		{
			sbt.callablesRecordBase = sbtRecordHeadersBuffer.dPointer() + sizeof(SbtRecordHeader) * offset;
			sbt.callablesRecordStrideInBytes = static_cast<unsigned int>(sizeof(SbtRecordHeader));
			sbt.callablesRecordCount = programs[OptixProgramType::P_DirectCallable].size() + programs[OptixProgramType::P_ContinuationCallable].size();
		}
	}

	const OptixShaderBindingTable& SbtPipeline::getSbt()
	{
		if (isDirty)
		{
			initSbtMap();
		}

		return sbt;
	}

	const std::map<uint32_t, sbtPosition>& SbtPipeline::getSbtMap()
	{
		if (isDirty)
		{
			initSbtMap();
		}
		return sbtMap;
	}

	int SbtPipeline::getProgramSbt(std::string programName)
	{
		const std::map<uint32_t, sbtPosition>& map  = getSbtMap();
		const uint32_t                        hash = stringHash(programName.data());
		if (map.find(hash) != map.end())
		{
			const sbtPosition& pos = map.at(hash);
			return pos.y;
		}
		else
		{
			VTX_ERROR("Program {} not found in SBT", programName);
			return -1;
		}
	}

	//int SbtPipeline::getProgramSbt(const CudaMap<uint32_t, sbtPosition>& map, std::string programName)
	//{
	//	const uint32_t hash = stringHash(programName.data());
	//	if (map.contains(hash))
	//	{
	//		return map[hash].y;
	//	}
	//	else
	//	{
	//		VTX_ERROR("Program {} not found in SBT", programName);
	//		return -1;
	//	}
	//}

	void PipelineOptix::registerProgram(const std::shared_ptr<ProgramOptix>& program, std::vector<std::string> sbtNames)
	{
		if (sbtNames.size() == 1 && sbtNames[0] == "")
		{
			sbtNames[0] = "Default";
		}
		for (const auto& sbtName : sbtNames)
		{
			if (sbtPipelines.find(sbtName) == sbtPipelines.end())
			{
				sbtPipelines[sbtName] = SbtPipeline();
			}

			sbtPipelines[sbtName].registerProgram(program);
		}
		isDirty = true;
	}

	std::vector<OptixProgramGroup> PipelineOptix::getAllProgramGroups()
	{
		if (isDirty || allProgramGroups.empty())
		{
			for (auto& [sbtName, sbtPipe] : sbtPipelines)
			{
				std::vector<OptixProgramGroup>& sbtPrograms = sbtPipe.getProgramGroups();
				for (auto& pg : sbtPrograms)
				{
					//Check if pg is already in allProgramGroups
					if (std::find(allProgramGroups.begin(), allProgramGroups.end(), pg) == allProgramGroups.end())
					{
						allProgramGroups.push_back(pg);
					}
				}
			}
		}

		return allProgramGroups;
	}


	void PipelineOptix::computeStackSize()
	{
		VTX_INFO("Optix Wrapper: Computing Stack Sizes");

		const OptixPipelineLinkOptions& pipelineLinkOptions = optix::getState()->pipelineLinkOptions;
		const std::vector<OptixProgramGroup>& pgs = getAllProgramGroups();
		const OptixPipeline& pipe = getRenderingPipeline()->getPipeline();

		OptixStackSizes ssp = {}; // Whole Program Stack Size
		for (const OptixProgramGroup pg : pgs) {
			OptixStackSizes ss;

			OPTIX_CHECK(optixProgramGroupGetStackSize(pg, &ss, pipe));

			ssp.cssRG = std::max(ssp.cssRG, ss.cssRG);
			ssp.cssMS = std::max(ssp.cssMS, ss.cssMS);
			ssp.cssCH = std::max(ssp.cssCH, ss.cssCH);
			ssp.cssAH = std::max(ssp.cssAH, ss.cssAH);
			ssp.cssIS = std::max(ssp.cssIS, ss.cssIS);
			ssp.cssCC = std::max(ssp.cssCC, ss.cssCC);
			ssp.dssDC = std::max(ssp.dssDC, ss.dssDC);
		}

		// Temporaries
		const unsigned int cssCcTree = ssp.cssCC;
		const unsigned int cssChOrMsPlusCcTree = std::max(ssp.cssCH, ssp.cssMS) + cssCcTree;
		const unsigned int maxDcDepth = getOptions()->maxDcDepth;

		// Arguments
		const unsigned int directCallableStackSizeFromTraversal = ssp.dssDC * maxDcDepth; // FromTraversal: DC is invoked from IS or AH.    // Possible stack size optimizations.
		const unsigned int directCallableStackSizeFromState = ssp.dssDC * maxDcDepth; // FromState:     DC is invoked from RG, MS, or CH. // Possible stack size optimizations.
		const unsigned int continuationStackSize =
			ssp.cssRG +
			cssCcTree +
			cssChOrMsPlusCcTree * (std::max(1u, pipelineLinkOptions.maxTraceDepth) - 1u) +
			std::min(1u, pipelineLinkOptions.maxTraceDepth) * std::max(cssChOrMsPlusCcTree, ssp.cssAH + ssp.cssIS);
		const unsigned int maxTraversableGraphDepth = getOptions()->maxTraversableGraphDepth;

		const OptixResult result = optixPipelineSetStackSize(pipe,
			directCallableStackSizeFromTraversal,
			directCallableStackSizeFromState,
			continuationStackSize,
			maxTraversableGraphDepth);

		OPTIX_CHECK(result);
	}

	void PipelineOptix::createPipeline()
	{
		const std::vector<OptixProgramGroup>& pgs = getAllProgramGroups();

		VTX_INFO("Optix Wrapper: Creating Pipeline");
		const OptixDeviceContext& context = optix::getState()->optixContext;
		const OptixPipelineLinkOptions& pipelineLinkOptions = optix::getState()->pipelineLinkOptions;
		const OptixPipelineCompileOptions& pipelineCompileOptions = optix::getState()->pipelineCompileOptions;

		char log[2048];
		size_t logSize = sizeof(log);

		const OptixResult result = optixPipelineCreate(context,
													   &pipelineCompileOptions,
													   &pipelineLinkOptions,
													   pgs.data(),
													   static_cast<int>(pgs.size()),
													   log,
													   &logSize,
													   &pipeline);
		//VTX_ASSERT_CONTINUE(logSize <= 1, log);
		OPTIX_CHECK(result);

		// We do it here to avoid infinite loop where getPipeline() calls createPipeline() which calls getPipeline()...
		isDirty = false;
		computeStackSize();

		for(auto& [sbtName, sbtPipe] : sbtPipelines)
		{
			sbtPipe.initSbtMap();
		}
	}

	const OptixPipeline& PipelineOptix::getPipeline()
	{
		if (!pipeline || isDirty)
		{
			createPipeline();
		}
		return pipeline;
	}

	int PipelineOptix::getProgramSbt(const std::string& programName, std::string sbtName)
	{
		if (sbtName.empty())
		{
			sbtName = "Default";
		}

		return sbtPipelines[sbtName].getProgramSbt(programName);
	}

	const OptixShaderBindingTable& PipelineOptix::getSbt(std::string sbtName)
	{
		if (sbtName.empty())
		{
			sbtName = "Default";
		}

		return sbtPipelines[sbtName].getSbt();
	}

	void PipelineOptix::launchOptixKernel(const math::vec2i& launchDimension, const std::string& pipelineName)
	{
		const optix::State& state = *(optix::getState());
		const OptixPipeline& pipeline = optix::getRenderingPipeline()->getPipeline();
		const OptixShaderBindingTable& sbt = optix::getRenderingPipeline()->getSbt(pipelineName);

		const auto result = optixLaunch(/*! pipeline we're launching launch: */
			pipeline, state.stream,
			/*! parameters and SBT */
			onDeviceData->launchParamsData.getDevicePtr(),
			onDeviceData->launchParamsData.imageBuffer.bytesSize(),
			&sbt,
			/*! dimensions of the launch: */
			launchDimension.x,
			launchDimension.y,
			1
		);
		OPTIX_CHECK(result);
		CUDA_SYNC_CHECK();
	}

	OptixTraversableHandle createGeometryAcceleration(CUdeviceptr vertexData, uint32_t verticesNumber, uint32_t verticesStride, CUdeviceptr indexData, uint32_t indexNumber, uint32_t indicesStride)
	{
		VTX_INFO("Optix Wrapper: Computing BLAS");

		//CUDA_SYNC_CHECK();

		OptixDeviceContext& optixContext = getState()->optixContext;
		CUstream& stream = getState()->stream;

		/// BLAS Inputs ///
		OptixBuildInput buildInput = {};

		buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

		buildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
		buildInput.triangleArray.vertexStrideInBytes = verticesStride;
		buildInput.triangleArray.numVertices = verticesNumber;
		buildInput.triangleArray.vertexBuffers = &vertexData;

		buildInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
		buildInput.triangleArray.indexStrideInBytes = indicesStride;
		buildInput.triangleArray.numIndexTriplets = indexNumber / 3;
		buildInput.triangleArray.indexBuffer = indexData;

		unsigned int inputFlags[1] = { OPTIX_GEOMETRY_FLAG_NONE };

		buildInput.triangleArray.flags = inputFlags;
		buildInput.triangleArray.numSbtRecords = 1;

		/// BLAS Options ///
		OptixAccelBuildOptions accelOptions = {};
		accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
		accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

		/// Prepare Compaction ///
		OptixAccelBufferSizes blasBufferSizes;
		OPTIX_CHECK(optixAccelComputeMemoryUsage(optixContext,
												 &accelOptions,
												 &buildInput,
												 1,
												 &blasBufferSizes));

		CUDABuffer compactedSizeBuffer;
		compactedSizeBuffer.alloc(sizeof(uint64_t));

		OptixAccelEmitDesc emitDesc = {};
		emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
		emitDesc.result = compactedSizeBuffer.dPointer();

		/// First build ///

		CUDABuffer tempBuffer;
		tempBuffer.alloc(blasBufferSizes.tempSizeInBytes);

		CUDABuffer outputBuffer;
		outputBuffer.alloc(blasBufferSizes.outputSizeInBytes);

		OptixTraversableHandle traversable;

		OPTIX_CHECK(optixAccelBuild(optixContext,
									stream,
									&accelOptions,
									&buildInput,
									1,
									tempBuffer.dPointer(),
									tempBuffer.bytesSize(),
									outputBuffer.dPointer(),
									outputBuffer.bytesSize(),
									&traversable,
									&emitDesc, 1));
		//CUDA_SYNC_CHECK();

		/// Compaction ///
		uint64_t compactedSize;
		compactedSizeBuffer.download(&compactedSize);

		CUdeviceptr d_gas = outputBuffer.dPointer();
		if (compactedSize < outputBuffer.bytesSize()) {
			CUDABuffer outputBuffer_compacted;
			outputBuffer_compacted.alloc(compactedSize);
			OPTIX_CHECK(optixAccelCompact(optixContext,
										  /*stream:*/nullptr,
										  traversable,
										  outputBuffer_compacted.dPointer(),
										  outputBuffer_compacted.bytesSize(),
										  &traversable));
			d_gas = outputBuffer_compacted.dPointer();

			auto savedBytes = outputBuffer.bytesSize() - compactedSize;
			VTX_INFO("Optix Wrapper: Compacted GAS, saved {} bytes", savedBytes);
			//CUDA_SYNC_CHECK();
			outputBuffer.free(); // << the UNcompacted, temporary output buffer
		}

		/// Clean Up ///
		tempBuffer.free();
		compactedSizeBuffer.free();

		return traversable;
	}

	OptixInstance createInstance(const uint32_t instanceId, const math::affine3f& transform, const OptixTraversableHandle traversable)
	{
		// First check if there is a valid material assigned to this instance.

		

		OptixInstance optixInstance = {};

		float matrix[12];
		transform.toFloat(matrix);
		memcpy(optixInstance.transform, matrix, sizeof(float) * 12);

		//++m_SequentialInstanceID;
		//OptixInstance.instanceId = m_SequentialInstanceID; // User defined instance index, queried with optixGetInstanceId().
		optixInstance.instanceId = instanceId; // User defined instance index, queried with optixGetInstanceId().
		optixInstance.visibilityMask = 255;
		optixInstance.sbtOffset = 0;
		optixInstance.flags = OPTIX_INSTANCE_FLAG_NONE;
		optixInstance.traversableHandle = traversable;

		return optixInstance;
	}

	static CUDABuffer IASBuffer;

	OptixTraversableHandle createInstanceAcceleration(const std::vector<OptixInstance>& optixInstances, OptixTraversableHandle& topTraversable)
	{
		//CUDA_SYNC_CHECK();
		VTX_INFO("Optix Wrapper: Computing TLAS");

		auto& optixContext = optix::getState()->optixContext;
		auto& stream = optix::getState()->stream;

		// Construct the TLAS by attaching all flattened instances.
		const size_t instancesSizeInBytes = sizeof(OptixInstance) * optixInstances.size();

		CUDABuffer instancesBuffer;
		instancesBuffer.upload(optixInstances);

		OptixBuildInput instanceInput = {};

		instanceInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
		instanceInput.instanceArray.instances = instancesBuffer.dPointer();
		instanceInput.instanceArray.numInstances = static_cast<unsigned int>(optixInstances.size());


		OptixAccelBuildOptions accelBuildOptions = {};

		accelBuildOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE;
		accelBuildOptions.buildFlags |= OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;

		if (topTraversable == 0) {
			VTX_INFO("Optix Wrapper: Building TLAS");
			accelBuildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
		}
		else {
			VTX_INFO("Optix Wrapper: Updating TLAS");
			accelBuildOptions.operation = OPTIX_BUILD_OPERATION_UPDATE;
		}


		OptixAccelBufferSizes accelBufferSizes;
		OPTIX_CHECK(optixAccelComputeMemoryUsage(optixContext,
												 &accelBuildOptions,
												 &instanceInput,
												 1,
												 &accelBufferSizes));


		CUDABuffer tempBuffer;
		tempBuffer.alloc(accelBufferSizes.tempSizeInBytes);

		IASBuffer.resize(accelBufferSizes.outputSizeInBytes);

		CUDABuffer aabbBuffer;
		aabbBuffer.alloc(sizeof(OptixAabb));

		OptixAccelEmitDesc emitDesc = {};
		emitDesc.type = OPTIX_PROPERTY_TYPE_AABBS;
		emitDesc.result = aabbBuffer.dPointer();

		OPTIX_CHECK(optixAccelBuild(optixContext,
									stream,
									&accelBuildOptions,
									&instanceInput,
									1,
									tempBuffer.dPointer(),
									tempBuffer.bytesSize(),
									IASBuffer.dPointer(),
									IASBuffer.bytesSize(),
									&topTraversable,
									&emitDesc, 1));

		OptixAabb aabb;
		aabbBuffer.download<OptixAabb>(&aabb);

		CUDA_SYNC_CHECK();

		/// Clean Up ///
		tempBuffer.free();
		instancesBuffer.free();

		return topTraversable;
	}
}

