#define NOMINMAX
#include <algorithm>
#include "OptixWrapper.h"
#include "Core/Log.h"
#include "CUDAChecks.h"
#include "Core/Options.h"
#include "Core/Utils.h"

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

		cudaGetDeviceProperties(&state.deviceProps, deviceId);
		VTX_INFO("Optix Wrapper : running on device: {}", state.deviceProps.name);

		CUresult  cuRes = cuCtxGetCurrent(&state.cudaContext);
		VTX_ASSERT_CLOSE(cuRes == CUDA_SUCCESS, "Optix Wrapper : Error querying current context: error code {}", cuRes);

		OPTIX_CHECK(optixDeviceContextCreate(state.cudaContext, nullptr, &state.optixContext));
		OPTIX_CHECK(optixDeviceContextSetLogCallback(state.optixContext, context_log_cb, nullptr, 4));
		if(getOptions()->isDebug && !(getOptions()->enableCache))
		{
			OPTIX_CHECK(optixDeviceContextSetCacheEnabled(state.optixContext, 0));
		}
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
			state.pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG;
			state.pipelineCompileOptions.exceptionFlags =
				OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW |
				OPTIX_EXCEPTION_FLAG_TRACE_DEPTH |
				OPTIX_EXCEPTION_FLAG_USER |
				OPTIX_EXCEPTION_FLAG_DEBUG;
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

	std::shared_ptr<ProgramOptix> createDcProgram(std::shared_ptr<ModuleOptix> module, std::string functionName)
	{
		const auto function = std::make_shared<optix::FunctionOptix>();
		function->name = functionName;
		function->module = module;
		function->type = optix::OptixFunctionType::F_DirectCallable;

		auto program = std::make_shared<optix::ProgramOptix>();
		program->type = optix::OptixProgramType::P_DirectCallable;
		program->name = getStrippedProgramName(functionName);
		program->directCallableFunction = function;

		optix::getRenderingPipeline()->registerProgram(program);
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
		VTX_ASSERT_CONTINUE(logSize <= 1, log);
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
		VTX_ASSERT_CONTINUE(logSize <= 1, log);
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

	void PipelineOptix::registerProgram(std::shared_ptr<ProgramOptix> program)
	{
		programs[program->type].push_back(program);
		isDirty = true;
	}

	std::vector<OptixProgramGroup> PipelineOptix::getProgramGroups()
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
					sbtMap.insert(hash, sbtPosition{ globalSbtPosition,localSbtPosition });
					++globalSbtPosition;
					++localSbtPosition;
				}
			}
		}

		return programGroups;
	}

	void PipelineOptix::createPipeline()
	{
		const std::vector<OptixProgramGroup>& pgs = getProgramGroups();

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
		VTX_ASSERT_CONTINUE(logSize <= 1, log);
		OPTIX_CHECK(result);

		computeStackSize();
		createSbt();
		isDirty = false;
	}

	const OptixPipeline& PipelineOptix::getPipeline()
	{
		if (!pipeline)
		{
			createPipeline();
		}
		return pipeline;
	}

	void PipelineOptix::computeStackSize()
	{
		VTX_INFO("Optix Wrapper: Computing Stack Sizes");

		const OptixPipelineLinkOptions& pipelineLinkOptions = optix::getState()->pipelineLinkOptions;
		const std::vector<OptixProgramGroup>& pgs = getProgramGroups();
		const OptixPipeline& pipe = getPipeline();

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

	void PipelineOptix::createSbt()
	{
		VTX_INFO("Optix Wrapper: Filling Shader Table");


		const std::vector<OptixProgramGroup>& pgs = getProgramGroups();
		const OptixPipeline& pipe = getPipeline();

		const int numHeaders = static_cast<int>(pgs.size());

		std::vector<SbtRecordHeader> sbtRecordHeaders(numHeaders);

		for (int i = 0; i < numHeaders; ++i)
		{
			const OptixResult result = optixSbtRecordPackHeader(pgs[i], &sbtRecordHeaders[i]);
			OPTIX_CHECK(result);
		}

		sbtRecordHeadersBuffer.upload(sbtRecordHeaders);

		sbt = {};
		sbt.raygenRecord = sbtRecordHeadersBuffer.dPointer();

		sbt.exceptionRecord = sbtRecordHeadersBuffer.dPointer() + sizeof(SbtRecordHeader) * OptixProgramType::P_Exception;

		sbt.missRecordBase = sbtRecordHeadersBuffer.dPointer() + sizeof(SbtRecordHeader) * OptixProgramType::P_Miss;
		sbt.missRecordStrideInBytes = static_cast<unsigned int>(sizeof(SbtRecordHeader));
		sbt.missRecordCount = programs[OptixProgramType::P_Miss].size();

		sbt.hitgroupRecordBase = sbtRecordHeadersBuffer.dPointer() + sizeof(SbtRecordHeader) * OptixProgramType::P_Hit;
		sbt.hitgroupRecordStrideInBytes = static_cast<unsigned int>(sizeof(SbtRecordHeader));
		sbt.hitgroupRecordCount = programs[OptixProgramType::P_Hit].size();

		sbt.callablesRecordBase = sbtRecordHeadersBuffer.dPointer() + sizeof(SbtRecordHeader) * OptixProgramType::P_DirectCallable;
		sbt.callablesRecordStrideInBytes = static_cast<unsigned int>(sizeof(SbtRecordHeader));
		sbt.callablesRecordCount = programs[OptixProgramType::P_DirectCallable].size() + programs[OptixProgramType::P_ContinuationCallable].size();

	}

	const OptixShaderBindingTable& PipelineOptix::getSbt()
	{
		if (isDirty)
		{
			createPipeline();
		}

		return sbt;
	}

	const CudaMap<uint32_t, sbtPosition>& PipelineOptix::getSbtMap()
	{
		if (isDirty)
		{
			createPipeline();
		}
		return sbtMap; 
	}

	int PipelineOptix::getProgramSbt(std::string programName)
	{
		const CudaMap<uint32_t, sbtPosition>& map = getSbtMap();
		const uint32_t hash = stringHash(programName.data());
		if (map.contains(hash))
		{
			return map[hash].y;
		}
		else
		{
			VTX_ERROR("Program {} not found in SBT", programName);
			return -1;
		}
	}

	int PipelineOptix::getProgramSbt(const CudaMap<uint32_t, sbtPosition>& map, std::string programName)
	{
		const uint32_t hash = stringHash(programName.data());
		if (map.contains(hash))
		{
			return map[hash].y;
		}
		else
		{
			VTX_ERROR("Program {} not found in SBT", programName);
			return -1;
		}
	}

	OptixTraversableHandle createGeometryAcceleration(CUdeviceptr vertexData, uint32_t verticesNumber, uint32_t verticesStride, CUdeviceptr indexData, uint32_t indexNumber, uint32_t indicesStride)
	{
		VTX_INFO("Optix Wrapper: Computing BLAS");

		CUDA_SYNC_CHECK();

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
		CUDA_SYNC_CHECK();

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
			CUDA_SYNC_CHECK();
			outputBuffer.free(); // << the UNcompacted, temporary output buffer
		}

		/// Clean Up ///
		tempBuffer.free();
		compactedSizeBuffer.free();

		return traversable;
	}

	OptixInstance createInstance(uint32_t instanceId, math::affine3f transform, OptixTraversableHandle traversable)
	{
		// First check if there is a valid material assigned to this instance.

		

		OptixInstance optixInstance = {};

		float* matrix = transform;
		memcpy(optixInstance.transform, transform, sizeof(float) * 12);

		//++m_SequentialInstanceID;
		//OptixInstance.instanceId = m_SequentialInstanceID; // User defined instance index, queried with optixGetInstanceId().
		optixInstance.instanceId = instanceId; // User defined instance index, queried with optixGetInstanceId().
		optixInstance.visibilityMask = 255;
		optixInstance.sbtOffset = 0;
		optixInstance.flags = OPTIX_INSTANCE_FLAG_NONE;
		optixInstance.traversableHandle = traversable;

		return optixInstance;
	}

	OptixTraversableHandle createInstanceAcceleration(const std::vector<OptixInstance>& optixInstances)
	{
		CUDA_SYNC_CHECK();
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

		accelBuildOptions.buildFlags = OPTIX_BUILD_FLAG_NONE;
		accelBuildOptions.buildFlags |= OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
		accelBuildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;


		OptixAccelBufferSizes accelBufferSizes;
		OPTIX_CHECK(optixAccelComputeMemoryUsage(optixContext,
												 &accelBuildOptions,
												 &instanceInput,
												 1,
												 &accelBufferSizes));


		CUDABuffer tempBuffer;
		tempBuffer.alloc(accelBufferSizes.tempSizeInBytes);

		CUDABuffer IAS;
		IAS.alloc(accelBufferSizes.outputSizeInBytes);

		OptixTraversableHandle TopTraversable;

		OPTIX_CHECK(optixAccelBuild(optixContext,
									stream,
									&accelBuildOptions,
									&instanceInput,
									1,
									tempBuffer.dPointer(),
									tempBuffer.bytesSize(),
									IAS.dPointer(),
									IAS.bytesSize(),
									&TopTraversable,
									nullptr, 0));

		CUDA_SYNC_CHECK();

		/// Clean Up ///
		tempBuffer.free();
		instancesBuffer.free();

		return TopTraversable;
	}
}

