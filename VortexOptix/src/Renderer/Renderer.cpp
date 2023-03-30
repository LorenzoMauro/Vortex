#include "Renderer.h"

#include <optix_function_table_definition.h>
#include "CUDAChecks.h"
#include "Core/Options.h"
#include "Core/Utils.h"
#include <algorithm>
#include <cudaGL.h>

namespace vtx {

	///*! SBT record for a raygen program */
	//struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord
	//{
	//	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	//	// just a dummy value - later examples will use more interesting
	//	// data here
	//	void* data;
	//};
	//
	///*! SBT record for a miss program */
	//struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) MissRecord
	//{
	//	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	//	// just a dummy value - later examples will use more interesting
	//	// data here
	//	void* data;
	//};
	//
	///*! SBT record for a hitgroup program */
	//struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) HitgroupRecord
	//{
	//	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	//	// just a dummy value - later examples will use more interesting
	//	// data here
	//	int objectID;
	//};

	Renderer::Renderer() {
		InitOptix();
		createContext();
		setModuleCompilersOptions();
		setPipelineCompilersOptions();
		setPipelineLinkOptions();
		CreateModules();
		CreatePrograms();
		CreatePipeline();
		SetStackSize();
		CreateSBD();
		launchParamsBuffer.alloc(sizeof(launchParams));
	}

	/* Check For Capable Devices and Initialize Optix*/
	void vtx::Renderer::InitOptix() {
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

	/*! creates and configures a optix device context */
	void Renderer::createContext()
	{
		VTX_INFO("Creating Optix Context");
		const int deviceID = 0;
		CUDA_CHECK(cudaSetDevice(deviceID));
		CUDA_CHECK(cudaStreamCreate(&stream));

		cudaGetDeviceProperties(&deviceProps, deviceID);
		VTX_INFO("running on device: {}", deviceProps.name);

		CUresult  cuRes = cuCtxGetCurrent(&cudaContext);
		if (cuRes != CUDA_SUCCESS)
			fprintf(stderr, "Error querying current context: error code %d\n", cuRes);

		OPTIX_CHECK(optixDeviceContextCreate(cudaContext, 0, &optixContext));
		OPTIX_CHECK(optixDeviceContextSetLogCallback(optixContext, context_log_cb, nullptr, 4));
	}

	void Renderer::setModuleCompilersOptions()
	{
		VTX_INFO("Setting Module Compiler Options");
		// OptixModuleCompileOptions
		moduleCompileOptions = {};

		moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
		if (options.isDebug) {
			moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0; // No optimizations.
			moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;     // Full debug. Never profile kernels with this setting!
		}
		else {
			moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3; // All optimizations, is the default.
			moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
			//moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO; // To activate if OPTIX VERSION is < 70400
		}
	}

	void Renderer::setPipelineCompilersOptions(){
		// OptixPipelineCompileOptions
		VTX_INFO("Setting Pipeline Compiler Options");
		pipelineCompileOptions = {};

		pipelineCompileOptions.usesMotionBlur = false;
		pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
		pipelineCompileOptions.numPayloadValues = 2;  // I need two to encode a 64-bit pointer to the per ray payload structure.
		pipelineCompileOptions.numAttributeValues = 2;  // The minimum is two for the triangle barycentrics.
		if (options.isDebug) {
			pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG;
			pipelineCompileOptions.exceptionFlags =
				OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW |
				OPTIX_EXCEPTION_FLAG_TRACE_DEPTH |
				OPTIX_EXCEPTION_FLAG_USER |
				OPTIX_EXCEPTION_FLAG_DEBUG;
		}
		else
			pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
		pipelineCompileOptions.pipelineLaunchParamsVariableName = options.LaunchParamName.c_str();
		if (options.OptixVersion != 70000)
			pipelineCompileOptions.usesPrimitiveTypeFlags = 
			OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE |
			OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_CUBIC_BSPLINE;
	}

	void Renderer::setPipelineLinkOptions()
	{
		// OptixPipelineLinkOptions
		VTX_INFO("Setting Pipeline Linker Options");
		pipelineLinkOptions = {};

		pipelineLinkOptions.maxTraceDepth = 2;
		if (options.isDebug) {
			pipelineLinkOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL; // Full debug. Never profile kernels with this setting!
		}
		else {
			pipelineLinkOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
			//m_plo.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO; // Version < 7.4
		}
	}

	void Renderer::CreateModules()
	{
		VTX_INFO("Creating Modules");
		// Each source file results in one OptixModule.
		modules.resize(NUM_MODULE_IDENTIFIERS);
		modulesPath.resize(NUM_MODULE_IDENTIFIERS);
		modulesPath[MODULE_ID_DEVICEPROGRAM] = utl::absolutePath(options.modulePath + "devicePrograms.ptx");

		// Create all modules:
		for (size_t i = 0; i < modulesPath.size(); ++i)
		{
			// Since OptiX 7.5.0 the program input can either be *.ptx source code or *.optixir binary code.
			// The module filenames are automatically switched between *.ptx or *.optixir extension based on the definition of USE_OPTIX_IR
			std::vector<char> programData = utl::readData(modulesPath[i]);


			char log[2048];
			size_t sizeof_log = sizeof(log);
			OPTIX_CHECK(optixModuleCreateFromPTX(
				optixContext,
				&moduleCompileOptions,
				&pipelineCompileOptions,
				programData.data(), 
				programData.size(),
				log, &sizeof_log,
				&modules[i]));
			if (sizeof_log > 1){
				VTX_WARN(log);
			}

		}
	}

	void Renderer::CreatePrograms()
	{
		VTX_INFO("Creating Programs");
		OptixProgramGroupOptions pgOptions = {};

		std::vector<OptixProgramGroupDesc>	programGroupDescriptions(NUM_PROGRAMGROUP_IDENTIFIERS);
		programGroups.resize(NUM_PROGRAMGROUP_IDENTIFIERS);
		OptixProgramGroupDesc* pgd;

		// Raygen program:
		pgd = &programGroupDescriptions[PROGRAMGROUP_ID_RAYGEN];
		pgd->kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
		pgd->raygen.module = modules[MODULE_ID_DEVICEPROGRAM];
		pgd->raygen.entryFunctionName = "__raygen__renderFrame";

		// Miss program:
		pgd = &programGroupDescriptions[PROGRAMGROUP_ID_MISS];
		pgd->kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
		pgd->raygen.module = modules[MODULE_ID_DEVICEPROGRAM];
		pgd->raygen.entryFunctionName = "__miss__radiance";

		//Hit Program
		pgd = &programGroupDescriptions[PROGRAMGROUP_ID_HIT];
		pgd->kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
		pgd->hitgroup.moduleCH = modules[MODULE_ID_DEVICEPROGRAM];
		pgd->hitgroup.entryFunctionNameCH = "__closesthit__radiance";
		pgd->hitgroup.moduleAH = modules[MODULE_ID_DEVICEPROGRAM];
		pgd->hitgroup.entryFunctionNameAH = "__anyhit__radiance";

		char log[2048];
		size_t sizeof_log = sizeof(log);
		OPTIX_CHECK(optixProgramGroupCreate(
			optixContext,
			programGroupDescriptions.data(),
			(unsigned int) programGroupDescriptions.size(),
			&pgOptions,
			log, &sizeof_log,
			programGroups.data()
		));
		if (sizeof_log > 1) {
			VTX_WARN(log);
		}
	}

	void Renderer::CreatePipeline()
	{
		VTX_INFO("Creating Pipeline");

		char log[2048];
		size_t sizeof_log = sizeof(log);
		OPTIX_CHECK(optixPipelineCreate(
			optixContext,
			&pipelineCompileOptions,
			&pipelineLinkOptions,
			programGroups.data(),
			(int)programGroups.size(),
			log, &sizeof_log,
			&pipeline
		));
		if (sizeof_log > 1) {
			VTX_WARN(log);
		}
	}

	void Renderer::SetStackSize()
	{
		VTX_INFO("Computing Stack Sizes");

		OptixStackSizes ssp = {}; // Whole Program Stack Size
		for (OptixProgramGroup pg : programGroups) {
			OptixStackSizes ss;

			OPTIX_CHECK(optixProgramGroupGetStackSize(pg, &ss));

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

		// FromTraversal: DC is invoked from IS or AH.      
		// Possible stack size optimizations.
		unsigned int directCallableStackSizeFromTraversal = ssp.dssDC * maxDCDepth; 
		// FromState:     DC is invoked from RG, MS, or CH. 
		// Possible stack size optimizations.
		unsigned int directCallableStackSizeFromState = ssp.dssDC * maxDCDepth; 
		unsigned int continuationStackSize = 
			ssp.cssRG + 
			cssCCTree + 
			cssCHOrMSPlusCCTree * (std::max(1u, pipelineLinkOptions.maxTraceDepth) - 1u) +
			std::min(1u, pipelineLinkOptions.maxTraceDepth) * std::max(cssCHOrMSPlusCCTree, ssp.cssAH + ssp.cssIS);
		unsigned int maxTraversableGraphDepth = 2;

		OPTIX_CHECK(optixPipelineSetStackSize(
			pipeline,
			directCallableStackSizeFromTraversal, 
			directCallableStackSizeFromState, 
			continuationStackSize,
			maxTraversableGraphDepth));

	}

	void Renderer::CreateSBD()
	{
		VTX_INFO("Filling Shader Table");

		std::vector<RaygenRecord> RaygenRecords;
		FillAndUploadRecord(
			programGroups[PROGRAMGROUP_ID_RAYGEN],
			RaygenRecords,
			nullptr);
		RaygenRecordBuffer.alloc_and_upload(RaygenRecords);

		std::vector<MissRecord> MissRecords;
		FillAndUploadRecord(
			programGroups[PROGRAMGROUP_ID_MISS],
			MissRecords,
			nullptr);
		MissRecordBuffer.alloc_and_upload(RaygenRecords);

		int numObjects = 1;
		std::vector<HitgroupRecord> HitgroupRecords;
		OptixProgramGroup hitgroupPG = programGroups[PROGRAMGROUP_ID_HIT];
		for (int i = 0; i < numObjects; i++) {
			FillAndUploadRecord(
				programGroups[PROGRAMGROUP_ID_HIT],
				HitgroupRecords,
				i);
		}
		HitRecordBuffer.alloc_and_upload(HitgroupRecords);

		sbt = {};
		sbt.raygenRecord = RaygenRecordBuffer.d_pointer();
		sbt.missRecordBase = MissRecordBuffer.d_pointer();
		sbt.missRecordStrideInBytes = sizeof(MissRecord);
		sbt.missRecordCount = (int)MissRecords.size();
		sbt.hitgroupRecordBase = HitRecordBuffer.d_pointer();
		sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
		sbt.hitgroupRecordCount = (int)HitgroupRecords.size();

	}

	void Renderer::Render()
	{
		if (launchParams.fbSize.x == 0) return;

		launchParamsBuffer.upload(&launchParams, 1);
		launchParams.frameID++;

		OPTIX_CHECK(optixLaunch(/*! pipeline we're launching launch: */
			pipeline, stream,
			/*! parameters and SBT */
			launchParamsBuffer.d_pointer(),
			launchParamsBuffer.sizeInBytes,
			&sbt,
			/*! dimensions of the launch: */
			launchParams.fbSize.x,
			launchParams.fbSize.y,
			1
		));

		/// Update Display texture
		// Map the Texture object directly and copy the output buffer. 
		CU_CHECK(cuGraphicsMapResources(1, &m_cudaGraphicsResource, stream)); // This is an implicit cuSynchronizeStream().

		CU_CHECK(cuGraphicsSubResourceGetMappedArray(&dstArray, m_cudaGraphicsResource, 0, 0)); // arrayIndex = 0, mipLevel = 0

		CUDA_MEMCPY3D params = {};

		params.srcMemoryType = CU_MEMORYTYPE_DEVICE;
		params.srcDevice = launchParams.colorBuffer;
		params.srcPitch = launchParams.fbSize.x * sizeof(uint32_t);
		params.srcHeight = launchParams.fbSize.y;

		params.dstMemoryType = CU_MEMORYTYPE_ARRAY;
		params.dstArray = dstArray;
		params.WidthInBytes = launchParams.fbSize.x * sizeof(uint32_t);
		params.Height = launchParams.fbSize.y;
		params.Depth = 1;

		CU_CHECK(cuMemcpy3D(&params)); // Copy from linear to array layout.

		CU_CHECK(cuGraphicsUnmapResources(1, &m_cudaGraphicsResource, stream)); // This is an implicit cuSynchronizeStream().

	}

	void Renderer::Resize(uint32_t width, uint32_t height) {
		if (width == 0 | height == 0) {
			launchParams.fbSize.x = launchParams.fbSize.y = 0;
			return;
		}
		if (m_cudaGraphicsResource != nullptr) {
			CU_CHECK(cuGraphicsUnregisterResource(m_cudaGraphicsResource));
		}
		cudaColorBuffer.resize(width * height * sizeof(uint32_t));
		launchParams.fbSize.x = width;
		launchParams.fbSize.y = height;
		launchParams.colorBuffer = cudaColorBuffer.d_pointer();

		glFrameBuffer.SetSize(width, height);

		CU_CHECK(cuGraphicsGLRegisterImage(&m_cudaGraphicsResource, glFrameBuffer.m_ColorAttachment, GL_TEXTURE_2D, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD));

	}

	GLuint Renderer::GetFrame() {
		glFrameBuffer.Bind();
		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);
		glFrameBuffer.Unbind();
		Render();
		return glFrameBuffer.m_ColorAttachment;
	}
}

