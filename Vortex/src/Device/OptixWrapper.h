#pragma once
#include "cuda.h"
#include "cuda_runtime.h"
#include "UploadCode/CUDABuffer.h"
#include "UploadCode/CUDAMap.h"
#include "Core/Math.h"
#include <map>

#include "Denoiser.h"
#include "Core/VortexID.h"

namespace vtx::optix
{
	/*SBT Record Template*/
	struct SbtRecordHeader
	{
		__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	};

	struct State {
		// Context and Streams
		CUcontext						cudaContext{};
		CUstream						stream{};
		cudaDeviceProp					deviceProps{};
		OptixDeviceContext				optixContext{};

		//Options
		OptixModuleCompileOptions		moduleCompileOptions{};
		OptixPipelineCompileOptions		pipelineCompileOptions{};
		OptixPipelineLinkOptions		pipelineLinkOptions{};

		bool                 isValid = false;
		OptixDenoiserWrapper denoiser{};
		std::vector<CUstream>             materialStreams;
	};

	// index 1 is global, index 2 is local
	using sbtPosition = math::vec2i;

	enum OptixFunctionType
	{
		F_Raygen,
		F_Exception,
		F_Miss,
		F_ClosestHit,
		F_AnyHit,
		F_DirectCallable,
		F_ContinuationCallable
	};

	enum OptixProgramType
	{
		P_Raygen,
		P_Exception,
		P_Miss,
		P_Hit,
		P_DirectCallable,
		P_ContinuationCallable,

		P_NumberOfProgramType
	};

	struct ModuleOptix
	{
		OptixModule module;
		std::string name;
		std::string code;
		std::string path;

		const std::string& getCode();
		void createModule();
		OptixModule& getModule();
	};

	struct FunctionOptix
	{
		std::string name;
		std::shared_ptr<ModuleOptix> module;
		OptixFunctionType type;
	};

	struct ProgramOptix
	{
		OptixProgramType				type;
		std::string						name;
		OptixProgramGroupDesc			pgd{};
		OptixProgramGroup				pg{};
		bool							pgdCreated = false;

		std::shared_ptr<FunctionOptix>	raygenFunction;
		std::shared_ptr<FunctionOptix>	exceptionFunction;
		std::shared_ptr<FunctionOptix>	missFunction;
		std::shared_ptr<FunctionOptix>	closestHitFunction;
		std::shared_ptr<FunctionOptix>	anyHitFunction;
		std::shared_ptr<FunctionOptix>	directCallableFunction;
		std::shared_ptr<FunctionOptix>	continuationCallableFunction;
		std::shared_ptr<FunctionOptix>	mdlCallableFunction;

		void createProgramGroupDesc();
		const OptixProgramGroupDesc& getProgramGroupDescription();

		void createProgramGroup();
		const OptixProgramGroup& getProgramGroup();
	};

	typedef std::map<OptixProgramType, std::vector<std::shared_ptr<ProgramOptix>>> ProgramMap;
	typedef std::map<std::string, ProgramMap> SbtMap;

	struct SbtPipeline
	{
		ProgramMap																programs;
		CudaMap<uint32_t, sbtPosition>											sbtMap;
		std::vector<OptixProgramGroup>											programGroups;
		bool																	isDirty = true;
		OptixShaderBindingTable													sbt = {};
		SbtMap																	sbtPrograms;
		CUDABuffer																sbtRecordHeadersBuffer;

		void                                  initSbtMap();
		std::vector<OptixProgramGroup>        getProgramGroups();
		void                                  createSbt();
		void                                  registerProgram(const std::shared_ptr<ProgramOptix>& program);
		const OptixShaderBindingTable&        getSbt();
		const CudaMap<uint32_t, sbtPosition>& getSbtMap();
		int                                   getProgramSbt(std::string programName);
		//static int getProgramSbt(const CudaMap<uint32_t, sbtPosition>& map, std::string programName);
	};

	struct PipelineOptix
	{
		void registerProgram(const std::shared_ptr<ProgramOptix>& program, std::vector<std::string> sbtNames = {""});

		std::vector<OptixProgramGroup> getAllProgramGroups();

		void computeStackSize();

		void createPipeline();

		const OptixPipeline& getPipeline();

		int getProgramSbt(const std::string& programName, std::string sbtName = "");

		const OptixShaderBindingTable& getSbt(std::string sbtName = "");

		static void launchOptixKernel(const math::vec2i& launchDimension, const std::string& pipelineName);

	private:
		OptixPipeline															pipeline = nullptr;
		bool																	isDirty = true;
		std::vector<OptixProgramGroup>											allProgramGroups;
		std::map<std::string, SbtPipeline>										sbtPipelines;
	};

	State* getState();

	PipelineOptix* getRenderingPipeline();

	/* Initialize Optix, contexts and stream and relative options*/
	void init();

	void shutDown();

	/* Check For Capable Devices and Initialize Optix*/
	void startOptix();

	/*! creates and configures a optix device context */
	void createContext();

	/*Global Module Compiler Options*/
	void setModuleCompilersOptions();

	/*Global Pipeline Compiler Options*/
	void setPipelineCompilersOptions();

	/*Global Pipeline Link Options*/
	void setPipelineLinkOptions();

	void createRenderingPipeline();

	std::vector<cudaStream_t>& createMaterialStreamVector(const int size);

	/*Utility to directly create a direct callable program from a function name and module*/
	std::shared_ptr<ProgramOptix> createDcProgram(const std::shared_ptr<ModuleOptix>& module, const std::string& functionName, vtxID id = 0, const std::vector<std::string>& sbtName = {""});

	/*Utility to create BLAS*/
	OptixTraversableHandle createGeometryAcceleration(CUdeviceptr vertexData, uint32_t verticesNumber, uint32_t verticesStride, 
													  CUdeviceptr indexData, uint32_t indexNumber, uint32_t indicesStride);

	OptixInstance createInstance(uint32_t instanceId, const math::affine3f& transform, OptixTraversableHandle traversable);

	OptixTraversableHandle createInstanceAcceleration(const std::vector<OptixInstance>& optixInstances);


}
