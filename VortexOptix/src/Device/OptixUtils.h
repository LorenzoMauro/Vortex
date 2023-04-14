#pragma once
#include "cuda.h"
#include "optix.h"
#include "cuda_runtime.h"
#include "CUDABuffer.h"

namespace vtx::optix
{

	struct State {
		// Context and Streams
		CUcontext						cudaContext;
		CUstream						stream;
		cudaDeviceProp					deviceProps;
		OptixDeviceContext				optixContext;

		//Options
		OptixModuleCompileOptions		moduleCompileOptions;
		OptixPipelineCompileOptions		pipelineCompileOptions;
		OptixPipelineLinkOptions		pipelineLinkOptions;

		bool							isValid = false;
	};

	struct OptixRenderingPipelineData {
		//std::vector<std::string>		modulesPath;
		std::vector<OptixModule>		modules;
		std::vector<OptixProgramGroup>	programGroups;
		std::vector<OptixProgramGroup>	raygenProgramGroups;
		std::vector<OptixProgramGroup>	missProgramGroups;
		std::vector<OptixProgramGroup>	hitProgramGroups;
		std::vector<OptixProgramGroup>	CallableProgramGroups;

		OptixPipeline					pipeline;

		//Shader Table Data;
		OptixShaderBindingTable			sbt;

		CUDABuffer						RaygenRecordBuffer;
		CUDABuffer						MissRecordBuffer;
		CUDABuffer						HitRecordBuffer;
		CUDABuffer						CallableRecordBuffer;

		bool							isValid = false;
	};

	State* getState();

	OptixRenderingPipelineData* getRenderingPipeline();

	/* Initialize Optix, contexts and stream and relative options*/
	void init();

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
	
	/*Create Optix Modules*/
	void CreateModules();

	/*Create Programs*/
	void CreatePrograms();

	/*Create Pipeline*/
	void CreatePipeline();

	/*Set Stack Sizes*/
	void SetStackSize();

	/*Create SBT*/
	void CreateSBT();
}
