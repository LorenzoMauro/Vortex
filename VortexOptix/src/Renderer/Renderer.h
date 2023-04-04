#pragma once
#include "glFrameBuffer.h"
#include <cstdint>
#include "CUDABuffer.h"
#include "LaunchParams.h"
#include "DeviceData.h"
#include "Scene/SceneGraph.h"

namespace vtx {


	/*SBT Record Template*/
	struct SbtRecordHeader
	{
		__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	};

	class Renderer : public scene::Node {
	public:

		Renderer();

		void Resize(uint32_t width, uint32_t height);

		GLuint GetFrame();

		void ElaborateScene();

		void setCamera(std::shared_ptr<scene::Camera> _camera) {
			camera = _camera;
		}

		void setScene(std::shared_ptr<scene::Group> _scene) {
			sceneRoot = _scene;
		}

	private:
		/* Check For Capable Devices and Initialize Optix*/
		void InitOptix();

		/*! creates and configures a optix device context */
		void createContext();

		/*Global Module Compiler Options*/
		void setModuleCompilersOptions();

		/*Global Pipeline Compiler Options*/
		void setPipelineCompilersOptions();

		/*Global Pipeline Link Options*/
		void setPipelineLinkOptions();

		/*Create Optix Modules*/
		void CreateModules();

		/*Create Programs*/
		void CreatePrograms();

		/*Create Pipeline*/
		void CreatePipeline();

		/*Set Stack Sizes*/
		void SetStackSize();


		/* Helper Function to Create a Record for the SBT */
		template<typename R, typename D>
		void FillAndUploadRecord(OptixProgramGroup& pg, std::vector<R>& records, D Data)
		{
			R rec;
			rec.Data = Data;
			OPTIX_CHECK(optixSbtRecordPackHeader(pg, &rec));
			records.push_back(rec);
		}

		/*Create SBT*/
		void CreateSBT();

		void Render();


	public:
		// Context and Streams
		CUcontext						cudaContext;
		CUstream						stream;
		cudaDeviceProp					deviceProps;
		OptixDeviceContext				optixContext;

		//Options
		OptixModuleCompileOptions		moduleCompileOptions;
		OptixPipelineCompileOptions		pipelineCompileOptions;
		OptixPipelineLinkOptions		pipelineLinkOptions;

		// Vector Containing all modules specified in the ModuleIdentifier enum
		std::vector<std::string>		modulesPath;
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

		//GL Interop
		glFrameBuffer					glFrameBuffer;
		CUgraphicsResource				m_cudaGraphicsResource = nullptr;
		CUarray							dstArray;

		//Scene Representation on Device
		DeviceScene						deviceData;

		CUDABuffer						launchParamsBuffer;
		CUDABuffer						cudaColorBuffer;
		LaunchParams					launchParams;

		std::shared_ptr<scene::Camera>	camera;
		std::shared_ptr<scene::Group>	sceneRoot;
	};
}