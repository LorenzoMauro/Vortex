#pragma once
#include "glFrameBuffer.h"
#include <cstdint>
#include "CUDABuffer.h"
#include "LaunchParams.h"
#include "DeviceData.h"

namespace vtx {


	/*SBT Record Template*/
	template<typename T>
	struct sbtRecord {
		__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
		T Data;
	};

	typedef sbtRecord<void*> RaygenRecord;
	typedef sbtRecord<void*> MissRecord;
	typedef sbtRecord<int> HitgroupRecord;

	enum ModuleIdentifier {
		MODULE_ID_DEVICEPROGRAM,

		NUM_MODULE_IDENTIFIERS
	};

	enum ProgramGroupIdentifier {
		PROGRAMGROUP_ID_RAYGEN,
		PROGRAMGROUP_ID_MISS,
		PROGRAMGROUP_ID_HIT,



		NUM_PROGRAMGROUP_IDENTIFIERS
	};

	class Renderer {
	public:

		Renderer();

		void Resize(uint32_t width, uint32_t height);

		GLuint GetFrame();

		void ElaborateScene(std::shared_ptr<scene::Group> Root);

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

		OptixPipeline					pipeline;

		//Shader Table Data;
		OptixShaderBindingTable			sbt;

		std::vector<RaygenRecord>		RaygenRecords;
		std::vector<MissRecord>			MissRecords;
		std::vector<HitgroupRecord>		HitgroupRecords;

		CUDABuffer						RaygenRecordBuffer;
		CUDABuffer						MissRecordBuffer;
		CUDABuffer						HitRecordBuffer;

		//GL Interop
		glFrameBuffer					glFrameBuffer;
		CUgraphicsResource				m_cudaGraphicsResource = nullptr;
		CUarray							dstArray;

		//Scene Representation on Device
		DeviceScene						deviceData;

		CUDABuffer						launchParamsBuffer;
		CUDABuffer						cudaColorBuffer;
		LaunchParams					launchParams;



	};
}