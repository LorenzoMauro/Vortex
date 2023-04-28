#pragma once
#include <optix_types.h>
#include "CUDAMap.h"
#include "Core/VortexID.h"
#include "Device/DevicePrograms/LaunchParams.h"

#define UPLOAD_DATA \
		vtx::device::UploadData::getInstance()

namespace vtx::device
{
	struct UploadData {

		static UploadData* getInstance()
		{
			static UploadData uploadDataInstance;
			return &uploadDataInstance;
		}

		UploadData(const UploadData&) = delete;             // Disable copy constructor
		UploadData& operator=(const UploadData&) = delete;  // Disable assignment operator
		UploadData(UploadData&&) = delete;                  // Disable move constructor
		UploadData& operator=(UploadData&&) = delete;       // Disable move assignment operator
		
		void shutDown()
		{
			
		}

		//This first Set of Data is used to manage the cuda memory buffer memory, not create a new memory everytime something changes.
		// Each node will use its own buffer, and the buffer will be updated when the node is visited.

		// The following maps will be uploaded to the device
		// the launch params will contain the pointers to the maps
		// data is reference by ids;
		CudaMap<vtxID, InstanceData>			instanceDataMap;
		CudaMap<vtxID, GeometryData>			geometryDataMap;
		CudaMap<vtxID, MaterialData>			materialDataMap;
		CudaMap<vtxID, ShaderData>				shaderDataMap;
		CudaMap<vtxID, TextureData>				textureDataMap;
		CudaMap<vtxID, BsdfData>				bsdfDataMap;
		CudaMap<vtxID, LightProfileData>		lightProfileDataMap;
		CudaMap<vtxID, LightData>				lightDataMap;

		CameraData								cameraData;
		bool									isCameraUpdated;
		FrameBufferData							frameBufferData;
		bool									isFrameBufferUpdated;
		int										frameId = 0;
		bool									isFrameIdUpdated;
		RendererDeviceSettings					settings;
		bool									isSettingsUpdated;

		SbtProgramIdx							programs;

		std::vector<OptixInstance>				optixInstances;
		LaunchParams							launchParams;

	private:
		~UploadData() = default;
		UploadData() = default;
	};
}
