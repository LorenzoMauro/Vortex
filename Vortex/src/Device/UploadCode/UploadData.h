﻿#pragma once
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
		CudaMap<vtxID, std::tuple<InstanceData,		InstanceData*>>				instanceDataMap;
		CudaMap<vtxID, std::tuple<GeometryData,		GeometryData*>>				geometryDataMap;
		CudaMap<vtxID, std::tuple<MaterialData,		MaterialData*>>				materialDataMap;
		CudaMap<vtxID, std::tuple<ShaderData,		ShaderData*>>				shaderDataMap;
		CudaMap<vtxID, std::tuple<TextureData,		TextureData*>>				textureDataMap;
		CudaMap<vtxID, std::tuple<BsdfData,			BsdfData*>>					bsdfDataMap;
		CudaMap<vtxID, std::tuple<LightProfileData, LightProfileData*>>			lightProfileDataMap;
		CudaMap<vtxID, std::tuple<LightData,		LightData*>>				lightDataMap;

		CameraData								cameraData;
		bool									isCameraUpdated;
		FrameBufferData							frameBufferData;
		bool									isFrameBufferUpdated = false;
		int										frameId = 0;
		bool									isFrameIdUpdated = false;
		RendererDeviceSettings					settings;
		bool									isSettingsUpdated = false;
		ToneMapperSettings						toneMapperSettings;
		bool									isToneMapperSettingsUpdated = false;			

		SbtProgramIdx							programs;

		std::vector<OptixInstance> optixInstances;
		LaunchParams               launchParams;

	private:
		~UploadData() = default;
		UploadData() = default;
	};
}
