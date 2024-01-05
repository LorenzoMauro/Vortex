#include "ImageWindowPopUp.h"
#include <imgui.h>
#include <unordered_map>
#include "Device/InteropWrapper.h"
#include "Device/CudaFunctions/cudaFunctions.h"
#include "Device/UploadCode/CUDABuffer.h"

namespace vtx::gui
{
	static std::unordered_map<std::string, InteropWrapper> interopWrappers;

	void popUpImageWindow(const std::string& nameIdentifier, CUDABuffer imageBuffer, int width, int height, int nChannels, bool forceRebuild)
	{
		if (imageBuffer.bytesSize() == 0)
		{
			VTX_WARN("Pop Up Image Window {} has null Buffer!", nameIdentifier);
			return;
		}

		{
			CUDABuffer* displayBuffer = &imageBuffer;
			bool deleteDisplayBuffer = false;
			if (nChannels != 4)
			{
				switch(nChannels)
				{
				case 1:
					{
						displayBuffer = new CUDABuffer();
						vtx::cuda::copyRtoRGBA(imageBuffer, *displayBuffer, width, height);
						deleteDisplayBuffer = true;
						break;
					}
				case 3:
					{
						displayBuffer = new CUDABuffer();
						vtx::cuda::copyRGBtoRGBA(imageBuffer, *displayBuffer, width, height);
						deleteDisplayBuffer = true;
						break;
					}
				}
			}

			if(displayBuffer->bytesSize() == 0)
			{
				VTX_WARN("Pop Up Image Window {} has null Buffer!", nameIdentifier);
				return;
			}

			InteropWrapper& imageInterop = interopWrappers[nameIdentifier];

			if (
				imageInterop.cuArray == nullptr ||
				imageInterop.cuGraphicResource == nullptr ||
				width != (int)imageInterop.glFrameBuffer.width ||
				height != (int)imageInterop.glFrameBuffer.height ||
				forceRebuild
				)
			{
				imageInterop.prepare(width, height, 4, InteropUsage::SingleThreaded);
				imageInterop.copyToGlBuffer(displayBuffer->dPointer(), width, height);
			}

			const auto glBf = imageInterop.glFrameBuffer;
			ImGui::Image((ImTextureID)glBf.colorAttachment, ImVec2{ static_cast<float>(glBf.width), static_cast<float>(glBf.height) }, ImVec2{ 0, 1 }, ImVec2{ 1, 0 });

			if (deleteDisplayBuffer)
			{
				displayBuffer->free();
				delete displayBuffer;
			}
		}
	}

}
