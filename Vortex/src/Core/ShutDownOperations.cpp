#include "ShutDownOperations.h"
#include "ImGuiOp.h"
#include "Device/OptixWrapper.h"
#include "Device/UploadCode/UploadBuffers.h"
#include "Device/UploadCode/UploadData.h"
#include "MDL/MdlWrapper.h"
#include "Scene/SIM.h"
#include "Scene/Nodes/Renderer.h"


void vtx::shutDownOperations()
{
	// Wait for all rendering threads to finish
	VTX_INFO("ShutDown: Waiting For Render Threads");
	const std::vector<std::shared_ptr<graph::Renderer>> rendererNodes = graph::SIM::getAllNodeOfType<graph::Renderer>(graph::NT_RENDERER);

	bool renderingConcluded = true;

	while(renderingConcluded!=true)
	{
		renderingConcluded = true;
		for (const std::shared_ptr<graph::Renderer> renderer : rendererNodes)
		{
			if(!renderer->isReady())
			{
				renderingConcluded = false;
			}
		}
	}
	VTX_INFO("ShutDown: Render Threads exitedd");

	UPLOAD_BUFFERS->shutDown();
	UPLOAD_DATA->shutDown();

	optix::shutDown();
	mdl::shutDown();
	shutDownImGui();
}
