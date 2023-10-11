#include "ShutDownOperations.h"
#include "ImGuiOp.h"
#include "Device/OptixWrapper.h"
#include "Device/UploadCode/UploadBuffers.h"
#include "Device/UploadCode/DeviceDataCoordinator.h"
#include "MDL/MdlWrapper.h"
#include "Scene/SceneIndexManager.h"
#include "Scene/Nodes/Renderer.h"


void vtx::shutDownOperations()
{
	// Wait for all rendering threads to finish
	VTX_INFO("ShutDown: Waiting For Render Threads");
	const std::vector<std::shared_ptr<graph::Renderer>> rendererNodes = graph::Scene::getSim()->getAllNodeOfType<graph::Renderer>(graph::NT_RENDERER);

	bool renderingConcluded = false;

	while(renderingConcluded!=true)
	{
		renderingConcluded = true;
		for (const std::shared_ptr<graph::Renderer>& renderer : rendererNodes)
		{
			if(!renderer->isReady())
			{
				renderingConcluded = false;
			}
			else
			{
				renderer->settings.runOnSeparateThread = false;
			}
		}
	}
	VTX_INFO("ShutDown: Render Threads exited");

	UPLOAD_BUFFERS->shutDown();

	optix::shutDown();
	mdl::shutDown();
	shutDownImGui();
}
