#include "ShutDownOperations.h"
#include "ImGuiOp.h"
#include "Device/UploadCode/UploadBuffers.h"
#include "Device/UploadCode/UploadData.h"
#include "MDL/MdlWrapper.h"
#include "Scene/SIM.h"
#include "Scene/Nodes/Renderer.h"


void vtx::shutDownOperations()
{
	// Wait for all rendering threads to finish
	VTX_INFO("ShutDown: Waiting For Render Threads");
	const std::vector<std::shared_ptr<graph::Node>> nodes = graph::SIM::getAllNodeOfType(graph::NT_RENDERER);
	std::vector<std::shared_ptr<graph::Renderer>> rendererNodes;

	bool renderingConcluded = true;
	for (const std::shared_ptr<graph::Renderer>& node : rendererNodes)
	{
		auto renderer = std::dynamic_pointer_cast<graph::Renderer>(node);
		rendererNodes.push_back(renderer);

		if (!renderer->isReady())
		{
			renderingConcluded = false;
		}
	}

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
