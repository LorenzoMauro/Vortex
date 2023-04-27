#pragma once
#include <cstdint>
#include "Device/glFrameBuffer.h"
#include "Device/DeviceVisitor.h"
#include "Scene/Node.h"
#include "Camera.h"
#include "Group.h"
#include "Core/Timer.h"

namespace vtx::graph
{

	struct RendererSettings
	{
		int			iteration;
		int			maxBounces;
		int			maxSamples;
		bool		accumulate;
		bool		isUpdated;
	};

	class Renderer : public Node
	{
	public:

		Renderer();

		void resize(uint32_t width, uint32_t height);

		void render();

		GLuint getFrame();

		void setCamera(std::shared_ptr<Camera> _camera);

		void setScene(std::shared_ptr<Group> _sceneRoot);

		std::shared_ptr<Camera> getCamera();

		std::shared_ptr<Group> getScene();

		void traverse(const std::vector<std::shared_ptr<NodeVisitor>>& orderedVisitors) override;

		void accept(std::shared_ptr<NodeVisitor> visitor) override;

	private:
		void copyToGl();

	public:
		//GL Interop
		GlFrameBuffer									glFrameBuffer;
		CUgraphicsResource								cudaGraphicsResource = nullptr;
		CUarray											dstArray;

		std::shared_ptr<Camera>							camera;
		std::shared_ptr<Group>							sceneRoot;

		uint32_t										width;
		uint32_t										height;
		RendererSettings								settings;
		bool											resized = true;
		FrameBufferData::FrameBufferType				frameBufferType = FrameBufferData::FB_NOISY;

		vtx::Timer timer;
		float      frameTime;
		float      fps;
		float      totalTimeSeconds;
		float      sppS;
		float      averageFrameTime;
	};

}
