#pragma once
#include <cstdint>
#include "Device/glFrameBuffer.h"
#include "Device/DeviceData.h"

#include "Scene/Node.h"
#include "Camera.h"
#include "Group.h"

namespace vtx::graph
{
	class Renderer : public Node
	{
	public:

		Renderer();

		void resize(uint32_t width, uint32_t height);

		static void render();

		GLuint getFrame();

		void elaborateScene(std::shared_ptr<Renderer> rendererNode);

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
		std::shared_ptr<device::DeviceVisitor>			deviceVisitor;

		uint32_t										width;
		uint32_t										height;
		bool											resized = true;
	};

}
