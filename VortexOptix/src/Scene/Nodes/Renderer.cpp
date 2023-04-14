#include "Renderer.h"
#include <optix_function_table_definition.h>
#include "Device/CUDAChecks.h"
#include "Core/Options.h"
#include "Core/Utils.h"
#include <cudaGL.h>
#include "Device/OptixUtils.h"
#include "Scene/Traversal.h"

namespace vtx::graph
{
	Renderer::Renderer() :
		Node(NT_RENDERER),
		width(getOptions()->width),
		height(getOptions()->height)
	{
		deviceVisitor = std::make_shared<device::DeviceVisitor>();
	}

	void Renderer::elaborateScene(std::shared_ptr<Renderer> rendererNode)
	{
		std::vector visitors = { std::dynamic_pointer_cast<NodeVisitor>(deviceVisitor) };
		traverse(visitors);
		device::finalizeUpload();
	}

	void Renderer::setCamera(std::shared_ptr<Camera> _camera) {
		camera = _camera;
	}

	void Renderer::setScene(std::shared_ptr<Group> _sceneRoot) {
		sceneRoot = _sceneRoot;
	}

	std::shared_ptr<Camera> Renderer::getCamera() {
		return camera;
	}

	std::shared_ptr<Group> Renderer::getScene() {
		return sceneRoot;
	}

	void Renderer::traverse(const std::vector<std::shared_ptr<NodeVisitor>>& orderedVisitors)
	{
		camera->traverse(orderedVisitors);
		sceneRoot->traverse(orderedVisitors);
		ACCEPT(orderedVisitors);
	}

	void Renderer::accept(std::shared_ptr<NodeVisitor> visitor)
	{
		visitor->visit(sharedFromBase<Renderer>());
	}

	void Renderer::render()
	{
		const LaunchParams& launchParams = device::getLaunchParams();
		const CUDABuffer& launchParamsBuffer = device::getLaunchParamsBuffer();

		if (launchParams.fbSize.x == 0) return;

		device::incrementFrame();

		const optix::State& state = *(optix::getState());
		const optix::OptixRenderingPipelineData& orp = *(optix::getRenderingPipeline());

		const auto result = optixLaunch(/*! pipeline we're launching launch: */
			orp.pipeline, state.stream,
			/*! parameters and SBT */
			launchParamsBuffer.d_pointer(),
			launchParamsBuffer.sizeInBytes,
			&orp.sbt,
			/*! dimensions of the launch: */
			launchParams.fbSize.x,
			launchParams.fbSize.y,
			1
		);
		OPTIX_CHECK(result);
	}

	void Renderer::resize(uint32_t _width, uint32_t _height) {
		if (width == _width && height == _height) {
			return;
		}
		width = _width;
		height = _height;
		resized = true;
		camera->resize(width, height);
	}

	void Renderer::copyToGl() {
		optix::State& state = *(optix::getState());
		optix::OptixRenderingPipelineData& orp = *(optix::getRenderingPipeline());
		LaunchParams& launchParams = device::getLaunchParams();

		CU_CHECK(cuGraphicsMapResources(1, &cudaGraphicsResource, state.stream)); // This is an implicit cuSynchronizeStream().

		CU_CHECK(cuGraphicsSubResourceGetMappedArray(&dstArray, cudaGraphicsResource, 0, 0)); // arrayIndex = 0, mipLevel = 0

		CUDA_MEMCPY3D params = {};

		params.srcMemoryType = CU_MEMORYTYPE_DEVICE;
		params.srcDevice = launchParams.colorBuffer;
		params.srcPitch = launchParams.fbSize.x * sizeof(uint32_t);
		params.srcHeight = launchParams.fbSize.y;

		params.dstMemoryType = CU_MEMORYTYPE_ARRAY;
		params.dstArray = dstArray;
		params.WidthInBytes = launchParams.fbSize.x * sizeof(uint32_t);
		params.Height = launchParams.fbSize.y;
		params.Depth = 1;

		CU_CHECK(cuMemcpy3D(&params)); // Copy from linear to array layout.

		CU_CHECK(cuGraphicsUnmapResources(1, &cudaGraphicsResource, state.stream)); // This is an implicit cuSynchronizeStream().
	}

	GLuint Renderer::getFrame() {
		glFrameBuffer.Bind();
		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);
		glFrameBuffer.Unbind();
		copyToGl();
		return glFrameBuffer.m_ColorAttachment;
	}
}

