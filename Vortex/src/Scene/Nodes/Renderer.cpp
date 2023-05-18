#include "Renderer.h"
#include <optix_function_table_definition.h>
#include "Device/CUDAChecks.h"
#include "Core/Options.h"
#include "Core/Utils.h"
#include <cudaGL.h>
#include "Device/OptixWrapper.h"
#include "Device/UploadCode/UploadBuffers.h"
#include "Device/UploadCode/UploadData.h"
#include "Scene/Traversal.h"
#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include "Device/UploadCode/UploadFunctions.h"

namespace vtx::graph
{


	Renderer::Renderer() :
		Node(NT_RENDERER),
		width(getOptions()->width),
		height(getOptions()->height),
		threadData(&Renderer::render, this)
	{
		

		settings = RendererSettings();
		settings.iteration = 0;
		settings.maxBounces = getOptions()->maxBounces;
		settings.accumulate = getOptions()->accumulate;
		settings.maxSamples = getOptions()->maxSamples;
		settings.samplingTechnique = getOptions()->samplingTechnique;
		settings.displayBuffer = getOptions()->displayBuffer;

		settings.minClamp = getOptions()->maxClamp;
		settings.maxClamp = getOptions()->minClamp;
		settings.isUpdated = true;


		settings.noiseKernelSize = getOptions()->noiseKernelSize;
		settings.adaptiveSampling = getOptions()->adaptiveSampling;
		settings.minAdaptiveSamples = getOptions()->minAdaptiveSamples;
		settings.minPixelSamples = getOptions()->minPixelSamples;
		settings.maxPixelSamples = getOptions()->maxPixelSamples;
		settings.albedoNormalNoiseInfluence = getOptions()->albedoNormalNoiseInfluence;
		settings.noiseCutOff = getOptions()->noiseCutOff;

		toneMapperSettings.whitePoint = getOptions()->whitePoint;
		toneMapperSettings.colorBalance = getOptions()->colorBalance;
		toneMapperSettings.burnHighlights = getOptions()->burnHighlights;
		toneMapperSettings.crushBlacks = getOptions()->crushBlacks;
		toneMapperSettings.saturation = getOptions()->saturation;
		toneMapperSettings.gamma = getOptions()->gamma;
		toneMapperSettings.isUpdated = true;

		drawFrameBuffer.setSize(width, height);
		drawFrameBuffer.bind();
		glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);
		drawFrameBuffer.unbind();

		displayFrameBuffer.setSize(width, height);
		displayFrameBuffer.bind();
		glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);
		displayFrameBuffer.unbind();
	}

	void Renderer::setCamera(const std::shared_ptr<Camera>& cameraNode) {
		camera = cameraNode;
	}

	void Renderer::setScene(const std::shared_ptr<Group>& sceneRootNode) {
		sceneRoot = sceneRootNode;
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
		ACCEPT(Renderer,orderedVisitors)
	}

	/*void Renderer::accept(std::shared_ptr<NodeVisitor> visitor)
	{
		visitor->visit(sharedFromBase<Renderer>());
	}*/

	bool Renderer::isReady(const bool setBusy) {
		if (threadData.renderMutex.try_lock()) {
			std::unique_lock<std::mutex> lock(threadData.renderMutex, std::adopt_lock);
			if (!threadData.renderThreadBusy) {
				if (setBusy) {
					threadData.renderThreadBusy = true;
				}
				return true;
			}
			return false;
		}
		return false;
	}

	void Renderer::threadedRender()
	{
		if(isReady(true))
		{
			std::unique_lock<std::mutex> lock(threadData.renderMutex);
			threadData.renderThreadBusy = true;
			threadData.renderCondition.notify_one();
		}

	}

	void Renderer::render()
	{
		timer.reset();

		const LaunchParams& launchParams = UPLOAD_DATA->launchParams;
		const CUDABuffer& launchParamsBuffer = UPLOAD_BUFFERS->launchParamsBuffer;

		
		if (launchParams.frameBuffer.frameSize.x == 0)
		{
			return;
		}

		if (settings.adaptiveSampling && settings.minAdaptiveSamples <= settings.iteration)
		{
			device::computeNoiseInfo(sharedFromBase<Renderer>());
		}

		device::incrementFrame();

		const optix::State& state = *(optix::getState());
		const OptixPipeline& pipeline = optix::getRenderingPipeline()->getPipeline();
		const OptixShaderBindingTable& sbt = optix::getRenderingPipeline()->getSbt();

		CUDA_SYNC_CHECK();
		const auto result = optixLaunch(/*! pipeline we're launching launch: */
			pipeline, state.stream,
			/*! parameters and SBT */
			launchParamsBuffer.dPointer(),
			launchParamsBuffer.bytesSize(),
			&sbt,
			/*! dimensions of the launch: */
			launchParams.frameBuffer.frameSize.x,
			launchParams.frameBuffer.frameSize.y,
			1
		);
		OPTIX_CHECK(result);
		copyToGl();
		frameTime = timer.elapsedMillis();
		if (settings.iteration == 0)
		{
			totalTimeSeconds = 0;
		}
		else
		{
			totalTimeSeconds += frameTime / 1000.0f;
		}

		fps = (float)(settings.iteration + 1) / totalTimeSeconds;
		sppS = ((float)width * (float)height * ((float)settings.iteration + 1)) / totalTimeSeconds;
		averageFrameTime = (float)(settings.iteration + 1) / (totalTimeSeconds * 1000.0f);
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
		// Update the GL buffer here
		CUDA_SYNC_CHECK();
		optix::State& state = *(optix::getState());
		const LaunchParams& launchParams = UPLOAD_DATA->launchParams;

		if(resizeGlBuffer)
		{
			if (cudaGraphicsResource != nullptr) {
				const CUresult result = cuGraphicsUnregisterResource(cudaGraphicsResource);
				CU_CHECK(result);
			}
			drawFrameBuffer.setSize(launchParams.frameBuffer.frameSize.x, launchParams.frameBuffer.frameSize.y);

			const CUresult result = cuGraphicsGLRegisterImage(&cudaGraphicsResource,
															  drawFrameBuffer.colorAttachment,
															  GL_TEXTURE_2D, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD);
			CU_CHECK(result);
			resizeGlBuffer = false;
		}

		drawFrameBuffer.bind();
		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);
		drawFrameBuffer.unbind();

		CUresult CUresult = cuGraphicsMapResources(1, &cudaGraphicsResource, state.stream); // This is an implicit cuSynchronizeStream().
		CU_CHECK(CUresult);
		CUresult = cuGraphicsSubResourceGetMappedArray(&dstArray, cudaGraphicsResource, 0, 0); // arrayIndex = 0, mipLevel = 0
		CU_CHECK(CUresult);

		CUDA_MEMCPY3D params = {};
		params.srcMemoryType = CU_MEMORYTYPE_DEVICE;
		params.srcDevice = launchParams.frameBuffer.outputBuffer;
		params.srcPitch = launchParams.frameBuffer.frameSize.x * sizeof(math::vec4f);
		params.srcHeight = launchParams.frameBuffer.frameSize.y;

		params.dstMemoryType = CU_MEMORYTYPE_ARRAY;
		params.dstArray = dstArray;
		params.WidthInBytes = launchParams.frameBuffer.frameSize.x * sizeof(math::vec4f);
		params.Height = launchParams.frameBuffer.frameSize.y;
		params.Depth = 1;

		CU_CHECK(cuMemcpy3D(&params)); // Copy from linear to array layout.

		CU_CHECK(cuGraphicsUnmapResources(1, &cudaGraphicsResource, state.stream)); // This is an implicit cuSynchronizeStream().

		//// SWAP BUFFERS ///////////////////
		//threadData.bufferUpdateReady = false;
		//auto* tempBuffer = updatedFrameBuffer;
		displayFrameBuffer.copyToThis(drawFrameBuffer);
		//currentFrameBuffer = tempBuffer;
	}

	void Renderer::setWindow(GLFWwindow* window)
	{
		// Create a shared OpenGL context for the separate thread
		glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
		sharedContext = glfwCreateWindow(1, 1, "Shared Context", nullptr, window);
	}

	GlFrameBuffer Renderer::getFrame() {
		//if(threadData.bufferUpdateReady)
		//{
		//	// This section is helpfull when the renderer update is slow, else it's better not to have it!
		//	copyToGl();
		//	threadData.bufferUpdateReady = false;
		//}
		return displayFrameBuffer;
		
	}
}

