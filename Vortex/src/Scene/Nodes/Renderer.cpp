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
#include "Device/KernelInfos.h"
#include "Device/DevicePrograms/CudaKernels.h"
#include "Device/Wrappers/dWrapper.h"
namespace vtx::graph
{
	Renderer::Renderer() :
		Node(NT_RENDERER),
		width(getOptions()->width),
		height(getOptions()->height),
		threadData(&Renderer::render, this),
		settings(),
		waveFrontIntegrator(&settings)
	{
		settings.runOnSeparateThread = getOptions()->runOnSeparateThread;
		settings.iteration = 0;
		settings.maxBounces = getOptions()->maxBounces;
		settings.accumulate = getOptions()->accumulate;
		settings.maxSamples = getOptions()->maxSamples;
		settings.samplingTechnique = getOptions()->samplingTechnique;
		settings.displayBuffer = getOptions()->displayBuffer;
		settings.useRussianRoulette = getOptions()->useRussianRoulette;
		settings.minClamp = getOptions()->maxClamp;
		settings.maxClamp = getOptions()->minClamp;
		settings.isUpdated = true;

		settings.useWavefront = getOptions()->useWavefront;
		settings.optixShade = getOptions()->optixShade;
		settings.fitWavefront = getOptions()->fitWavefront;
		settings.parallelShade = getOptions()->parallelShade;
		settings.useLongPathKernel = getOptions()->useLongPathKernel;
		settings.longPathPercentage = getOptions()->longPathPercentage;

		settings.useNetwork = getOptions()->useNetwork;

		settings.noiseKernelSize = getOptions()->noiseKernelSize;
		settings.adaptiveSampling = getOptions()->adaptiveSampling;
		settings.minAdaptiveSamples = getOptions()->minAdaptiveSamples;
		settings.minPixelSamples = getOptions()->minPixelSamples;
		settings.maxPixelSamples = getOptions()->maxPixelSamples;
		settings.albedoNormalNoiseInfluence = getOptions()->albedoNormalNoiseInfluence;
		settings.noiseCutOff = getOptions()->noiseCutOff;

		 
		settings.fireflyKernelSize = getOptions()->fireflyKernelSize;
		settings.fireflyThreshold = getOptions()->fireflyThreshold;
		settings.removeFireflies = getOptions()->removeFireflies;

		settings.enableDenoiser = getOptions()->enableDenoiser;

		settings.denoiserStart = getOptions()->denoiserStart;
		settings.denoiserBlend = getOptions()->denoiserBlend;

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

		const LaunchParams& launchParams = UPLOAD_DATA->launchParams;
		const CUDABuffer& launchParamsBuffer = UPLOAD_BUFFERS->launchParamsBuffer;


		if (launchParams.frameBuffer.frameSize.x == 0 || launchParams.frameBuffer.frameSize.y == 0)
		{
			return;
		}
		if(!settings.accumulate)
		{
			settings.iteration = 0;
		}
		if (settings.iteration == 0)
		{
			resetKernelStats();
			timer.reset();
			internalIteration = 0;
		}

		{
			//timer.reset();
			const std::pair<cudaEvent_t, cudaEvent_t> events = GetProfilerEvents(eventNames[R_NOISE_COMPUTATION]);
			cudaEventRecord(events.first);

			if (settings.adaptiveSampling && settings.minAdaptiveSamples <= settings.iteration)
			{
				if (settings.adaptiveSampling && settings.iteration >= settings.minAdaptiveSamples)
				{
					noiseComputation(launchParamsBuffer.castedPointer<LaunchParams>(),settings,getID());
				}
			}
			cudaEventRecord(events.second);

		}

		{
			const std::pair<cudaEvent_t, cudaEvent_t> events = GetProfilerEvents(eventNames[R_TRACE]);
			cudaEventRecord(events.first);

			if(!settings.useWavefront)
			{
				const optix::State& state = *(optix::getState());
				const OptixPipeline& pipeline = optix::getRenderingPipeline()->getPipeline();
				const OptixShaderBindingTable& sbt = optix::getRenderingPipeline()->getSbt();

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
			}
			else
			{
				waveFrontIntegrator.render();
			}
			cudaEventRecord(events.second);
		}

		{

			const std::pair<cudaEvent_t, cudaEvent_t> events = GetProfilerEvents(eventNames[R_POSTPROCESSING]);
			cudaEventRecord(events.first);

			toneMapRadianceKernel(launchParamsBuffer.castedPointer<LaunchParams>(), width, height, eventNames[R_TONE_MAP_RADIANCE]);
			math::vec3f* beauty = nullptr;
			if (settings.removeFireflies)
			{
				removeFireflies(launchParamsBuffer.castedPointer<LaunchParams>(), settings.fireflyKernelSize, settings.fireflyThreshold, width, height);
				beauty = GET_BUFFER(device::Buffers::FrameBufferBuffers, getID(), fireflyRemoval).castedPointer<math::vec3f>();
				//CUDA_SYNC_CHECK();
			}

			if(settings.enableDenoiser && settings.iteration>settings.denoiserStart)
			{
				CUDABuffer& denoiserRadianceInput = settings.removeFireflies ? GET_BUFFER(device::Buffers::FrameBufferBuffers, getID(), fireflyRemoval) : GET_BUFFER(device::Buffers::FrameBufferBuffers, getID(), hdriRadiance);
				CUDABuffer& albedoBuffer = GET_BUFFER(device::Buffers::FrameBufferBuffers, getID(), albedoNormalized);
				CUDABuffer& normalBuffer = GET_BUFFER(device::Buffers::FrameBufferBuffers, getID(), normalNormalized);
				optix::getState()->denoiser.setInputs(denoiserRadianceInput, albedoBuffer, normalBuffer);
				beauty = optix::getState()->denoiser.denoise(settings.denoiserBlend);
			}

			switchOutput(launchParamsBuffer.castedPointer<LaunchParams>(), width, height, beauty);
			cudaEventRecord(events.second);
		}

		{
			const std::pair<cudaEvent_t, cudaEvent_t> events = GetProfilerEvents(eventNames[R_DISPLAY]);
			cudaEventRecord(events.first);
			copyToGl();
			cudaEventRecord(events.second);
		}

		CudaEventTimes cuTimes = getCudaEventTimes();
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
		//CUDA_SYNC_CHECK();
		const optix::State& state = *(optix::getState());
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
		overallTime = timer.elapsedMillis();
		internalIteration++;
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

