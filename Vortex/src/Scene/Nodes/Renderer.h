#pragma once
#include <cstdint>
#include "Device/glFrameBuffer.h"
#include "Device/DeviceVisitor.h"
#include "Scene/Node.h"
#include "Camera.h"
#include "Group.h"
#include "Core/Timer.h"
#include <mutex>
#include <condition_variable>

#include "Device/DevicePrograms/WavefrontIntegrator.h"

namespace vtx::graph
{
	struct RendererSettings
	{
		int                                       iteration;
		int                                       maxBounces;
		int                                       maxSamples;
		bool                                      accumulate;
		RendererDeviceSettings::SamplingTechnique samplingTechnique;
		RendererDeviceSettings::DisplayBuffer     displayBuffer;
		bool                                      isUpdated;
		float                                     minClamp;
		float                                     maxClamp;
		int                                       noiseKernelSize;
		bool                                      adaptiveSampling;
		int                                       minAdaptiveSamples;
		int                                       minPixelSamples;
		int                                       maxPixelSamples;
		float                                     albedoNormalNoiseInfluence;
		float                                     noiseCutOff;
		int                                       fireflyKernelSize;
		float                                     fireflyThreshold;
		bool                                      removeFireflies;
		bool                                      enableDenoiser;
		int                                       denoiserStart;
		float                                     denoiserBlend;
		bool                                      useWavefront;
		bool                                      useRussianRoulette;
		bool                                       fitWavefront;
	};

	struct ToneMapperSettings
	{
		math::vec3f whitePoint;
		math::vec3f colorBalance;
		float       burnHighlights;
		float       crushBlacks;
		float       saturation;
		float       gamma;
		bool         isUpdated;
	};

	class Renderer : public Node
	{
	public:

		Renderer();

		void resize(uint32_t width, uint32_t height);

		void render();

		GlFrameBuffer getFrame();

		void setCamera(const std::shared_ptr<Camera>& cameraNode);

		void setScene(const std::shared_ptr<Group>& sceneRootNode);

		std::shared_ptr<Camera> getCamera();

		std::shared_ptr<Group> getScene();

		void traverse(const std::vector<std::shared_ptr<NodeVisitor>>& orderedVisitors) override;

		//void accept(std::shared_ptr<NodeVisitor> visitor) override;

		bool isReady(bool setBusy = false);

		void threadedRender();

		void copyToGl();
		void  setWindow(GLFWwindow* window);

		int getWavefrontLaunches();
		KernelTimes& getWaveFrontTimes();

	public:
		//GL Interop
		WaveFrontIntegrator								waveFrontIntegrator;
		GlFrameBuffer									drawFrameBuffer;
		GlFrameBuffer									displayFrameBuffer;
		CUgraphicsResource								cudaGraphicsResource = nullptr;
		CUarray											dstArray;

		std::shared_ptr<Camera>							camera;
		std::shared_ptr<Group>							sceneRoot;

		uint32_t										width;
		uint32_t										height;
		RendererSettings								settings;
		ToneMapperSettings								toneMapperSettings;
		bool											resized = true;

		vtx::Timer timer;

		float		noiseComputationTime;
		float		traceComputationTime;
		float		postProcessingComputationTime;
		float		displayComputationTime;
		float      frameTime;
		float      fps;
		float      totalTimeSeconds;
		float      sppS;
		float      averageFrameTime;

		struct ThreadData {
			template <typename Fn>
			ThreadData(Fn renderFunction, Renderer* instance):
				renderThreadBusy(false),
				exitRenderThread(false),
				bufferUpdateReady(false){
				renderThread = std::thread([this, renderFunction, instance] {
					while (true) {
						std::unique_lock<std::mutex> lock(renderMutex);
						renderCondition.wait(lock, [this] { return exitRenderThread || renderThreadBusy; });
						if (exitRenderThread) {
							break;
						}
						glfwMakeContextCurrent(instance->sharedContext); // Make the shared context current in this thread
						std::invoke(renderFunction, instance); // Use std::invoke to call the member function pointer
						renderThreadBusy = false;
						bufferUpdateReady = true;
					}
				});
			}

			~ThreadData() {
				{
					std::lock_guard<std::mutex> lock(renderMutex);
					exitRenderThread = true;
				}
				renderCondition.notify_one();
				renderThread.join();
			}

			bool renderThreadBusy;
			bool exitRenderThread;
			std::mutex renderMutex;
			std::condition_variable renderCondition;
			std::thread renderThread;
			std::atomic<bool> bufferUpdateReady;
		} threadData;

		bool        resizeGlBuffer;
		GLFWwindow* sharedContext;
	};

}
