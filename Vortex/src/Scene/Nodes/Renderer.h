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

#include "RendererSettings.h"
#include "Device/DevicePrograms/WavefrontIntegrator.h"
namespace vtx::graph
{
	class Renderer : public Node
	{
	public:

		Renderer();

		void resize(int width, int height);

		void render();

		GlFrameBuffer getFrame();

		std::vector<std::shared_ptr<Node>> getChildren() const override;

		bool isReady(bool setBusy = false);

		void threadedRender();

		void copyToGl();
		void  setWindow(GLFWwindow* window);

		void accept(NodeVisitor& visitor) override;

		vtxID getInstanceIdOnClick(int pixelID);
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
		bool											isSizeLocked = false;
		RendererSettings								settings;
		bool											resized = true;
		vtxID         selectedId = 0;

		vtx::Timer timer;

		float overallTime;
		int internalIteration = 0;

		struct ThreadData {
			template <typename Fn>
			ThreadData(Fn renderFunction, Renderer* instance):
				renderThreadBusy(false),
				exitRenderThread(false),
				bufferUpdateReady(false){
				renderThread = std::thread([this, renderFunction, instance] {
					while (true) {
						// start timer
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
