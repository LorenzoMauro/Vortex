#pragma once
#include <imgui.h>
#include <map>
#include <memory>
#include <string>
#include <typeindex>  // For std::type_index
#include <vector>
#include <unordered_map>

namespace vtx {
	class WindowManager;

	class Window {
	public:
		virtual ~Window() = default;

		virtual void OnAttach() {}
		virtual void OnDetach() {}

		virtual void OnUpdate(float ts) {}

		virtual void preRender(){}
		virtual void renderMainContent(){}
		virtual void renderMenuBar(){}
		virtual void renderToolBar(){}

		void prepareChildWindow();
		void endChildWindowPrep();
		void OnUIRender();

		void drawStripedBackground();


		void setWindowManager(const std::shared_ptr<vtx::WindowManager>& _windowManager);
		std::shared_ptr<vtx::WindowManager> windowManager = nullptr;

		bool isOpen = true;
		std::string name = "Window";
		float toolbarPercentage = 0.1f;
		bool useToolbar = true;
		bool createWindow = true;
		bool isBorderLess = false;
		bool useStripedBackground = false;
		ImGuiWindowFlags windowFlags = 0;// = ImGuiWindowFlags_AlwaysUseWindowPadding;
		float resizerSize = 2.0f;
		float childPaddingHeight = 10.0f;
		float childPaddingWidth = 10.0f;
	};

	class WindowManager : public std::enable_shared_from_this<WindowManager>
	{
	public:

		template<typename T, typename... Args>
		void createWindow(Args&&... args) {
			static_assert(std::is_base_of_v<Window, T>, "Pushed type is not subclass of Window!");
			auto window = std::make_shared<T>(std::forward<Args>(args)...);
			window->setWindowManager(shared_from_this());
			addWindow(window);
		}

		void addWindow(const std::shared_ptr<Window>& window);

		void removeWindow(const std::shared_ptr<Window>& window);

		void updateWindows(const float timeStep) const;

		void renderWindows() const;

		void removeClosedWindows();

		std::map<std::string, std::vector<int>> selectedNodes;
	private:
		std::vector<std::shared_ptr<Window>> windows;
		std::unordered_map<std::type_index, std::shared_ptr<Window>> windowMap;

	};
}
