#pragma once

#include "keycodes.h"
#include "Core/Math.h"

namespace vtx {

	class Input
	{
	public:
		static void SetWindowHandle(GLFWwindow* _window);
		//static void ResetWindowHandle();
		static bool IsKeyDown(KeyCode keycode);
		static bool IsMouseButtonDown(MouseButton button);

		static math::vec2f GetMousePosition();

		static void SetCursorMode(CursorMode mode);
		static float MouseWheel();
	};

}
