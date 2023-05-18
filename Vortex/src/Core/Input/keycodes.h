#pragma once

#include <stdint.h>
#include <iostream>
#include <imgui.h>
#include "GLFW/glfw3.h"

namespace vtx {

	typedef enum class KeyCode : uint16_t
	{
		// From glfw3.h
		Space = 32,
		Apostrophe = 39, /* ' */
		Comma = 44, /* , */
		Minus = 45, /* - */
		Period = 46, /* . */
		Slash = 47, /* / */

		D0 = 48, /* 0 */
		D1 = 49, /* 1 */
		D2 = 50, /* 2 */
		D3 = 51, /* 3 */
		D4 = 52, /* 4 */
		D5 = 53, /* 5 */
		D6 = 54, /* 6 */
		D7 = 55, /* 7 */
		D8 = 56, /* 8 */
		D9 = 57, /* 9 */

		Semicolon = 59, /* ; */
		Equal = 61, /* = */

		A = 65,
		B = 66,
		C = 67,
		D = 68,
		E = 69,
		F = 70,
		G = 71,
		H = 72,
		I = 73,
		J = 74,
		K = 75,
		L = 76,
		M = 77,
		N = 78,
		O = 79,
		P = 80,
		Q = 81,
		R = 82,
		S = 83,
		T = 84,
		U = 85,
		V = 86,
		W = 87,
		X = 88,
		Y = 89,
		Z = 90,

		LeftBracket = 91,  /* [ */
		Backslash = 92,  /* \ */
		RightBracket = 93,  /* ] */
		GraveAccent = 96,  /* ` */

		World1 = 161, /* non-US #1 */
		World2 = 162, /* non-US #2 */

		/* Function keys */
		Escape = 256,
		Enter = 257,
		Tab = 258,
		Backspace = 259,
		Insert = 260,
		Delete = 261,
		Right = 262,
		Left = 263,
		Down = 264,
		Up = 265,
		PageUp = 266,
		PageDown = 267,
		Home = 268,
		End = 269,
		CapsLock = 280,
		ScrollLock = 281,
		NumLock = 282,
		PrintScreen = 283,
		Pause = 284,
		F1 = 290,
		F2 = 291,
		F3 = 292,
		F4 = 293,
		F5 = 294,
		F6 = 295,
		F7 = 296,
		F8 = 297,
		F9 = 298,
		F10 = 299,
		F11 = 300,
		F12 = 301,
		F13 = 302,
		F14 = 303,
		F15 = 304,
		F16 = 305,
		F17 = 306,
		F18 = 307,
		F19 = 308,
		F20 = 309,
		F21 = 310,
		F22 = 311,
		F23 = 312,
		F24 = 313,
		F25 = 314,

		/* Keypad */
		KP0 = 320,
		KP1 = 321,
		KP2 = 322,
		KP3 = 323,
		KP4 = 324,
		KP5 = 325,
		KP6 = 326,
		KP7 = 327,
		KP8 = 328,
		KP9 = 329,
		KPDecimal = 330,
		KPDivide = 331,
		KPMultiply = 332,
		KPSubtract = 333,
		KPAdd = 334,
		KPEnter = 335,
		KPEqual = 336,

		LeftShift = 340,
		LeftControl = 341,
		LeftAlt = 342,
		LeftSuper = 343,
		RightShift = 344,
		RightControl = 345,
		RightAlt = 346,
		RightSuper = 347,
		Menu = 348
	} Key;

	enum class KeyState
	{
		None = -1,
		Pressed,
		Held,
		Released
	};

	enum class CursorMode
	{
		Normal = 0,
		Hidden = 1,
		Locked = 2
	};

	typedef enum class MouseButton : uint16_t
	{
		Button0 = 0,
		Button1 = 1,
		Button2 = 2,
		Button3 = 3,
		Button4 = 4,
		Button5 = 5,
		Left = Button0,
		Right = Button1,
		Middle = Button2
	} Button;

	static ImGuiKey MapInputToImGui(MouseButton button) {
		switch (button) {
			case(MouseButton::Left): return ImGuiKey_MouseLeft;
			case(MouseButton::Right): return ImGuiKey_MouseRight;
			case(MouseButton::Middle): return ImGuiKey_MouseMiddle;
		}
	}

	static ImGuiKey MapInputToImGui(KeyCode code) {
		switch ((int)code)
		{
			case GLFW_KEY_TAB: return ImGuiKey_Tab;
			case GLFW_KEY_LEFT: return ImGuiKey_LeftArrow;
			case GLFW_KEY_RIGHT: return ImGuiKey_RightArrow;
			case GLFW_KEY_UP: return ImGuiKey_UpArrow;
			case GLFW_KEY_DOWN: return ImGuiKey_DownArrow;
			case GLFW_KEY_PAGE_UP: return ImGuiKey_PageUp;
			case GLFW_KEY_PAGE_DOWN: return ImGuiKey_PageDown;
			case GLFW_KEY_HOME: return ImGuiKey_Home;
			case GLFW_KEY_END: return ImGuiKey_End;
			case GLFW_KEY_INSERT: return ImGuiKey_Insert;
			case GLFW_KEY_DELETE: return ImGuiKey_Delete;
			case GLFW_KEY_BACKSPACE: return ImGuiKey_Backspace;
			case GLFW_KEY_SPACE: return ImGuiKey_Space;
			case GLFW_KEY_ENTER: return ImGuiKey_Enter;
			case GLFW_KEY_ESCAPE: return ImGuiKey_Escape;
			case GLFW_KEY_APOSTROPHE: return ImGuiKey_Apostrophe;
			case GLFW_KEY_COMMA: return ImGuiKey_Comma;
			case GLFW_KEY_MINUS: return ImGuiKey_Minus;
			case GLFW_KEY_PERIOD: return ImGuiKey_Period;
			case GLFW_KEY_SLASH: return ImGuiKey_Slash;
			case GLFW_KEY_SEMICOLON: return ImGuiKey_Semicolon;
			case GLFW_KEY_EQUAL: return ImGuiKey_Equal;
			case GLFW_KEY_LEFT_BRACKET: return ImGuiKey_LeftBracket;
			case GLFW_KEY_BACKSLASH: return ImGuiKey_Backslash;
			case GLFW_KEY_RIGHT_BRACKET: return ImGuiKey_RightBracket;
			case GLFW_KEY_GRAVE_ACCENT: return ImGuiKey_GraveAccent;
			case GLFW_KEY_CAPS_LOCK: return ImGuiKey_CapsLock;
			case GLFW_KEY_SCROLL_LOCK: return ImGuiKey_ScrollLock;
			case GLFW_KEY_NUM_LOCK: return ImGuiKey_NumLock;
			case GLFW_KEY_PRINT_SCREEN: return ImGuiKey_PrintScreen;
			case GLFW_KEY_PAUSE: return ImGuiKey_Pause;
			case GLFW_KEY_KP_0: return ImGuiKey_Keypad0;
			case GLFW_KEY_KP_1: return ImGuiKey_Keypad1;
			case GLFW_KEY_KP_2: return ImGuiKey_Keypad2;
			case GLFW_KEY_KP_3: return ImGuiKey_Keypad3;
			case GLFW_KEY_KP_4: return ImGuiKey_Keypad4;
			case GLFW_KEY_KP_5: return ImGuiKey_Keypad5;
			case GLFW_KEY_KP_6: return ImGuiKey_Keypad6;
			case GLFW_KEY_KP_7: return ImGuiKey_Keypad7;
			case GLFW_KEY_KP_8: return ImGuiKey_Keypad8;
			case GLFW_KEY_KP_9: return ImGuiKey_Keypad9;
			case GLFW_KEY_KP_DECIMAL: return ImGuiKey_KeypadDecimal;
			case GLFW_KEY_KP_DIVIDE: return ImGuiKey_KeypadDivide;
			case GLFW_KEY_KP_MULTIPLY: return ImGuiKey_KeypadMultiply;
			case GLFW_KEY_KP_SUBTRACT: return ImGuiKey_KeypadSubtract;
			case GLFW_KEY_KP_ADD: return ImGuiKey_KeypadAdd;
			case GLFW_KEY_KP_ENTER: return ImGuiKey_KeypadEnter;
			case GLFW_KEY_KP_EQUAL: return ImGuiKey_KeypadEqual;
			case GLFW_KEY_LEFT_SHIFT: return ImGuiKey_LeftShift;
			case GLFW_KEY_LEFT_CONTROL: return ImGuiKey_LeftCtrl;
			case GLFW_KEY_LEFT_ALT: return ImGuiKey_LeftAlt;
			case GLFW_KEY_LEFT_SUPER: return ImGuiKey_LeftSuper;
			case GLFW_KEY_RIGHT_SHIFT: return ImGuiKey_RightShift;
			case GLFW_KEY_RIGHT_CONTROL: return ImGuiKey_RightCtrl;
			case GLFW_KEY_RIGHT_ALT: return ImGuiKey_RightAlt;
			case GLFW_KEY_RIGHT_SUPER: return ImGuiKey_RightSuper;
			case GLFW_KEY_MENU: return ImGuiKey_Menu;
			case GLFW_KEY_0: return ImGuiKey_0;
			case GLFW_KEY_1: return ImGuiKey_1;
			case GLFW_KEY_2: return ImGuiKey_2;
			case GLFW_KEY_3: return ImGuiKey_3;
			case GLFW_KEY_4: return ImGuiKey_4;
			case GLFW_KEY_5: return ImGuiKey_5;
			case GLFW_KEY_6: return ImGuiKey_6;
			case GLFW_KEY_7: return ImGuiKey_7;
			case GLFW_KEY_8: return ImGuiKey_8;
			case GLFW_KEY_9: return ImGuiKey_9;
			case GLFW_KEY_A: return ImGuiKey_A;
			case GLFW_KEY_B: return ImGuiKey_B;
			case GLFW_KEY_C: return ImGuiKey_C;
			case GLFW_KEY_D: return ImGuiKey_D;
			case GLFW_KEY_E: return ImGuiKey_E;
			case GLFW_KEY_F: return ImGuiKey_F;
			case GLFW_KEY_G: return ImGuiKey_G;
			case GLFW_KEY_H: return ImGuiKey_H;
			case GLFW_KEY_I: return ImGuiKey_I;
			case GLFW_KEY_J: return ImGuiKey_J;
			case GLFW_KEY_K: return ImGuiKey_K;
			case GLFW_KEY_L: return ImGuiKey_L;
			case GLFW_KEY_M: return ImGuiKey_M;
			case GLFW_KEY_N: return ImGuiKey_N;
			case GLFW_KEY_O: return ImGuiKey_O;
			case GLFW_KEY_P: return ImGuiKey_P;
			case GLFW_KEY_Q: return ImGuiKey_Q;
			case GLFW_KEY_R: return ImGuiKey_R;
			case GLFW_KEY_S: return ImGuiKey_S;
			case GLFW_KEY_T: return ImGuiKey_T;
			case GLFW_KEY_U: return ImGuiKey_U;
			case GLFW_KEY_V: return ImGuiKey_V;
			case GLFW_KEY_W: return ImGuiKey_W;
			case GLFW_KEY_X: return ImGuiKey_X;
			case GLFW_KEY_Y: return ImGuiKey_Y;
			case GLFW_KEY_Z: return ImGuiKey_Z;
			case GLFW_KEY_F1: return ImGuiKey_F1;
			case GLFW_KEY_F2: return ImGuiKey_F2;
			case GLFW_KEY_F3: return ImGuiKey_F3;
			case GLFW_KEY_F4: return ImGuiKey_F4;
			case GLFW_KEY_F5: return ImGuiKey_F5;
			case GLFW_KEY_F6: return ImGuiKey_F6;
			case GLFW_KEY_F7: return ImGuiKey_F7;
			case GLFW_KEY_F8: return ImGuiKey_F8;
			case GLFW_KEY_F9: return ImGuiKey_F9;
			case GLFW_KEY_F10: return ImGuiKey_F10;
			case GLFW_KEY_F11: return ImGuiKey_F11;
			case GLFW_KEY_F12: return ImGuiKey_F12;
			default: return ImGuiKey_None;
		}
	}

	inline std::ostream& operator<<(std::ostream& os, KeyCode keyCode)
	{
		os << static_cast<int32_t>(keyCode);
		return os;
	}

	inline std::ostream& operator<<(std::ostream& os, MouseButton button)
	{
		os << static_cast<int32_t>(button);
		return os;
	}
}

