#pragma once
#include "glad/glad.h"
#include "glFrameBuffer.h"
#include <cstdint>

namespace vtx {
	class Renderer {
	public:

		void Resize(uint32_t width, uint32_t height) {
			m_framebuffer.SetSize(width, height);
		}

		GLuint GetFrame() {
			m_framebuffer.Bind();
			glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
			glClear(GL_COLOR_BUFFER_BIT);
			m_framebuffer.Unbind();
			return m_framebuffer.GetColorAttachment();
		};
	public:
		glFrameBuffer					m_framebuffer;
	};
}