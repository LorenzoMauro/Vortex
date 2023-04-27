#pragma once
#include <glad/glad.h>
#include <cstdint>
#include "Core/Log.h"
#include "Core/Options.h"

namespace vtx {
	class GlFrameBuffer
	{
	public:
		GlFrameBuffer() :
			m_Height(getOptions()->height),
			m_Width(getOptions()->width)
		{
			generate();
		}

		void SetSize(uint32_t width, uint32_t height) {
			if (width != m_Width || height != m_Height) {
				m_Width = width;
				m_Height = height;
				generate();
			}
		}
		void generate() {
			glCreateFramebuffers(1, &m_RendererID);
			glBindFramebuffer(GL_FRAMEBUFFER, m_RendererID);

			glCreateTextures(GL_TEXTURE_2D, 1, &m_ColorAttachment);
			glBindTexture(GL_TEXTURE_2D, m_ColorAttachment);
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, m_Width, m_Height, 0, GL_RGBA, GL_FLOAT, nullptr);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

			glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_ColorAttachment, 0);

			glCreateTextures(GL_TEXTURE_2D, 1, &m_DepthAttachment);
			glBindTexture(GL_TEXTURE_2D, m_DepthAttachment);

			glTexStorage2D(GL_TEXTURE_2D, 1, GL_DEPTH24_STENCIL8, m_Width, m_Height);
			glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D, m_DepthAttachment, 0);

			glCheckFramebufferStatus(GL_FRAMEBUFFER == GL_FRAMEBUFFER_COMPLETE);

		}

		void Bind() {
			glBindFramebuffer(GL_FRAMEBUFFER, m_RendererID);
		}

		void Unbind() {
			glBindFramebuffer(GL_FRAMEBUFFER, 0);
		}

		GLuint GetColorAttachment() {
			return m_ColorAttachment;
		}

		void UploadExternalColorAttachment(void* data) {
			glBindTexture(GL_TEXTURE_2D, m_ColorAttachment);
			glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_Width, m_Height, GL_RGBA, GL_UNSIGNED_BYTE, data);
		}

		void CheckFrameBufferStatus()
		{
			GLenum status;
			status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
			switch (status) {
			case GL_FRAMEBUFFER_COMPLETE:
				break;

			case GL_FRAMEBUFFER_UNSUPPORTED:
				/* choose different formats */
				break;

			default:
				VTX_ERROR("Framebuffer Error");
				exit(-1);
			}
		}


	public:
		GLuint			m_RendererID;
		GLuint			m_DepthAttachment;
		GLuint			m_StencilAttachment;
		GLuint			m_ColorAttachment;
		uint32_t		m_Height;
		uint32_t		m_Width;
	};

}
