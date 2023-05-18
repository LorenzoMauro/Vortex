#pragma once
#include <glad/glad.h>
#include <cstdint>
#include "Core/Options.h"

namespace vtx {

	class GlFrameBuffer
	{
	public:
		GlFrameBuffer();

		void setSize(uint32_t widthParam, uint32_t heightParam);
		void generate();

		void bind();

		static void unbind();

		GLuint getColorAttachment() const;

		void uploadExternalColorAttachment(const void* data) const;

		void copyToThis(const GlFrameBuffer& src);

		static void checkFrameBufferStatus();


	public:
		GLuint			frameBufferId;
		GLuint			depthAttachment;
		GLuint			stencilAttachment;
		GLuint			colorAttachment;
		uint32_t		height;
		uint32_t		width;
	};

}
