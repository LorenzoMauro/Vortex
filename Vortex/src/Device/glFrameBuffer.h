#pragma once
#include <glad/glad.h>
#include <cstdint>
#include "Core/Options.h"

namespace vtx {

	class GlFrameBuffer
	{
	public:
		GlFrameBuffer();

		void setSize(const uint32_t widthParam, const uint32_t heightParam, const bool forceRegenerate = false);
		void generate();

		void bind();

		void unbind();

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
