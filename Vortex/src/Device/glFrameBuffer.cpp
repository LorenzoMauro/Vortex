#include "glFrameBuffer.h"
#include "Core/Log.h"
#include "glad/glad.h"

namespace vtx
{
	GlFrameBuffer::GlFrameBuffer() :
		height(getOptions()->height),
		width(getOptions()->width)
	{
		generate();
		
	}

	void GlFrameBuffer::setSize(const uint32_t widthParam, const uint32_t heightParam) {
		if (width != widthParam || height != heightParam) {
			width = widthParam;
			height = heightParam;
			generate();
		}
	}

	void GlFrameBuffer::generate() {
		glCreateFramebuffers(1, &frameBufferId);
		glBindFramebuffer(GL_FRAMEBUFFER, frameBufferId);

		glCreateTextures(GL_TEXTURE_2D, 1, &colorAttachment);
		glBindTexture(GL_TEXTURE_2D, colorAttachment);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, colorAttachment, 0);

		glCreateTextures(GL_TEXTURE_2D, 1, &depthAttachment);
		glBindTexture(GL_TEXTURE_2D, depthAttachment);

		glTexStorage2D(GL_TEXTURE_2D, 1, GL_DEPTH24_STENCIL8, width, height);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D, depthAttachment, 0);

		glCheckFramebufferStatus(GL_FRAMEBUFFER == GL_FRAMEBUFFER_COMPLETE);

	}

	void GlFrameBuffer::bind() {
		glBindFramebuffer(GL_FRAMEBUFFER, frameBufferId);
	}

	void GlFrameBuffer::unbind() {
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}

	GLuint GlFrameBuffer::getColorAttachment() const
	{
		return colorAttachment;
	}

	void GlFrameBuffer::uploadExternalColorAttachment(const void* data) const
	{
		glBindTexture(GL_TEXTURE_2D, colorAttachment);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, data);
	}

	void GlFrameBuffer::copyToThis(const GlFrameBuffer& src)
	{
		setSize(src.width, src.height);
		glBindFramebuffer(GL_READ_FRAMEBUFFER, src.frameBufferId); // Source framebuffer
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, frameBufferId); // Destination framebuffer
		glBlitFramebuffer(0, 0, width, height, 0, 0, src.width, src.height, GL_COLOR_BUFFER_BIT, GL_NEAREST);
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}

	void GlFrameBuffer::checkFrameBufferStatus()
	{
		switch (GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER)) {
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
}
