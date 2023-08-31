﻿#pragma once
#include "Scene/Node.h"
#include <mi/mdl_sdk.h>
#include "Core/Math.h"
#include "Device/UploadCode/CUDABuffer.h"

namespace vtx::graph
{
	using namespace mi;
	using namespace base;
	using namespace neuraylib;

	class Texture : public Node
	{
	public:
		Texture() :
			Node(NT_MDL_TEXTURE),
			shape(ITarget_code::Texture_shape_invalid)
		{
		}

		Texture(const char* _databaseName, ITarget_code::Texture_shape _shape):
			Node(NT_MDL_TEXTURE),
			databaseName(_databaseName),
			shape(_shape)
		{
		}

		Texture(std::string filePath) :
			Node(NT_MDL_TEXTURE)
		{
			this->filePath = filePath;
			loadFromFile = true;
		}

		~Texture() override
		{
			for (const auto& ptr : imageLayersPointers)
			{
				free(const_cast<void*>(ptr));
			}
		}

		void init() override;

	protected:
		void accept(NodeVisitor& visitor) override;
	public:
		std::string										filePath; // In case the image is loaded from a file
		std::string										databaseName; // In case the image is loaded from the database
		ITarget_code::Texture_shape						shape;

		size_t											pixelBytesSize;
		std::vector<const void*>						imageLayersPointers;
		math::vec4ui									dimension;
		CUarray_format_enum								format;
		float											effectiveGamma;

		bool											isInitialized = false;
		bool											loadFromFile = false;
		Size mdlIndex;
	};
}
