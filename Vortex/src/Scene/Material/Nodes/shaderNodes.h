#pragma once
#include "mi/mdl_sdk.h"
#include "Scene/Material/mdlTools.h"
#include "Scene/Node.h"

namespace vtx::graph
{
	using namespace mi;
	using namespace base;
	using namespace neuraylib;

	class ImageNode : public Node {
	public:
		ImageNode() : Node(NT_SHADER) {
			name = "Image." + std::to_string(getUID());
		}
            
		ImageNode(std::string _imagePath) : Node(NT_SHADER) {
			name = "Image." + std::to_string(getUID());
			SetImageFile(_imagePath);
		}

		void SetImageFile(std::string _imagePath) {
			path = _imagePath;

			ITransaction* transaction = mdl::getGlobalTransaction();
			image = transaction->create<IImage>(name.c_str());
			result = image->reset_file(path.c_str());
			VTX_ASSERT_RETURN(result == 0, "Error with creating Image node of image {}", path);
			transaction->store(image.get(), name.c_str());
		}

	public:
		Handle<IImage> image;
		std::string path;
		std::string name;
		Sint32 result;
	};

	class TextureNode : public Node
	{
		TextureNode() : Node(NT_SHADER) {
			name = "Texture." + std::to_string(getUID());
		};

		TextureNode(std::shared_ptr<ImageNode> _image) : Node(NT_SHADER) {
			name = "Texture." + std::to_string(getUID());
		};

		void SetImageNode(std::shared_ptr<ImageNode> _image) {
			image = _image;
			ITransaction* transaction = mdl::getGlobalTransaction();
			texture = transaction->create<ITexture>(name.c_str());
			texture->set_image(image->name.c_str());
			texture->set_gamma(2.2f);
			transaction->store(texture.get(), name.c_str());
		}

	public:
		std::shared_ptr<ImageNode> image;
		Handle<ITexture> texture;
		std::string name;
	};

}
