#pragma once
#include <map>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include "Core/Math.h"
#include "Core/Utils.h"
#include "Core/VortexID.h"


namespace vtx
{
	namespace graph
	{
		class Group;
		class Instance;
		class Mesh;
		class Material;
	}
}

namespace vtx::importer
{
    template<typename T>
    struct TextureAndValue
    {
	    std::string path;
		T value;
    };

    struct AssimpMaterialProperties {
        TextureAndValue<math::vec3f> diffuse{"", {-1.f,-1.f,-1.f,}};
        TextureAndValue<float> ambientOcclusion{"", -1};
        TextureAndValue<float> roughness{ "", -1 };
        TextureAndValue<float> specular{ "", -1 };
        TextureAndValue<float> metallic{ "", -1 };
        TextureAndValue<float> normal{ "", -1 };
        TextureAndValue<float> bump{ "", -1 };

        void determineProperties(const aiMaterial* material, std::string scenePath);
    };

    // Process materials in the aiScene
	std::vector<std::shared_ptr<graph::Material>> processMaterials(const aiScene* scene, std::string scenePath);

    void convertAssimpMeshToMeshNode(aiMesh* aiMesh, std::shared_ptr<graph::Mesh> meshNode);

    math::affine3f convertAssimpMatrix(const aiMatrix4x4& aiMatrix);

    std::shared_ptr<graph::Instance> processAssimpNode(aiMesh* node, const unsigned assimpMeshId, std::map<unsigned, vtxID>& meshMap, const std::vector<std::shared_ptr<graph::Material>>& importedMaterials);

	std::shared_ptr<graph::Group> processAssimpNode(const aiNode* node, const aiScene* scene, std::map<unsigned, vtxID>& meshMap, const std::vector<std::shared_ptr<graph::Material>>& importedMaterials);

    std::shared_ptr<graph::Group> importSceneFile(std::string filePath);
}
