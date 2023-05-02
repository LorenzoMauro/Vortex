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
    // Process materials in the aiScene
    std::vector<std::shared_ptr<graph::Material>> processMaterials(const aiScene* scene);

    void convertAssimpMeshToMeshNode(aiMesh* aiMesh, std::shared_ptr<graph::Mesh> meshNode);

    math::affine3f convertAssimpMatrix(const aiMatrix4x4& aiMatrix);

    std::shared_ptr<graph::Instance> processAssimpNode(aiMesh* node, const unsigned assimpMeshId, std::map<unsigned, vtxID>& meshMap);;

    std::shared_ptr<graph::Group> processAssimpNode(const aiNode* node, const aiScene* scene, std::map<unsigned, vtxID>& meshMap);

    std::shared_ptr<graph::Group> importSceneFile(std::string filePath);
}
