﻿#pragma once
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
		class Node;
		class Camera;
		class Group;
		class Instance;
		class Mesh;
		class Material;
	}
}

namespace vtx::importer
{

	enum class SwapType
	{
		None,
		yToZ
	};

    template<typename T>
    struct TextureAndValue
    {
	    std::string path;
		T value;
    };

    struct AssimpMaterialProperties {
		TextureAndValue<math::vec3f> diffuse{"", {-1.f, -1.f, -1.f,}};
		TextureAndValue<float>       ambientOcclusion{"", -1};
		TextureAndValue<float>       roughness{"", -1};
		TextureAndValue<float>       specular{"", -1};
		TextureAndValue<float>       metallic{"", -1};
		TextureAndValue<float>       normal{"", -1};
		TextureAndValue<float>       bump{"", -1};
		TextureAndValue<math::vec3f> emissionColor{ "", {-1.f, -1.f, -1.f,} };
		TextureAndValue<float>       emissionIntensity{ "", -1 };
		TextureAndValue<float>       clearcoatAmount{ "", -1 };
		TextureAndValue<float>       clearcoatRoughness{ "", -1 };
		TextureAndValue<float>       clearcoatNormal{ "", -1 };
		TextureAndValue<float>       transmission{ "", -1 };
		TextureAndValue<math::vec3f> sheenColor{ "", {-1.f, -1.f, -1.f,} };
		TextureAndValue<float>       sheenRoughness{ "", -1 };
		TextureAndValue<float>       anisotropy{ "", -1 };
		std::string                  name;
		TextureAndValue<math::vec3f> ORM{ "", {-1.f, -1.f, -1.f,} };
		TextureAndValue<float>       opacity{ "", -1 };

		void determineProperties(const aiMaterial* material, std::string scenePath);
    };

    // Process materials in the aiScene
	std::vector<std::shared_ptr<graph::Material>> processMaterials(const aiScene* scene, std::string scenePath);

	std::shared_ptr<graph::Mesh> convertAssimpMeshToMeshNode(const aiMesh* aiMesh, SwapType swap, float scaleFactor);

    math::affine3f convertAssimpMatrix(const aiMatrix4x4& aiMatrix, SwapType swap, float scaleFactor);

    std::shared_ptr<graph::Instance> processAssimpNode(aiMesh* node, const unsigned assimpMeshId, std::map<unsigned, vtxID>& meshMap, const std::vector<std::shared_ptr<graph::Material>>& importedMaterials, SwapType swap, float scaleFactor);

	std::shared_ptr<graph::Node> processAssimpNode(const aiNode* node, const aiScene* scene, std::map<unsigned, vtxID>& meshMap, const std::vector<std::shared_ptr<graph::Material>>& importedMaterials, SwapType swap, float scaleFactor);

	std::tuple<std::shared_ptr<graph::Group>, std::vector<std::shared_ptr<graph::Camera>>> importSceneFile(std::string filePath);
}
