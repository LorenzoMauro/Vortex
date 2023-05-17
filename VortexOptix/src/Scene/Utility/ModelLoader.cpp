#include "ModelLoader.h"
#include "Scene/Nodes/Group.h"
#include "Scene/Nodes/Instance.h"
#include "Scene/Nodes/Shader/mdl/ShaderNodes.h"

namespace vtx::importer
{
    // Types sorted by preferences
    void getTexturePath(const aiMaterial* material, const std::vector<aiTextureType>& potentialTypes, std::string& returnPath, std::string& scenePath)
    {
        aiString path;
        for(const auto type : potentialTypes)
	    {
		    if(material->GetTexture(type, 0, &path) == AI_SUCCESS)
		    {
                returnPath = utl::absolutePath(path.C_Str(), scenePath);
		    	return;
		    }
	    }
    }

    struct GetValueInput {
        const char* pKey;
        int type;
        int index;
    };

    void getColorValue(const aiMaterial* material, const std::vector<GetValueInput>& potentialTypes, math::vec3f& color)
    {
	    aiColor4D aiColor;
		for(const auto type : potentialTypes)
		{
			if(material->Get(type.pKey, type.type, type.index, aiColor) == AI_SUCCESS)
			{
                color = math::vec3f(aiColor.r, aiColor.g, aiColor.b);
		    	return;
		    }
	    }
    }


    void getFloatValue(const aiMaterial* material, const std::vector<GetValueInput>& potentialTypes, float& color)
    {
        ai_real value;
        for (const auto type : potentialTypes)
        {
            if (material->Get(type.pKey, type.type, type.index, value) == AI_SUCCESS)
            {
                color = value;
                return;
            }
        }
    }

    std::shared_ptr<graph::shader::ImportedNode> createPrincipledMaterial(AssimpMaterialProperties properties)
    {
        auto principled = ops::createPbsdfGraph();

        // Diffuse
        if(!properties.diffuse.path.empty())
        {
            principled->setSocketDefault(DIFFUSE_TEXTURE_SOCKET, mdl::createTextureConstant(properties.diffuse.path));
        }
        if(properties.diffuse.value != math::vec3f(-1.0f))
        {
            principled->setSocketDefault(DIFFUSE_COLOR_SOCKET, mdl::createConstantColor(properties.diffuse.value));
        }

        // Ambient Occlusion
        if (!properties.ambientOcclusion.path.empty())
        {
            principled->setSocketDefault(AO_TEXTURE_SOCKET, mdl::createTextureConstant(properties.ambientOcclusion.path, mi::neuraylib::IType_texture::TS_2D, 1.0f));
            principled->setSocketDefault(AO_TO_DIFFUSE_SOCKET, mdl::createConstantFloat(0.5f));
        }

        // MetallNess
        if (!properties.metallic.path.empty())
        {
            principled->setSocketDefault(METALLIC_TEXTURE_SOCKET, mdl::createTextureConstant(properties.metallic.path, mi::neuraylib::IType_texture::TS_2D, 1.0f));
        	principled->setSocketDefault(METALLIC_TEXTURE_INFLUENCE_SOCKET, mdl::createConstantFloat(1.0f));
        }
        if (properties.metallic.value >= 0.0f)
        {
            principled->setSocketDefault(METALLIC_CONSTANT_SOCKET, mdl::createConstantFloat(properties.metallic.value));
        }

        // Roughness
        if (!properties.roughness.path.empty())
        {
            principled->setSocketDefault(ROUGHNESS_TEXTURE_SOCKET, mdl::createTextureConstant(properties.roughness.path, mi::neuraylib::IType_texture::TS_2D, 1.0f));
            principled->setSocketDefault(ROUGHNESS_TEXTURE_INFLUENCE_SOCKET, mdl::createConstantFloat(1.0f));
        }
        if (properties.roughness.value >= 0.0f)
        {
            principled->setSocketDefault(ROUGHNESS_CONSTANT_SOCKET, mdl::createConstantFloat(properties.roughness.value));
        }

        //Normal
        if (!properties.normal.path.empty())
        {
            principled->setSocketDefault(NORMALMAP_TEXTURE_SOCKET, mdl::createTextureConstant(properties.normal.path, mi::neuraylib::IType_texture::TS_2D, 1.0f));
            principled->setSocketDefault(NORMALMAP_FACTOR_SOCKET, mdl::createConstantFloat(0.7f));
        }
        if (properties.normal.value >= 0.0f)
        {
            principled->setSocketDefault(NORMALMAP_FACTOR_SOCKET, mdl::createConstantFloat(properties.normal.value));
        }

        //Bump
        if (!properties.bump.path.empty())
        {
            principled->setSocketDefault(BUMPMAP_TEXTURE_SOCKET, mdl::createTextureConstant(properties.bump.path, mi::neuraylib::IType_texture::TS_2D, 1.0f));
        }

        //Specular Level
        if (properties.specular.value >= 0.0f)
        {
            principled->setSocketDefault(SPECULAR_LEVEL_SOCKET, mdl::createConstantFloat(properties.specular.value));
        }

        return principled;
    }


    void AssimpMaterialProperties::determineProperties(const aiMaterial* material, std::string scenePath)
    {
		// Diffuse texture and color
        getTexturePath(material, { aiTextureType_DIFFUSE, aiTextureType_BASE_COLOR }, diffuse.path, scenePath);
        getColorValue(material, { { AI_MATKEY_COLOR_DIFFUSE}, {AI_MATKEY_COLOR_SPECULAR} }, diffuse.value);

        // Ambient Occlusion texture and color
        getTexturePath(material, { aiTextureType_AMBIENT_OCCLUSION }, ambientOcclusion.path, scenePath);

        // Roughness texture and value
        getTexturePath(material, { aiTextureType_DIFFUSE_ROUGHNESS, aiTextureType_SHININESS }, roughness.path, scenePath);
        getFloatValue(material, { { AI_MATKEY_ROUGHNESS_FACTOR} }, roughness.value); //, {AI_MATKEY_SHININESS}

        // Specular texture and value
        getTexturePath(material, { aiTextureType_SPECULAR }, specular.path, scenePath);
        getFloatValue(material, { {AI_MATKEY_GLOSSINESS_FACTOR}, {AI_MATKEY_SPECULAR_FACTOR} }, specular.value);
        if (specular.value == -1.0f)
        {
            math::vec3f shininessColor = -1.0f;
            getColorValue(material, { { AI_MATKEY_COLOR_SPECULAR} }, shininessColor);
            if (shininessColor != diffuse.value && shininessColor.x == shininessColor.y && shininessColor.x == shininessColor.z)
            {
                if (shininessColor.x != -1.0f)
                {
                    specular.value = shininessColor.x;
                }
            }
        }

        // Metallic texture and value
        getTexturePath(material, { aiTextureType_METALNESS }, metallic.path, scenePath);
        getFloatValue(material, { { AI_MATKEY_METALLIC_FACTOR}, { AI_MATKEY_REFLECTIVITY } }, metallic.value);

        // Normal texture
        getTexturePath(material, { aiTextureType_NORMAL_CAMERA, aiTextureType_NORMALS }, normal.path, scenePath);
        getFloatValue(material, { { AI_MATKEY_BUMPSCALING} }, normal.value);
        // Bump texture
        getTexturePath(material, { aiTextureType_HEIGHT }, bump.path, scenePath);

        //float glossinessFactor = -1.0f;
        //getFloatValue(material, { { AI_MATKEY_GLOSSINESS_FACTOR} }, glossinessFactor);
        //
        //math::vec3f shininessColor = -1.0f;
        //getColorValue(material, { { AI_MATKEY_COLOR_SPECULAR} }, shininessColor);
        //float specularFactor = -1.0f;
        //getFloatValue(material, { { AI_MATKEY_SPECULAR_FACTOR} }, specularFactor);
        //float reflectivity = -1.0f;
        //getFloatValue(material, { { AI_MATKEY_REFLECTIVITY} }, reflectivity);
        //float roughnessfactor = -1.0f;
        //getFloatValue(material, { { AI_MATKEY_ROUGHNESS_FACTOR} }, roughnessfactor);
        //float metallicFactor = -1.0f;
        //getFloatValue(material, { { AI_MATKEY_METALLIC_FACTOR} }, metallicFactor);

        //float shininessStrenght = -1.0f;
        //getFloatValue(material, { { AI_MATKEY_SHININESS_STRENGTH} }, shininessStrenght);
        //float shininess = -1.0f;
        //getFloatValue(material, { { AI_MATKEY_SHININESS} }, shininess);

    }
    // Process materials in the aiScene
    std::vector<std::shared_ptr<graph::Material>> processMaterials(const aiScene* scene, std::string scenePath)
    {
        std::vector<std::shared_ptr<graph::Material>> materials;

        for (unsigned int i = 0; i < scene->mNumMaterials; ++i)
        {
            aiMaterial* aiMat = scene->mMaterials[i];

            // Create a new material in your renderer and set its properties based on aiMat
            auto material = ops::createNode<graph::Material>();
            // Set material properties, e.g., diffuse color, specular color, textures, etc.
            // ...

            AssimpMaterialProperties properties;
            properties.determineProperties(aiMat, scenePath);
            const auto principled = createPrincipledMaterial(properties);
            // You can now use the extracted variables within the function
            material->materialGraph = principled;
            materials.push_back(material);
        }

        return materials;
    }

    void convertAssimpMeshToMeshNode(aiMesh* aiMesh, std::shared_ptr<graph::Mesh> meshNode)
    {
        // Process vertices
        meshNode->vertices.resize(aiMesh->mNumVertices);
        meshNode->status.hasNormals = false;
        meshNode->status.hasTangents = true;
        meshNode->status.hasFaceAttributes = true;
        for (unsigned int i = 0; i < aiMesh->mNumVertices; ++i)
        {
            auto& vertex = meshNode->vertices[i];
            // Positions
            vertex.position = math::vec3f(aiMesh->mVertices[i].x, aiMesh->mVertices[i].y, aiMesh->mVertices[i].z);

            // Normals
            if (aiMesh->HasNormals())
            {
                vertex.normal = math::vec3f(aiMesh->mNormals[i].x, aiMesh->mNormals[i].y, aiMesh->mNormals[i].z);
            }
            else
            {
                meshNode->status.hasNormals = false;
            }

            // Tangents
            if (aiMesh->HasTangentsAndBitangents())
            {
                vertex.tangent = math::vec3f(aiMesh->mTangents[i].x, aiMesh->mTangents[i].y, aiMesh->mTangents[i].z);
            }
            else
            {
	            meshNode->status.hasTangents = false;
            }


            // Texture coordinates
            if (aiMesh->HasTextureCoords(0))
            {
                vertex.texCoord = math::vec3f(aiMesh->mTextureCoords[0][i].x, aiMesh->mTextureCoords[0][i].y, 0.0f);
            }
        }

        // Process faces and indices
        meshNode->faceAttributes.resize(aiMesh->mNumFaces);
        for (unsigned int i = 0; i < aiMesh->mNumFaces; ++i)
        {
            const aiFace& face = aiMesh->mFaces[i];

            // Store indices for the face
            for (unsigned int j = 0; j < face.mNumIndices; ++j)
            {
                meshNode->indices.push_back(face.mIndices[j]);
            }

            // Set face attributes (you can update it later based on the material, if needed)
            graph::FaceAttributes faceAttr;
            faceAttr.materialSlotId = 0;
            meshNode->faceAttributes[i] = faceAttr;
        }
    }

    math::affine3f convertAssimpMatrix(const aiMatrix4x4& aiMatrix)
    {
        math::affine3f matrix;
        matrix.l = math::LinearSpace3f(aiMatrix.a1, aiMatrix.a2, aiMatrix.a3,
                                       aiMatrix.b1, aiMatrix.b2, aiMatrix.b3,
                                       aiMatrix.c1, aiMatrix.c2, aiMatrix.c3);
        matrix.p = math::vec3f(aiMatrix.a4, aiMatrix.b4, aiMatrix.c4);

        return matrix;
    }

    std::shared_ptr<graph::Instance> processAssimpNode(aiMesh* node, const unsigned assimpMeshId, std::map<unsigned, vtxID>& meshMap, const std::vector<std::shared_ptr<graph::Material>>& importedMaterials)
    {
        std::shared_ptr<graph::Mesh> meshNode;
        if (meshMap.find(assimpMeshId) != meshMap.end())
        {
            meshNode = graph::SIM::getNode<graph::Mesh>(meshMap[assimpMeshId]);
        }
        else
        {
            meshNode = ops::createNode<graph::Mesh>();
            meshMap.insert({ assimpMeshId, meshNode->getID() });
            convertAssimpMeshToMeshNode(node, meshNode);
        }

        std::shared_ptr<graph::Instance> instanceNode = ops::createNode<graph::Instance>();
        // Set the meshNode as a child of the instanceNode
        instanceNode->setChild(meshNode);
        instanceNode->addMaterial(importedMaterials[node->mMaterialIndex]);
        return instanceNode;
    }

    std::shared_ptr<graph::Group> processAssimpNode(const aiNode* node, const aiScene* scene, std::map<unsigned, vtxID>& meshMap, const std::vector<std::shared_ptr<graph::Material>>& importedMaterials) {
        auto groupNode = ops::createNode<graph::Group>();

        // Process node transformation
        groupNode->transform->setAffine(convertAssimpMatrix(node->mTransformation));

        // Process node children
        for (unsigned int i = 0; i < node->mNumChildren; ++i) {
            groupNode->addChild(processAssimpNode(node->mChildren[i], scene, meshMap, importedMaterials));
        }

        // Process node meshes
        for (unsigned int i = 0; i < node->mNumMeshes; ++i) {
            aiMesh* aiMesh = scene->mMeshes[node->mMeshes[i]];
            groupNode->addChild(processAssimpNode(aiMesh, node->mMeshes[i], meshMap, importedMaterials));
        }

        return groupNode;
    }

    void processMetadata(const aiScene* scene) {
        if (scene->mMetaData)
        { 
           /* int32_t frontAxis = 0; 
            int32_t frontAxisSign = 1; 
            int32_t coordAxis = 1; 
            int32_t coordAxisSign = 1;
            int32_t upAxis = 2;
            int32_t upAxisSign = 1;
            double unitScaleFactor = 0.01;*/
            int32_t frontAxis = 0;
            int32_t frontAxisSign = 1;
            int32_t coordAxis = 2;  // Changed from 1 to 2
            int32_t coordAxisSign = 1;
            int32_t upAxis = 1;  // Changed from 2 to 1
            int32_t upAxisSign = 1;
            double unitScaleFactor = 0.01;
            for (unsigned MetadataIndex = 0; MetadataIndex < scene->mMetaData->mNumProperties; ++MetadataIndex)
            {
                if (strcmp(scene->mMetaData->mKeys[MetadataIndex].C_Str(), "UpAxis") == 0)
                {
                    scene->mMetaData->Get<int32_t>(MetadataIndex, upAxis);
                }
                if (strcmp(scene->mMetaData->mKeys[MetadataIndex].C_Str(), "UpAxisSign") == 0)
                {
                    scene->mMetaData->Get<int32_t>(MetadataIndex, upAxisSign);
                }
                if (strcmp(scene->mMetaData->mKeys[MetadataIndex].C_Str(), "FrontAxis") == 0)
                {
                    scene->mMetaData->Get<int32_t>(MetadataIndex, frontAxis);
                }
                if (strcmp(scene->mMetaData->mKeys[MetadataIndex].C_Str(), "FrontAxisSign") == 0)
                {
                    scene->mMetaData->Get<int32_t>(MetadataIndex, frontAxisSign);
                }
                if (strcmp(scene->mMetaData->mKeys[MetadataIndex].C_Str(), "CoordAxis") == 0)
                {
                    scene->mMetaData->Get<int32_t>(MetadataIndex, coordAxis);
                }
                if (strcmp(scene->mMetaData->mKeys[MetadataIndex].C_Str(), "CoordAxisSign") == 0)
                {
                    scene->mMetaData->Get<int32_t>(MetadataIndex, coordAxisSign);
                }
                if (strcmp(scene->mMetaData->mKeys[MetadataIndex].C_Str(), "UnitScaleFactor") == 0)
                {
                    scene->mMetaData->Get<double>(MetadataIndex, unitScaleFactor);
                }
            }

            aiVector3D upVec; 
            aiVector3D forwardVec; 
            aiVector3D rightVec;

            upVec[upAxis]           = upAxisSign * (float)unitScaleFactor;
            forwardVec[frontAxis]   = frontAxisSign * (float)unitScaleFactor;
            rightVec[coordAxis]     = coordAxisSign * (float)unitScaleFactor;

			aiMatrix4x4 mat(forwardVec.x,   forwardVec.y,   forwardVec.z,   0.0f,
							rightVec.x,     rightVec.y,     rightVec.z,     0.0f,
							upVec.x,        upVec.y,        upVec.z,        0.0f,
							0.0f,           0.0f,           0.0f,           1.0f);

            scene->mRootNode->mTransformation = mat;
        }
    }

    std::shared_ptr<graph::Group> importSceneFile(std::string filePath)
    {
        filePath = utl::absolutePath(filePath);
        VTX_INFO("Loading scene file: {}", filePath);
        Assimp::Importer importer;
        const aiScene* scene = importer.ReadFile(filePath,
                                                 aiProcess_Triangulate |
                                                 aiProcess_JoinIdenticalVertices |
                                                 aiProcess_SortByPType |
                                                 aiProcess_GenSmoothNormals |
                                                 //aiProcess_CalcTangentSpace |
                                                 //aiProcess_RemoveComponent (remove colors) |
                                                 //aiProcess_LimitBoneWeights |
                                                 aiProcess_ImproveCacheLocality |
                                                 aiProcess_RemoveRedundantMaterials |
                                                 //aiProcess_GenUVCoords |
                                                 aiProcess_FindDegenerates |
                                                 aiProcess_FindInvalidData |
                                                 aiProcess_FindInstances |
                                                 //aiProcess_ValidateDataStructure |
                                                 aiProcess_OptimizeMeshes |
                                                 //aiProcess_OptimizeGraph |
                                                 //aiProcess_Debone |
                                                 0);

        const bool successCondition = (scene && !(scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE) && scene->mRootNode);
        VTX_ASSERT_CONTINUE(successCondition, "Assimp Importer Errror: {}", importer.GetErrorString());

        processMetadata(scene);
        std::map<unsigned, vtxID> meshMap;
        VTX_INFO("Creating Scene Graph");
		const std::vector<std::shared_ptr<graph::Material>> importedMaterials = processMaterials(scene, utl::getFolder(filePath));
        return processAssimpNode(scene->mRootNode, scene, meshMap, importedMaterials);

    }


}
