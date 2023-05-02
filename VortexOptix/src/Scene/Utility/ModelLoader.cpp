#include "ModelLoader.h"
#include "Scene/Nodes/Group.h"
#include "Scene/Nodes/Instance.h"

namespace vtx::importer
{
    // Process materials in the aiScene
    std::vector<std::shared_ptr<graph::Material>> processMaterials(const aiScene* scene)
    {
        std::vector<std::shared_ptr<graph::Material>> materials;

        for (unsigned int i = 0; i < scene->mNumMaterials; ++i)
        {
            aiMaterial* aiMat = scene->mMaterials[i];

            // Create a new material in your renderer and set its properties based on aiMat
            auto material = std::make_shared<graph::Material>();
            // Set material properties, e.g., diffuse color, specular color, textures, etc.
            // ...

            materials.push_back(material);
        }

        return materials;
    }

    void convertAssimpMeshToMeshNode(aiMesh* aiMesh, std::shared_ptr<graph::Mesh> meshNode)
    {
        // Process vertices
        meshNode->vertices.resize(aiMesh->mNumVertices);
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

            // Tangents
            if (aiMesh->HasTangentsAndBitangents())
            {
                meshNode->status.hasTangents = true;
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

    std::shared_ptr<graph::Instance> processAssimpNode(aiMesh* node, const unsigned assimpMeshId, std::map<unsigned, vtxID>& meshMap)
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
        return instanceNode;
    }

    std::shared_ptr<graph::Group> processAssimpNode(const aiNode* node, const aiScene* scene, std::map<unsigned, vtxID>& meshMap) {
        auto groupNode = ops::createNode<graph::Group>();

        // Process node transformation
        groupNode->transform->setAffine(convertAssimpMatrix(node->mTransformation));

        // Process node children
        for (unsigned int i = 0; i < node->mNumChildren; ++i) {
            groupNode->addChild(processAssimpNode(node->mChildren[i], scene, meshMap));
        }

        // Process node meshes
        for (unsigned int i = 0; i < node->mNumMeshes; ++i) {
            aiMesh* aiMesh = scene->mMeshes[node->mMeshes[i]];
            groupNode->addChild(processAssimpNode(aiMesh, node->mMeshes[i], meshMap));
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
                                                 aiProcess_Triangulate | aiProcess_JoinIdenticalVertices | aiProcess_SortByPType);

        const bool successCondition = (scene && !(scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE) && scene->mRootNode);
        VTX_ASSERT_CONTINUE(successCondition, "Assimp Importer Errror: {}", importer.GetErrorString());

        processMetadata(scene);
        std::map<unsigned, vtxID> meshMap;
        VTX_INFO("Creating Scene Graph");
        return processAssimpNode(scene->mRootNode, scene, meshMap);

    }

}
