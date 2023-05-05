#include "Operations.h"

#include "ModelLoader.h"
#include "MDL/materialEditor.h"
#include "Scene/Graph.h"

namespace vtx::ops
{
    using namespace graph;

    // A simple unit cube built from 12 triangles.
    std::shared_ptr<Mesh> createBox()
    {
        VTX_INFO("Creating Box");
        std::shared_ptr<Mesh> mesh = createNode<Mesh>();

        const float left = -1.0f;
        const float right = 1.0f;
        const float bottom = -1.0f;
        const float top = 1.0f;
        const float back = -1.0f;
        const float front = 1.0f;

        VertexAttributes attrib;

        // Left.
        attrib.tangent = math::vec3f(0.0f, 0.0f, 1.0f);
        attrib.normal = math::vec3f(-1.0f, 0.0f, 0.0f);

        attrib.position = math::vec3f(left, bottom, back);
        attrib.texCoord = math::vec3f(0.0f, 0.0f, 0.0f);
        mesh->vertices.push_back(attrib);

        attrib.position = math::vec3f(left, bottom, front);
        attrib.texCoord = math::vec3f(1.0f, 0.0f, 0.0f);
        mesh->vertices.push_back(attrib);

        attrib.position = math::vec3f(left, top, front);
        attrib.texCoord = math::vec3f(1.0f, 1.0f, 0.0f);
        mesh->vertices.push_back(attrib);

        attrib.position = math::vec3f(left, top, back);
        attrib.texCoord = math::vec3f(0.0f, 1.0f, 0.0f);
        mesh->vertices.push_back(attrib);

        // Right.
        attrib.tangent = math::vec3f(0.0f, 0.0f, -1.0f);
        attrib.normal = math::vec3f(1.0f, 0.0f, 0.0f);

        attrib.position = math::vec3f(right, bottom, front);
        attrib.texCoord = math::vec3f(0.0f, 0.0f, 0.0f);
        mesh->vertices.push_back(attrib);

        attrib.position = math::vec3f(right, bottom, back);
        attrib.texCoord = math::vec3f(1.0f, 0.0f, 0.0f);
        mesh->vertices.push_back(attrib);

        attrib.position = math::vec3f(right, top, back);
        attrib.texCoord = math::vec3f(1.0f, 1.0f, 0.0f);
        mesh->vertices.push_back(attrib);

        attrib.position = math::vec3f(right, top, front);
        attrib.texCoord = math::vec3f(0.0f, 1.0f, 0.0f);
        mesh->vertices.push_back(attrib);

        // Back.  
        attrib.tangent = math::vec3f(-1.0f, 0.0f, 0.0f);
        attrib.normal = math::vec3f(0.0f, 0.0f, -1.0f);

        attrib.position = math::vec3f(right, bottom, back);
        attrib.texCoord = math::vec3f(0.0f, 0.0f, 0.0f);
        mesh->vertices.push_back(attrib);

        attrib.position = math::vec3f(left, bottom, back);
        attrib.texCoord = math::vec3f(1.0f, 0.0f, 0.0f);
        mesh->vertices.push_back(attrib);

        attrib.position = math::vec3f(left, top, back);
        attrib.texCoord = math::vec3f(1.0f, 1.0f, 0.0f);
        mesh->vertices.push_back(attrib);

        attrib.position = math::vec3f(right, top, back);
        attrib.texCoord = math::vec3f(0.0f, 1.0f, 0.0f);
        mesh->vertices.push_back(attrib);

        // Front.
        attrib.tangent = math::vec3f(1.0f, 0.0f, 0.0f);
        attrib.normal = math::vec3f(0.0f, 0.0f, 1.0f);

        attrib.position = math::vec3f(left, bottom, front);
        attrib.texCoord = math::vec3f(0.0f, 0.0f, 0.0f);
        mesh->vertices.push_back(attrib);

        attrib.position = math::vec3f(right, bottom, front);
        attrib.texCoord = math::vec3f(1.0f, 0.0f, 0.0f);
        mesh->vertices.push_back(attrib);

        attrib.position = math::vec3f(right, top, front);
        attrib.texCoord = math::vec3f(1.0f, 1.0f, 0.0f);
        mesh->vertices.push_back(attrib);

        attrib.position = math::vec3f(left, top, front);
        attrib.texCoord = math::vec3f(0.0f, 1.0f, 0.0f);
        mesh->vertices.push_back(attrib);

        // Bottom.
        attrib.tangent = math::vec3f(1.0f, 0.0f, 0.0f);
        attrib.normal = math::vec3f(0.0f, -1.0f, 0.0f);

        attrib.position = math::vec3f(left, bottom, back);
        attrib.texCoord = math::vec3f(0.0f, 0.0f, 0.0f);
        mesh->vertices.push_back(attrib);

        attrib.position = math::vec3f(right, bottom, back);
        attrib.texCoord = math::vec3f(1.0f, 0.0f, 0.0f);
        mesh->vertices.push_back(attrib);

        attrib.position = math::vec3f(right, bottom, front);
        attrib.texCoord = math::vec3f(1.0f, 1.0f, 0.0f);
        mesh->vertices.push_back(attrib);

        attrib.position = math::vec3f(left, bottom, front);
        attrib.texCoord = math::vec3f(0.0f, 1.0f, 0.0f);
        mesh->vertices.push_back(attrib);

        // Top.
        attrib.tangent = math::vec3f(1.0f, 0.0f, 0.0f);
        attrib.normal = math::vec3f(0.0f, 1.0f, 0.0f);

        attrib.position = math::vec3f(left, top, front);
        attrib.texCoord = math::vec3f(0.0f, 0.0f, 0.0f);
        mesh->vertices.push_back(attrib);

        attrib.position = math::vec3f(right, top, front);
        attrib.texCoord = math::vec3f(1.0f, 0.0f, 0.0f);
        mesh->vertices.push_back(attrib);

        attrib.position = math::vec3f(right, top, back);
        attrib.texCoord = math::vec3f(1.0f, 1.0f, 0.0f);
        mesh->vertices.push_back(attrib);

        attrib.position = math::vec3f(left, top, back);
        attrib.texCoord = math::vec3f(0.0f, 1.0f, 0.0f);
        mesh->vertices.push_back(attrib);

        FaceAttributes faceAttrib;
        for (unsigned int i = 0; i < 6; ++i)
        {
            const unsigned int idx = i * 4; // Four Mesh->vertices per box face.

            mesh->indices.push_back(idx);
            mesh->indices.push_back(idx + 1);
            mesh->indices.push_back(idx + 2);
            mesh->faceAttributes.push_back(faceAttrib);

            mesh->indices.push_back(idx + 2);
            mesh->indices.push_back(idx + 3);
            mesh->indices.push_back(idx);
            mesh->faceAttributes.push_back(faceAttrib);
        }
        return mesh;
    }

    void applyTransformation(graph::TransformAttribute& transformation, const math::affine3f& affine) {
        transformation.affineTransform = transformation.affineTransform * affine;
        transformation.updateFromAffine();
    }

    std::shared_ptr<graph::Mesh> createPlane()
    {
        VTX_INFO("Creating Plane");
        std::shared_ptr<graph::Mesh> mesh = createNode<graph::Mesh>();

        mesh->vertices.clear();
        mesh->indices.clear();

        math::vec3f corner;

        VertexAttributes attrib;

        // Positive z-axis is the geometry normal, create geometry on the xy-plane.
        corner = math::vec3f(-1.0f, -1.0f, 0.0f); // Lower left corner of the plane. texcoord (0.0f, 0.0f).

        attrib.tangent = math::vec3f(1.0f, 0.0f, 0.0f);
        attrib.normal = math::vec3f(0.0f, 0.0f, 1.0f);

        attrib.position = corner + math::vec3f(0.0f, 0.0f, 0.0f);
        attrib.texCoord = math::vec3f(0.0f, 0.0f, 0.0f);
        mesh->vertices.push_back(attrib);

        attrib.position = corner + math::vec3f(2.0f, 0.0f, 0.0f);
        attrib.texCoord = math::vec3f(1.0f, 0.0f, 0.0f);
        mesh->vertices.push_back(attrib);

        attrib.position = corner + math::vec3f(0.0f, 2.0f, 0.0f);
        attrib.texCoord = math::vec3f(0.0f, 1.0f, 0.0f);
        mesh->vertices.push_back(attrib);

        attrib.position = corner + math::vec3f(2.0f, 2.0f, 0.0f);
        attrib.texCoord = math::vec3f(1.0f, 1.0f, 0.0f);
        mesh->vertices.push_back(attrib);

        FaceAttributes faceAttrib;
        mesh->indices.push_back(0);
        mesh->indices.push_back(1);
        mesh->indices.push_back(3);
        mesh->faceAttributes.push_back(faceAttrib);

        mesh->indices.push_back(3);
        mesh->indices.push_back(2);
        mesh->indices.push_back(0);
        mesh->faceAttributes.push_back(faceAttrib);

        return mesh;
    }

    void updateMaterialSlots(std::shared_ptr<graph::Mesh> mesh, const int removedSlot)
    {
        for(FaceAttributes& face : mesh->faceAttributes)
        {
	        if(face.materialSlotId >= removedSlot && face.materialSlotId != 0)
	        {
                face.materialSlotId += -1;
	        }
        }
    }

    float gaussianFilter(const float* rgba, const unsigned int width, const unsigned int height, const unsigned int x, const unsigned int y, const bool isSpherical)
    {
        // Lookup is repeated in x and clamped to edge in y.
        unsigned int left;
        unsigned int right;
        unsigned int bottom = (0 < y) ? y - 1 : y; // clamp
        unsigned int top = (y < height - 1) ? y + 1 : y; // clamp

        // Match the filter to the texture object wrap setup for spherical and rectangular emission textures.
        if (isSpherical) // Spherical environment light 
        {
            left = (0 < x) ? x - 1 : width - 1; // repeat
            right = (x < width - 1) ? x + 1 : 0;         // repeat
        }
        else // Rectangular area light 
        {
            left = (0 < x) ? x - 1 : x; // clamp
            right = (x < width - 1) ? x + 1 : x; // clamp
        }


        // Center
        const float* p = rgba + (width * y + x) * 4;
        float intensity = (p[0] + p[1] + p[2]) * 0.619347f;

        // 4-neighbours
        p = rgba + (width * bottom + x) * 4;
        float f = p[0] + p[1] + p[2];
        p = rgba + (width * y + left) * 4;
        f += p[0] + p[1] + p[2];
        p = rgba + (width * y + right) * 4;
        f += p[0] + p[1] + p[2];
        p = rgba + (width * top + x) * 4;
        f += p[0] + p[1] + p[2];
        intensity += f * 0.0838195f;

        // 8-neighbours corners
        p = rgba + (width * bottom + left) * 4;
        f = p[0] + p[1] + p[2];
        p = rgba + (width * bottom + right) * 4;
        f += p[0] + p[1] + p[2];
        p = rgba + (width * top + left) * 4;
        f += p[0] + p[1] + p[2];
        p = rgba + (width * top + right) * 4;
        f += p[0] + p[1] + p[2];
        intensity += f * 0.0113437f;

        return intensity / 3.0f;
    }
    void computeTangents(std::vector<graph::VertexAttributes>& vertices, const std::vector<unsigned int>& indices)
    {
        for (size_t i = 0; i < indices.size(); i += 3)
        {
            const graph::VertexAttributes& v0 = vertices[indices[i]];
            const graph::VertexAttributes& v1 = vertices[indices[i + 1]];
            const graph::VertexAttributes& v2 = vertices[indices[i + 2]];

            const math::vec3f edge1 = v1.position - v0.position;
            const math::vec3f edge2 = v2.position - v0.position;

            const math::vec3f uv1 = v1.texCoord - v0.texCoord;
            const math::vec3f uv2 = v2.texCoord - v0.texCoord;

            float r = 1.0f / (uv1.x * uv2.y - uv1.y * uv2.x);

            math::vec3f tangent = (edge1 * uv2.y - edge2 * uv1.y) * r;

            vertices[indices[i]].tangent += tangent;
            vertices[indices[i + 1]].tangent += tangent;
            vertices[indices[i + 2]].tangent += tangent;
        }

        for (auto& vertex : vertices)
        {
            vertex.tangent = math::normalize(vertex.tangent);
        }
    }
    std::shared_ptr<graph::Group> simpleScene01()
    {
        auto sceneRoot = ops::createNode<Group>();
        VTX_INFO("Starting Scene");

        std::shared_ptr<Material> material1 = ops::createNode<Material>();
        material1->shader->name = "Stone_Mediterranean";
        material1->shader->path = "\\vMaterials_2\\Stone\\Stone_Mediterranean.mdl";
        //material1->shader->name = "Aluminum";
        //material1->shader->path = "\\vMaterials_2\\Metal\\Aluminum.mdl";
        //material1->shader->name = "bsdf_diffuse_reflection";
        //material1->shader->path = "\\bsdf_diffuse_reflection.mdl";

        std::shared_ptr<Material> materialEmissive = ops::createNode<Material>();
        materialEmissive->shader->name = "naturalwhite_4000k";
        materialEmissive->shader->path = "\\nvidia\\vMaterials\\AEC\\Lights\\Lights_Emitter.mdl";

        std::shared_ptr<Mesh> cube = ops::createBox();

        std::shared_ptr<Instance> Cube1 = ops::createNode<Instance>();
        Cube1->setChild(cube);
        Cube1->transform->translate(math::xAxis, 2.0f);
        Cube1->addMaterial(material1);

        std::shared_ptr<Instance> Cube2 = ops::createNode<Instance>();
        Cube2->setChild(cube);
        Cube2->transform->translate(math::yAxis, 2.0f);
        Cube2->addMaterial(material1);

        std::shared_ptr<Instance> Cube3 = ops::createNode<Instance>();
        Cube3->setChild(cube);
        Cube3->transform->rotateDegree(math::xAxis, 45.0f);
        Cube3->transform->translate(math::zAxis, 2.0f);
        Cube3->addMaterial(material1);

        std::shared_ptr<Mesh> plane = ops::createPlane();

        std::shared_ptr<Instance> GroundPlane = ops::createNode<Instance>();
        GroundPlane->setChild(plane);
        GroundPlane->transform->scale(100.0f);
        GroundPlane->transform->translate(math::zAxis, -1.0f);
        GroundPlane->addMaterial(material1);

        std::shared_ptr<Instance> AreaLight = ops::createNode<Instance>();
        AreaLight->setChild(plane);
        AreaLight->transform->rotateDegree(math::xAxis, 180.0f);
        AreaLight->transform->translate(math::zAxis, 7.0f);
        AreaLight->transform->scale(0.5f);
        AreaLight->addMaterial(materialEmissive);


        std::string envMapPath = getOptions()->dataFolder + "sunset_in_the_chalk_quarry_1k.hdr";
        //std::string envMapPath =  getOptions()->dataFolder  + "studio_small_03_1k.hdr";
        //std::string envMapPath =  getOptions()->dataFolder  + "16x16-in-1024x1024.png";
        //std::string envMapPath =  getOptions()->dataFolder  + "sunset03_EXR.exr";
        //std::string envMapPath =  getOptions()->dataFolder  + "morning07_EXR.exr";
        std::shared_ptr<Light> envLight = ops::createNode<Light>();
        auto attrib = std::make_shared<EvnLightAttributes>(envMapPath);
        envLight->attributes = attrib;

        sceneRoot->addChild(Cube1);
        sceneRoot->addChild(Cube2);
        sceneRoot->addChild(Cube3);
        sceneRoot->addChild(GroundPlane);
        sceneRoot->addChild(AreaLight);
        sceneRoot->addChild(envLight);

        return sceneRoot;
    }

    std::shared_ptr<graph::Group> importedScene()
    {
        //const std::string                   scenePath = getOptions()->dataFolder + "models/blenderTest2.fbx";
        const std::string                   scenePath = getOptions()->dataFolder + "models/blenderTest3.fbx";
        //const std::string                   scenePath = getOptions()->dataFolder + "models/sponza2/sponza.obj";
        //const std::string                   scenePath = getOptions()->dataFolder + "models/blenderTest.obj";
        //const std::string                   scenePath = getOptions()->dataFolder  + "models/blenderTest.fbx";
		const std::shared_ptr<graph::Group> sceneRoot = importer::importSceneFile(scenePath);

		const std::vector<std::shared_ptr<Node>> instances = SIM::getAllNodeOfType(NT_INSTANCE);

		const std::string moduleName   = "::CustomModule";
		const std::string materialName = "CustomMaterial";
        graph::createNewModule(moduleName, materialName);
		const std::shared_ptr<Material> material1 = ops::createNode<Material>();
        //material1->shader->name = "Stone_Mediterranean";
        //material1->shader->path = "\\vMaterials_2\\Stone\\Stone_Mediterranean.mdl";
        if(true)
        {
            //material1->shader->name = "bsdf_diffuse_reflection";
			//material1->shader->path = "\\bsdf_diffuse_reflection.mdl";
            //material1->shader->name = "Aluminum";
            //material1->shader->path = "\\vMaterials_2\\Metal\\Aluminum.mdl";
            material1->shader->name = "Stone_Mediterranean";
            material1->shader->path = "\\vMaterials_2\\Stone\\Stone_Mediterranean.mdl";
        }
        else
        {
            material1->shader->name = materialName;
            material1->shader->path = moduleName;
        }
        

        for(const std::shared_ptr<Node>& node : instances)
        {
			const std::shared_ptr<Instance> instance = std::dynamic_pointer_cast<Instance>(node);
            instance->addMaterial(material1);
		}

        //std::string envMapPath = getOptions()->dataFolder + "sunset_in_the_chalk_quarry_1k.hdr";
        //std::string envMapPath = getOptions()->dataFolder + "belfast_sunset_puresky_4k.hdr";
        std::string envMapPath = getOptions()->dataFolder + "mpumalanga_veld_puresky_4k.hdr";
        //std::string envMapPath = getOptions()->dataFolder + "blouberg_sunrise_2_1k.hdr";
        //std::string envMapPath = getOptions()->dataFolder + "qwantani_1k.hdr";
        
        //std::string envMapPath =  getOptions()->dataFolder  + "studio_small_03_1k.hdr";
        //std::string envMapPath =  getOptions()->dataFolder  + "CheckerBoard.png";
        //std::string envMapPath =  getOptions()->dataFolder  + "sunset03_EXR.exr";
        //std::string envMapPath =  getOptions()->dataFolder  + "morning07_EXR.exr";
		const std::shared_ptr<Light> envLight = ops::createNode<Light>();
		const auto                   attrib   = std::make_shared<EvnLightAttributes>(envMapPath);
        envLight->attributes                  = attrib;

        sceneRoot->addChild(envLight);

        return sceneRoot;
    }

}

