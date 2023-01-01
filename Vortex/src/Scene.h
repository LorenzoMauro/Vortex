#pragma once
#include <glm/gtc/matrix_transform.hpp>
#include "Camera.h"
#include<vector>
#include <string>
#include "Utils.h"


struct Material {
	glm::vec3 Albedo{ 0.5f };
	float Roughness = 0.5f;
	float Metallic = 0.0f;
	std::string Name = "Material";
};

struct Sphere
{
	glm::vec3 Origin{ 0.0f };
	float Radius = 0.5f;
	int MaterialIndex = 0;
};

struct World {
	glm::vec3 AmbientColor{ 0.0f, 0.0f, 0.0f };
};

struct SunLight {
	glm::vec3 Direction{ 1.0f, -1.0f, -1.0f };
	glm::vec3 Color{ 1.0f, 1.0f, 1.0f };
};

class Scene
{
public:

	Scene() {
		AddMaterial();
		SphereGroundSky();
		//AxisSpheresAndGround();
	}
	void AddSphere() {
		Sphere sphere;
		Spheres.push_back(sphere);
	}
	void DeleteSphere( int i) {
		Spheres.erase(Spheres.begin() + i);
	}
	void AddMaterial(std::string Name ="Material") {
		Material mat;
		
		std::string newName = Utils::FindAvailableName(Name, MaterialNames);
		
		mat.Name = newName;

		Materials.push_back(mat);
		MaterialNames.push_back(newName);
	}
	
	void DeleteMaterial(int i) {
		Materials.erase(Materials.begin() + i);
		MaterialNames.erase(MaterialNames.begin() + i);
		for (Sphere& sphere : Spheres) {
			if (sphere.MaterialIndex = i) {
				sphere.MaterialIndex = 0;
			}
		}
	}

	void SphereGroundSky() {
		AddSphere();
		AddSphere();
		AddMaterial();
		
		Spheres[0].Radius = 1000.0f;
		Spheres[0].Origin.z = -Spheres[0].Radius-1.f;
		Materials[Spheres[0].MaterialIndex].Roughness = 0.15f;
		Materials[Spheres[0].MaterialIndex].Albedo= glm::vec3(0.3f, 0.3f, 0.7f);
		
		Spheres[1].Radius = 1.0f;
		Spheres[1].MaterialIndex = 1;
		Materials[Spheres[1].MaterialIndex].Albedo = glm::vec3(0.8f, 0.0f, 0.8f);
		Materials[Spheres[1].MaterialIndex].Roughness = 0.5f;
		World.AmbientColor = glm::vec3(0.5f, 0.7f, 0.9f);
	}
	void AxisSpheresAndGround() {
		AddSphere();
		AddSphere();
		AddSphere();
		AddSphere();
		AddSphere();
		Spheres[1].Origin.x = 1.0f;
		Materials[Spheres[1].MaterialIndex].Albedo = glm::vec3(1.0f, 0.0f, 0.0f);
		Spheres[2].Origin.y = 1.0f;
		Materials[Spheres[2].MaterialIndex].Albedo = glm::vec3(0.0f, 1.0f, 0.0f);
		Spheres[3].Origin.z = 1.0f;
		Materials[Spheres[3].MaterialIndex].Albedo = glm::vec3(0.0f, 0.0f, 1.0f);
		Spheres[3].Origin.z = 1.0f;
		Spheres[4].Radius = 1000.0f;
		Spheres[4].Origin.z = -Spheres[4].Radius - 0.5f;
		Materials[Spheres[4].MaterialIndex].Albedo = glm::vec3(0.5f, 0.5f, 0.5f);
	}
public:
	Camera Camera{55.0f, 0.1f, 100.0f};
	std::vector<Sphere> Spheres;
	Material DefaultMaterial;
	std::vector<Material> Materials;
	std::vector<std::string> MaterialNames;
	SunLight SunLight;
	World World;
};