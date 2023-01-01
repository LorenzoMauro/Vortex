#pragma once
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include "Walnut/Image.h"
#include <memory>
#include "Camera.h"
#include "Scene.h"
#include "Ray.h"
#define PI 3.14159265358979323846

class Renderer {
public:
	static struct RenderSettings {
		bool Accumulate = true;
		int MaxBounces = 2;
		bool MultiThread = false;
		int MaxThreads = 8;
	};
	
	bool isSceneUpdated = false;
	
public:

	Renderer(Scene& m_scene_);
	
	void OnResize(uint32_t width, uint32_t heigth);

	void ClearImageData();

	void RestartRender();
	
	void Render();

	std::shared_ptr<Walnut::Image> GetFinalImage() const { return m_FinalImage; }

	RenderSettings& GetSettings() { return m_Settings; }
	
	int GetSamplesPerPixel() { return m_SampleCount; }
	
private:
	struct HitPayload
	{
		float HitDistance;
		glm::vec3 WorldPosition;
		glm::vec3 WorldNormal;
		
		int ObjectIndex;
	};
	
	glm::vec4 PerPixel(uint32_t x, uint32_t y);

	glm::vec3 Shading(const HitPayload& payload);
	glm::vec3 BgShading(const Ray& ray);
	
	HitPayload TraceRay(const Ray& ray);

	HitPayload OnHit(const Ray& ray, float hitDistance, int objectIndex);
		
	HitPayload OnMiss();
	
	float RaySphereIntersection(const Ray& ray, const Sphere& sphere);
	
	glm::vec3 SphereIntersectionNormal(const glm::vec3& intersection_point, const Sphere& sphere);

	
private:
	RenderSettings m_Settings;
	Scene &m_Scene;
	Camera& m_Camera = m_Scene.Camera;
	std::vector<Sphere>& m_Spheres = m_Scene.Spheres;
	std::vector<Material>& m_Materials = m_Scene.Materials;
	std::shared_ptr<Walnut::Image> m_FinalImage;
	uint32_t* m_ImageData = nullptr;
	glm::vec4* m_AccumulationData = nullptr;
	int m_SampleCount = 1;

	std::vector<uint32_t> m_PixelsWidthIterator;
	std::vector<uint32_t> m_PixelsHeightIterator;
};
