#include "Renderer.h"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include "Camera.h"
#include "Ray.h"
#include "Scene.h"
#include "glm/gtx/string_cast.hpp"
#include <iostream>
#include "Walnut/Random.h"
#include "Utils.h"
#include <execution>
#include <glm/gtc/random.hpp>

Renderer::Renderer(Scene& scene_) : m_Scene(scene_), m_Camera(scene_.Camera){}

void Renderer::OnResize(uint32_t width, uint32_t heigth) {

	if (m_FinalImage) {
		if (m_FinalImage->GetWidth() == width && m_FinalImage->GetHeight() == heigth) {
			return;
		}
		m_FinalImage->Resize(width, heigth);
	}
	else {
		m_FinalImage = std::make_shared<Walnut::Image>(width, heigth, Walnut::ImageFormat::RGBA);
	}
	delete[] m_ImageData;
	m_ImageData = new uint32_t[width * heigth];
	delete[] m_AccumulationData;
	m_AccumulationData = new glm::vec4[width * heigth];
	m_SampleCount = 1;

	m_PixelsHeightIterator.resize(heigth);
	m_PixelsWidthIterator.resize(width);

	for (int i = 0; i < heigth; i++) {
		m_PixelsHeightIterator[i] = i;
	}
	for (int i = 0; i < width; i++) {
		m_PixelsWidthIterator[i] = i;
	}
	
}

void Renderer::ClearImageData() {
	memset(m_AccumulationData, 0, m_FinalImage->GetWidth() * m_FinalImage->GetHeight() * sizeof(glm::vec4));
	memset(m_ImageData, 0, m_FinalImage->GetWidth() * m_FinalImage->GetHeight() * sizeof(uint32_t));
	m_FinalImage->SetData(m_ImageData);
	m_SampleCount = 1;
}

void Renderer::RestartRender(){
		m_SampleCount = 1;
}

void Renderer::Render() {

	if (m_SampleCount == 1) {
		memset(m_AccumulationData, 0, m_FinalImage->GetWidth() * m_FinalImage->GetHeight() * sizeof(glm::vec4));
	}
	
	if (m_Settings.MultiThread) {
		std::for_each(std::execution::par, m_PixelsHeightIterator.begin(), m_PixelsHeightIterator.end(), [this](uint32_t y) {
			std::for_each(m_PixelsWidthIterator.begin(), m_PixelsWidthIterator.end(), [this, y](uint32_t x) {
					glm::vec4 color = PerPixel(x, y);
					m_AccumulationData[y * m_FinalImage->GetWidth() + x] += color;

					glm::vec4 accumulatedColor = m_AccumulationData[y * m_FinalImage->GetWidth() + x] / (float)m_SampleCount;
					m_ImageData[x + y * m_FinalImage->GetWidth()] = Utils::ConvertToRGBA(accumulatedColor);
				});
			});
	}
	else {
		for (uint32_t y = 0; y < m_FinalImage->GetHeight(); y++)
		{
			for (uint32_t x = 0; x < m_FinalImage->GetWidth(); x++)
			{

				glm::vec4 color = PerPixel(x, y);
				m_AccumulationData[y * m_FinalImage->GetWidth() + x] += color;

				glm::vec4 accumulatedColor = m_AccumulationData[y * m_FinalImage->GetWidth() + x] / (float)m_SampleCount;
				m_ImageData[x + y * m_FinalImage->GetWidth()] = Utils::ConvertToRGBA(accumulatedColor);
			}
		}
	}
	

	// set value of m_ImageData to the value of m_AccumulationData divided by m_SampleCount
	if (m_Settings.Accumulate) {
		m_SampleCount++;
	}
	else {
		m_SampleCount = 1;
	}
	
	m_FinalImage->SetData(m_ImageData);

}

glm::vec4 Renderer::PerPixel(uint32_t x, uint32_t y)
{
	Ray ray;
	ray.Origin = m_Camera.GetPosition();
	ray.Direction = m_Camera.GetRayDirections()[x + y * m_FinalImage->GetWidth()];
	
	glm::vec3 finalColor{ 0.0f };
	int bounces = 2;
	float multiplier = 1.0f;
	
	for (int i=0; i < m_Settings.MaxBounces; i++) {
		HitPayload payload = TraceRay(ray);
		if (payload.HitDistance < 0) {
			finalColor += multiplier * BgShading(ray);
			break;
		}

		finalColor += multiplier * Shading(payload);
		const Sphere& sphere = m_Spheres[payload.ObjectIndex];
		const Material& material = m_Materials[sphere.MaterialIndex];
		
		ray.Origin = payload.WorldPosition + payload.WorldNormal * 0.0001f;
		glm::vec3 randomVector;
		if (m_Settings.MultiThread){
			randomVector = glm::vec3{ glm::linearRand(-0.5f, 0.5f),glm::linearRand(-0.5f, 0.5f),glm::linearRand(-0.5f, 0.5f) };
		}
		else {
			randomVector = Walnut::Random::Vec3(-0.5f, 0.5f);
		}
		
		ray.Direction = glm::reflect(ray.Direction, payload.WorldNormal + material.Roughness* randomVector);
		multiplier *= 0.3f;
		
	}
	
	return glm::vec4(finalColor, 1.0f);
}

glm::vec3 Renderer::Shading(const HitPayload& payload) {
	const Sphere& sphere = m_Scene.Spheres[payload.ObjectIndex];
	const Material& material = m_Scene.Materials[sphere.MaterialIndex];

	glm::vec3 lightDirection = -glm::normalize(m_Scene.SunLight.Direction);
	float lightIntensity = glm::max(0.0f, glm::dot(payload.WorldNormal, lightDirection));
	
	glm::vec3 color = material.Albedo * (m_Scene.SunLight.Color * lightIntensity + m_Scene.World.AmbientColor);
	
	return color;
}

glm::vec3 Renderer::BgShading(const Ray& ray)
{
	/*float bg_gradient_scalar = ray.Direction.z*0.5f + 0.5f;
	glm::vec3 bg_color = bg_gradient_scalar * m_Scene.m_World.AmbientColor;*/
	return m_Scene.World.AmbientColor;
}

Renderer::HitPayload Renderer::TraceRay(const Ray& ray) {

	float closestHitDistance = std::numeric_limits<float>::max();
	int closestSphere = -1;

	for (int i = 0; i < m_Spheres.size(); i++) {

		const Sphere& sphere = m_Spheres[i];

		float hitDistance = RaySphereIntersection(ray, sphere);
		if (hitDistance == -1) {
			continue;
		}
		else if (hitDistance < closestHitDistance) {
			closestHitDistance = hitDistance;
			closestSphere = (int)i;
		}
	}

	if (closestSphere < 0) {
		return OnMiss();
	}
	return OnHit(ray, closestHitDistance, closestSphere);

}

Renderer::HitPayload Renderer::OnHit(const Ray& ray, float hitDistance, int objectIndex) {
	const Sphere& closestSphere = m_Spheres[objectIndex];
	HitPayload payload;
	payload.ObjectIndex = objectIndex;
	payload.HitDistance = hitDistance;
	payload.WorldPosition = ray.Origin + hitDistance * ray.Direction;
	payload.WorldNormal = SphereIntersectionNormal(payload.WorldPosition, closestSphere);
	
	return payload;
}

Renderer::HitPayload Renderer::OnMiss() {
	HitPayload payload;
	payload.HitDistance = -1;
	return payload;
}


float Renderer::RaySphereIntersection(const Ray& ray, const Sphere& sphere) {
	glm::vec3 sphere_center = sphere.Origin;
	float sphere_radius = sphere.Radius;
	glm::vec3 deltaOriginSphereCenter = ray.Origin - sphere_center;
	float a = dot(ray.Direction, ray.Direction);
	float b = 2 * dot(deltaOriginSphereCenter, ray.Direction);
	float c = dot(deltaOriginSphereCenter, deltaOriginSphereCenter) - sphere_radius * sphere_radius;
	float delta = b * b - 4 * a * c;
	if (delta < 0) {
		return -1;
	}
	else {
		float t_hit = (-b - sqrt(delta)) / (2 * a);
		//float t2 = (-b + sqrt(delta)) / (2 * a);
		if (t_hit > 0) {
			return t_hit;
		}
		else {
			return -1;
		}
	}
}

glm::vec3 Renderer::SphereIntersectionNormal(const glm::vec3& intersection_point, const Sphere& sphere) {
	return glm::normalize(intersection_point - sphere.Origin);
}