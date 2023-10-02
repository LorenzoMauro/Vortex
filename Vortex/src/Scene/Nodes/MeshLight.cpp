#include "MeshLight.h"
#include "Scene/Traversal.h"
#include "Material.h"
#include "Mesh.h"

namespace vtx::graph
{
	struct FaceAttributes;

	MeshLight::MeshLight() : Node(NT_MESH_LIGHT)
	{
	}

	MeshLight::~MeshLight()
	{
	}

	void MeshLight::init()
	{
		computeAreaCdf();
		isValid = (cdfAreas.size() > 1);
		state.isInitialized = true;
	}

	std::vector<std::shared_ptr<Node>> MeshLight::getChildren() const
	{
		return {};
	}

	void MeshLight::accept(NodeVisitor& visitor)
	{
		if (!isValid)
		{
			return;
		}
		visitor.visit(as<MeshLight>());
	}

	inline void MeshLight::computeAreaCdf()
	{
		VTX_INFO("Computing Mesh Area Light for Mesh {} with Material {}", mesh->getUID(), material->getUID());
		const size_t numTriangles = mesh->indices.size() / 3;
		size_t actualNumTriangles = 0;

		float areaSurface = 0.0f;
		cdfAreas.push_back(areaSurface); // CDF starts with zero. One element more than number of triangles.

		for (size_t i = 0; i < numTriangles; ++i)
		{
			FaceAttributes& face = mesh->faceAttributes[i];
			unsigned int materialSlotId = face.materialSlotId;

			if (materialSlotId != materialRelativeIndex)
			{
				continue;
			}
			actualNumTriangles++;

			const size_t idx = i * 3;

			const unsigned int i0 = mesh->indices[idx];
			const unsigned int i1 = mesh->indices[idx + 1];
			const unsigned int i2 = mesh->indices[idx + 2];

			// All in object space.
			const math::vec3f v0 = mesh->vertices[i0].position;
			const math::vec3f v1 = mesh->vertices[i1].position;
			const math::vec3f v2 = mesh->vertices[i2].position;

#ifdef AREA_COORDINATE_SYSTEM_TRANSFORMED
			// TODO Why should I change coordinate system if I ma just computing the area of a triangle?
			// PERF Work in world space to do fewer transforms during explicit light hits.
			math::vec3f p0(math::vec4f(v0.x, v0.y, v0.z, 1.0f) * matrix);
			math::vec3f p1(math::vec4f(v1.x, v1.y, v1.z, 1.0f) * matrix);
			math::vec3f p2(math::vec4f(v2.x, v2.y, v2.z, 1.0f) * matrix);
			math::vec3f e0 = p1 - p0;
			math::vec3f e1 = p2 - p0;
#else
			math::vec3f e0 = v1 - v0;
			math::vec3f e1 = v2 - v0;
#endif


			// The triangle area is half of the parallelogram area (length of cross product).
			const float area = math::length(cross(e0, e1)) * 0.5f;
			//const float area = dp::math::length(e0 ^ e1) * 0.5f;

			areaSurface += area;

			cdfAreas.push_back(areaSurface); // Store the unnormalized sums of triangle surfaces.
			actualTriangleIndices.push_back(i);
		}

		// Normalize the CDF values. 
		// PERF This means only the area integral value is in world space and the CDF could be reused for instanced mesh lights.
		for (auto& val : cdfAreas)
		{
			val /= areaSurface;
			// The last cdf element will automatically be 1.0f.
			// If this happens to be smaller due to inaccuracies in the floating point calculations, 
			// the clamping to valid triangle indices inside the sample_light_mesh() function will 
			// prevent out of bounds accesses, no need for corrections here.
			// (The corrections would be to set all identical values below 1.0f at the end of this array to 1.0f.)
		}

		area = areaSurface;

		VTX_INFO("Fnished Computation of Mesh Area Light for Mesh {} with Material {}", mesh->getUID(), material->getUID());
	}

}
