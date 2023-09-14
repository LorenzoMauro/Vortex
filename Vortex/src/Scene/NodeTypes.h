#pragma once
#include <map>
#include <string>

namespace vtx::graph
{
	enum NodeType {
		NT_GROUP,

		NT_INSTANCE,
		NT_MESH,
		NT_TRANSFORM,
		NT_LIGHT,
		NT_MESH_LIGHT,
		NT_ENV_LIGHT,


		NT_CAMERA,
		NT_RENDERER,

		NT_MATERIAL,
		NT_MDL_SHADER,
		NT_MDL_TEXTURE,
		NT_MDL_BSDF,
		NT_MDL_LIGHTPROFILE,

		NT_SHADER_TEXTURE,
		NT_SHADER_DF,
		NT_SHADER_MATERIAL,
		NT_SHADER_SURFACE,
		NT_SHADER_IMPORTED,
		NT_SHADER_COLOR,
		NT_SHADER_FLOAT3,
		NT_SHADER_FLOAT,
		NT_SHADER_COORDINATE,

		NT_NUM_NODE_TYPES
	};

	static std::map<NodeType, std::string> nodeNames{
		{NT_GROUP, "Group"},
		{ NT_INSTANCE, "Instance" },
		{ NT_MESH, "Mesh" },
		{ NT_TRANSFORM, "Transform" },
		{ NT_LIGHT, "Light" },
		{ NT_MESH_LIGHT, "Mesh Light" },
		{ NT_ENV_LIGHT, "Environment Light" },
		{ NT_CAMERA, "Camera" },
		{ NT_RENDERER, "Renderer" },
		{ NT_MATERIAL, "Material" },
		{ NT_MDL_SHADER, "Shader" },
		{ NT_MDL_TEXTURE, "Texture" },
		{ NT_MDL_BSDF, "Bsdf Measurement" },
		{ NT_MDL_LIGHTPROFILE, "Light Profile" },
		{ NT_SHADER_TEXTURE, "Shader Texture" },
		{ NT_SHADER_DF, "Shader Distribution Function" },
		{ NT_SHADER_MATERIAL, "Shader Material" },
		{ NT_SHADER_SURFACE, "Shader Surface" },
		{ NT_SHADER_IMPORTED, "Shader Imported" },
		{ NT_SHADER_COLOR, "Shader Color" },
		{ NT_SHADER_FLOAT3, "Shader 3d Vector" },
		{ NT_SHADER_FLOAT,"Shader Float" },
		{ NT_SHADER_COORDINATE, "Shader Coordinates" },
	};
}
