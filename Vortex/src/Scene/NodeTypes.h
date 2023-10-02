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
		NT_MDL_TEXTURE,
		NT_MDL_BSDF,
		NT_MDL_LIGHTPROFILE,

		NT_SHADER_DF,
		NT_SHADER_MATERIAL,
		NT_SHADER_SURFACE,
		NT_SHADER_IMPORTED,
		NT_SHADER_COORDINATE,
		NT_SHADER_NORMAL_TEXTURE,
		NT_SHADER_MONO_TEXTURE,
		NT_SHADER_COLOR_TEXTURE,
		NT_SHADER_BUMP_TEXTURE,
		NT_NORMAL_MIX,
		NT_GET_CHANNEL,
		NT_PRINCIPLED_MATERIAL,

		NT_NUM_NODE_TYPES,
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
		{ NT_MDL_TEXTURE, "Texture" },
		{ NT_MDL_BSDF, "Bsdf Measurement" },
		{ NT_MDL_LIGHTPROFILE, "Light Profile" },
		{ NT_SHADER_DF, "Shader Distribution Function" },
		{ NT_SHADER_MATERIAL, "Shader Material" },
		{ NT_SHADER_SURFACE, "Shader Surface" },
		{ NT_SHADER_IMPORTED, "Shader Imported" },
		{ NT_SHADER_COORDINATE, "Shader Coordinates" },
		{ NT_SHADER_NORMAL_TEXTURE, "Shader Normal Texture" },
		{ NT_SHADER_MONO_TEXTURE, "Shader Mono Texture" },
		{ NT_SHADER_COLOR_TEXTURE, "Shader Color Texture" },
		{ NT_SHADER_BUMP_TEXTURE, "Shader Bump Texture" },
		{ NT_NORMAL_MIX, "Normal Mix" },
		{ NT_GET_CHANNEL, "Get Channel" },
		{ NT_PRINCIPLED_MATERIAL, "Principled Material" }
	};

	static std::map<std::string, NodeType> nameToNodeType{
		{"Group", NT_GROUP},
		{"Instance" , NT_INSTANCE },
		{"Mesh" , NT_MESH },
		{"Transform" , NT_TRANSFORM },
		{"Light" , NT_LIGHT },
		{"Mesh Light" , NT_MESH_LIGHT },
		{"Environment Light" , NT_ENV_LIGHT },
		{"Camera" , NT_CAMERA },
		{"Renderer" , NT_RENDERER },
		{"Material" , NT_MATERIAL },
		{"Texture" , NT_MDL_TEXTURE },
		{"Bsdf Measurement" , NT_MDL_BSDF },
		{"Light Profile" , NT_MDL_LIGHTPROFILE },
		{"Shader Distribution Function" , NT_SHADER_DF },
		{"Shader Material" , NT_SHADER_MATERIAL },
		{"Shader Surface" , NT_SHADER_SURFACE },
		{"Shader Imported" , NT_SHADER_IMPORTED },
		{"Shader Coordinates" , NT_SHADER_COORDINATE },
		{"Shader Normal Texture" , NT_SHADER_NORMAL_TEXTURE },
		{"Shader Mono Texture" , NT_SHADER_MONO_TEXTURE },
		{"Shader Color Texture" , NT_SHADER_COLOR_TEXTURE },
		{"Shader Bump Texture" , NT_SHADER_BUMP_TEXTURE },
		{"Normal Mix" , NT_NORMAL_MIX },
		{"Get Channel" , NT_GET_CHANNEL },
		{"Principled Material" , NT_PRINCIPLED_MATERIAL }
	};
}
