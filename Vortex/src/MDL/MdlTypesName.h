#pragma once
#include <map>
#include <mi/mdl_sdk.h>

namespace vtx::mdl
{
	static inline std::map<mi::neuraylib::IType::Kind, std::string> ITypeToString({
		{mi::neuraylib::IType::Kind::TK_ALIAS ,"Alias"},
		{mi::neuraylib::IType::Kind::TK_BOOL ,"Bool"},
		{mi::neuraylib::IType::Kind::TK_INT ,"Int"},
		{mi::neuraylib::IType::Kind::TK_ENUM ,"Enum"},
		{mi::neuraylib::IType::Kind::TK_FLOAT ,"Float"},
		{mi::neuraylib::IType::Kind::TK_DOUBLE ,"Double"},
		{mi::neuraylib::IType::Kind::TK_STRING ,"String"},
		{mi::neuraylib::IType::Kind::TK_VECTOR ,"Vector"},
		{mi::neuraylib::IType::Kind::TK_MATRIX ,"Matrix"},
		{mi::neuraylib::IType::Kind::TK_COLOR ,"Color"},
		{mi::neuraylib::IType::Kind::TK_ARRAY ,"Array"},
		{mi::neuraylib::IType::Kind::TK_STRUCT ,"Struct"},
		{mi::neuraylib::IType::Kind::TK_TEXTURE ,"Texture"},
		{mi::neuraylib::IType::Kind::TK_LIGHT_PROFILE ,"LightProfile"},
		{mi::neuraylib::IType::Kind::TK_BSDF_MEASUREMENT ,"BsdfMeasurement"},
		{mi::neuraylib::IType::Kind::TK_BSDF ,"Bsdf"},
		{mi::neuraylib::IType::Kind::TK_HAIR_BSDF ,"HairBsdf"},
		{mi::neuraylib::IType::Kind::TK_EDF ,"EDF"},
		{mi::neuraylib::IType::Kind::TK_VDF ,"VDF"}
	});

}