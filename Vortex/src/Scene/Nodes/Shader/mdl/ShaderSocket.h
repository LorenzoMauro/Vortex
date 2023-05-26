#pragma once
#include "MDL/MdlWrapper.h"


namespace vtx::graph::shader
{
	struct ShaderSocket
	{
		std::shared_ptr<ShaderNode> node;
		mdl::ParameterInfo			parameterInfo;


		//size_t					m_index;
		//std::string				m_name;
		//std::string				m_displayName;
		//std::string				m_groupName;
		//ParamKind				m_kind;
		//ParamKind				m_arrayElemKind;
		//mi::Size				m_arraySize;
		//mi::Size				m_arrayPitch;   // the distance between two array elements
		//char* m_dataPtr;
		//float					m_rangeMin;
		//float					m_rangeMax;
		//const EnumTypeInfo* m_enumInfo;
	};
}

