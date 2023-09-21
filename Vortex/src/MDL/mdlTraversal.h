#pragma once
#include<mi/mdl_sdk.h>
#include "Core/Math.h"

namespace vtx::mdl
{
	using namespace mi::base;
	using namespace mi::neuraylib;

	template <typename T>
	T getValue(mi::base::Handle<mi::neuraylib::IExpression const> expr)
	{
		return T();
	};

	template<>
	static inline bool getValue(const Handle<IExpression const> expr)
	{
		const Handle exprConstant(expr->get_interface<IExpression_constant>());
		const Handle<IValue const> value(exprConstant->get_value());
		const Handle<IValue_bool const> valueBool(value->get_interface<IValue_bool>());
		return valueBool->get_value();
	}

	template<>
	static inline std::string getValue(const Handle<IExpression const> expr)
	{
		const Handle exprConstant(expr->get_interface<IExpression_constant>());
		const Handle value(exprConstant->get_value());
		const Handle valueString(value->get_interface<IValue_string>());
		return valueString->get_value();
	}

	template<>
	static inline int getValue(const Handle<IExpression const> expr)
	{

		const Handle exprConstant(expr->get_interface<IExpression_constant>());
		const Handle value(exprConstant->get_value());
		if(IValue::Kind kind = value->get_kind(); kind==IValue::Kind::VK_ENUM)
		{
			const Handle valueEnum(value->get_interface<IValue_enum>());
			return valueEnum->get_value();
		}
		const Handle valueInt(value->get_interface<IValue_int>());
		return valueInt->get_value();
	}

	template<>
	static inline float getValue(const Handle<IExpression const> expr)
	{
		const Handle exprConstant(expr->get_interface<IExpression_constant>());
		const Handle value(exprConstant->get_value());
		const Handle valueFloat(value->get_interface<IValue_float>());
		return valueFloat->get_value();
	}


	template<>
	static inline double getValue(const Handle<IExpression const> expr)
	{
		const Handle exprConstant(expr->get_interface<IExpression_constant>());
		const Handle value(exprConstant->get_value());
		const Handle valueDouble(value->get_interface<IValue_double>());
		return valueDouble->get_value();
	}

	template<>
	static inline math::vec3f getValue(const Handle<IExpression const> expr)
	{
		const Handle exprConstant(expr->get_interface<IExpression_constant>());
		const Handle value(exprConstant->get_value());
		const Handle valueColor(value->get_interface<IValue_color>());
		mi::Color color;
		get_value(valueColor.get(), color);
		math::vec3f colorVec(color.r, color.g, color.b);
		return colorVec;
	}

	bool isValidDf(Handle<IExpression const> expr);

	template <typename T>
	struct ExprEvaluation
	{
		bool isConstant = false;
		bool isValid = false;
		T value;
	};

	template<typename T>
	ExprEvaluation<T> analyzeExpression(const mi::base::Handle<const mi::neuraylib::ICompiled_material> compiledMaterial, const std::string exprName)
	{
		Handle<IExpression const> expr(compiledMaterial->lookup_sub_expression(exprName.c_str()));

		ExprEvaluation<T> result;
		if(expr.get()==nullptr)
		{
			result.isValid = false;
			return result;
		}
		const IExpression::Kind exprKind = expr->get_kind();
		const IType::Kind       exprTypeKind = expr->get_type()->skip_all_type_aliases()->get_kind();


		if (exprTypeKind == IType::Kind::TK_BSDF || exprTypeKind == IType::Kind::TK_VDF || exprTypeKind == IType::Kind::TK_HAIR_BSDF || exprTypeKind == IType::Kind::TK_EDF)
		{
			if (exprKind == IExpression::EK_CONSTANT)
			{
				result.isValid = isValidDf(expr);
				result.isConstant = true;
				return result;
			}
		}
		switch (exprKind)
		{
			case IExpression::EK_CONSTANT:
			{
				result.isConstant = true;
				result.value = getValue<T>(expr);
				result.isValid = true;
			}
			break;
			case IExpression::EK_CALL:
			{
				result.isConstant = false;
				result.isValid = true;
			}
			break;
			case IExpression::EK_PARAMETER:
			{
				result.isConstant = false;
				result.isValid = true;
			}
			break;
			case IExpression::EK_DIRECT_CALL:
			{
				result.isConstant = false;
				result.isValid = true;
			}
			break;
			case IExpression::EK_TEMPORARY:
			{
				result.isConstant = false;
				result.isValid = true;
			}
			break;
			case IExpression::EK_FORCE_32_BIT:
			{
				result.isConstant = false;
				result.isValid = true;
			}
			break;
			default:;
		}
		return result;
	}

}