#include "traversal.h"

namespace vtx::mdl
{


	bool isValidDf(Handle<IExpression const> expr)
	{
		const Handle exprConstant(expr->get_interface<IExpression_constant>());
		const Handle value(exprConstant->get_value());
		if (value->get_kind() == IValue::VK_INVALID_DF)
		{
			return false;
		}
		return true;
	};

}
