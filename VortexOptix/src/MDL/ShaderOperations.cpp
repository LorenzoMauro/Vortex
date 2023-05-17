//#include "ShaderOperations.h"
//
//#include "Scene/Nodes/Shader/mdl/ShaderNodes.h"
//
//namespace vtx::mdl
//{
	/*FunctionInfo::FunctionInfo(const std::string& _moduleName, const std::string& _functionName)
		: moduleName(_moduleName), functionName(_functionName)
	{
	}
	FunctionInfo::FunctionInfo(const std::string& _moduleName, const std::string& _functionName, const std::string& _signature)
		: moduleName(_moduleName), functionName(_functionName), signature(_signature)
	{
	}

	void FunctionInfo::getParameters(int& potentialParameterCount)
	{

		
	}

	FunctionInfo getConstantColor(const math::vec3f& color)
	{
		MdlState*                          state = getState();
		const TransactionInterfaces*       tI    = state->getTransactionInterfaces();
		const Handle<IValue_factory>&      vf    = tI->valueFactory;
		const Handle<IExpression_factory>& ef    = tI->expressionFactory;
		const Handle<IValue>               colorValue(vf->create_color(color.x, color.y, color.z));
		const Handle<IExpression>          colorExpr(ef->create_constant(colorValue.get()));
		FunctionInfo                       colorFunction;
		colorFunction.expression = colorExpr;
		colorFunction.functionType = FunctionInfo::MDL_CONSTANT;
		return colorFunction;
	}


	FunctionInfo getConstantFloat(const float value)
	{
		MdlState*                          state = getState();
		const TransactionInterfaces*       tI    = state->getTransactionInterfaces();
		const Handle<IValue_factory>&      vf    = tI->valueFactory;
		const Handle<IExpression_factory>& ef    = tI->expressionFactory;
		const Handle<IValue>               floatValue(vf->create_float(value));
		const Handle<IExpression>          floatExpr(ef->create_constant(floatValue.get()));
		FunctionInfo                       floatFunction;
		floatFunction.functionType = FunctionInfo::MDL_CONSTANT;
		floatFunction.expression = floatExpr;
		return floatFunction;
	}

	void createNewFunctionInModule(const ModuleCreationParameters& moduleParameters)
	{
		
	}

	FunctionInfo getTextureConstant(const std::string& texturePath, const IType_texture::Shape shape , const float gamma)
	{
		MdlState*                    state = getState();
		const TransactionInterfaces* tI    = state->getTransactionInterfaces();

		const std::string textureFolder = utl::getFolder(texturePath);
		const std::string textureName   = "/" + utl::getFile(texturePath);
		mdl::addSearchPath(textureFolder);

		const Handle<IMdl_factory>& factory = state->factory;
		const Handle<IExpression_factory>& ef = tI->expressionFactory;

		const Handle<IValue_texture>   argValue(factory->create_texture(tI->transaction.get(), textureName.c_str(), shape, gamma, nullptr, true, nullptr));
		VTX_INFO("Texture {} loaded with gamma : {} ", textureName, argValue->get_gamma());
		const Handle<IExpression>      argExpr(ef->create_constant(argValue.get()));
		FunctionInfo                      textureFunction;
		textureFunction.functionType = FunctionInfo::MDL_CONSTANT;
		textureFunction.expression = argExpr;
		return textureFunction;
	};


	Handle<IExpression> generateExpression(ModuleCreationParameters& moduleInfo, graph::shader::ShaderInputSockets& arguments)
	{
		
		return functionInfo;
	}*/
	
//}

