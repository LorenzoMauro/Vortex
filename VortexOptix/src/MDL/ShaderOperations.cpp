#include "ShaderOperations.h"

namespace vtx::mdl
{
	FunctionInfo::FunctionInfo(const std::string& _moduleName, const std::string& _functionName)
		: moduleName(_moduleName), functionName(_functionName)
	{
	}
	FunctionInfo::FunctionInfo(const std::string& _moduleName, const std::string& _functionName, const std::string& _signature)
		: moduleName(_moduleName), functionName(_functionName), signature(_signature)
	{
	}
	std::string FunctionInfo::getSignature()
	{
		if (signature.empty())
		{
			const mdl::State* state = mdl::getState();
			ITransaction* transaction = mdl::getGlobalTransaction();
			const Handle<IMdl_execution_context> context = state->context;
			const Handle<IMdl_impexp_api>        impExp = state->impExpApi;
			Sint32 result;
			{
				//TODO check if module is already loaded
				result = impExp->load_module(transaction, mdl::removeMdlPrefix(moduleName).c_str(), context.get());
				VTX_ASSERT_CONTINUE((result >= 0 && mdl::logMessage(context.get())), state->lastError);

				const Handle<const IModule> module(transaction->access<IModule>(moduleName.c_str()));
				const Handle<const IArray>  overloads(module->get_function_overloads(functionName.c_str()));
				const Handle<const IString> mdlSignature(overloads->get_element<IString>(0));
				signature = mdlSignature->get_c_str();
			}
		}
		return signature;
	}

	void FunctionInfo::getParameters(int& potentialParameterCount)
	{

		const State* state = getState();
		ITransaction* transaction = getGlobalTransaction();

		const Handle<const IFunction_definition> functionDefinition(transaction->access<IFunction_definition>(getSignature().c_str()));
		const Handle<const IType_list>           types(functionDefinition->get_parameter_types());
		const Handle<const IExpression_list>	 defaults(functionDefinition->get_defaults());
		const Size                               paramCount = functionDefinition->get_parameter_count();

		for (int i = 0; i < paramCount; i++)
		{
			ParamInfo paramInfo;

			paramInfo.type = types->get_type(i);
			const char* name = types->get_name(i);
			paramInfo.argumentName = name;
			paramInfo.actualName = name + ("_" + std::to_string(potentialParameterCount));
			if (const int defaultIndex = defaults->get_index(name); defaultIndex != -1)
			{
				paramInfo.defaultValue = defaults->get_expression(defaultIndex);
			}
			potentialParameterCount++;

			parameters.insert({ name, paramInfo });
		}
	}

	FunctionInfo getConstantColor(const math::vec3f& color)
	{
		const State*                       state = getState();
		const Handle<IValue_factory>&      vf    = state->valueFactory;
		const Handle<IExpression_factory>& ef    = state->expressionFactory;
		const Handle<IValue>               colorValue(vf->create_color(color.x, color.y, color.z));
		const Handle<IExpression>          colorExpr(ef->create_constant(colorValue.get()));
		FunctionInfo                       colorFunction;
		colorFunction.expression = colorExpr;
		colorFunction.functionType = FunctionInfo::MDL_CONSTANT;
		return colorFunction;
	}


	FunctionInfo getConstantFloat(const float value)
	{
		const State*                       state = getState();
		const Handle<IValue_factory>&      vf    = state->valueFactory;
		const Handle<IExpression_factory>& ef    = state->expressionFactory;
		const Handle<IValue>               floatValue(vf->create_float(value));
		const Handle<IExpression>          floatExpr(ef->create_constant(floatValue.get()));
		FunctionInfo                       floatFunction;
		floatFunction.functionType = FunctionInfo::MDL_CONSTANT;
		floatFunction.expression = floatExpr;
		return floatFunction;
	}

	void createNewFunctionInModule(const ModuleCreationParameters& moduleParameters)
	{
		const State* state       = getState();
		ITransaction*     transaction = getGlobalTransaction();
		{
			const Handle<IMdl_execution_context> context = state->context;
			const Handle<IMdl_factory>           factory = state->factory;
			const Handle<IMdl_impexp_api>        impExp = state->impExpApi;
			Sint32 result;
			VTX_INFO("Creating new function {} in module: {} With parameters:", moduleParameters.functionName, moduleParameters.moduleName);
			for (int i = 0; i < moduleParameters.parameters->get_size(); i++)
			{
				VTX_INFO("Parameter: {}", moduleParameters.parameters->get_name(i));
			}

			const Handle moduleBuilder(factory->create_module_builder(transaction,
																	  ("mdl" + moduleParameters.moduleName).c_str(),
																	  MDL_VERSION_1_0,
																	  MDL_VERSION_LATEST,
																	  context.get()));
			// Add the material to the module.
			result = moduleBuilder->add_function(moduleParameters.functionName.c_str(),
												 moduleParameters.body.get(),
												 moduleParameters.parameters.get(),
												 moduleParameters.defaults.get(),
												 moduleParameters.parameterAnnotations.get(),
												 moduleParameters.annotations.get(),
												 moduleParameters.returnAnnotations.get(),
												 /*is_exported*/ true,
												 /*frequency_qualifier*/ IType::MK_NONE,
												 context.get());

			VTX_ASSERT_CONTINUE((mdl::logMessage(context.get()) && result >= 0), "");

			// Print the exported MDL source code to the console.
			const Handle<IString> moduleSource(transaction->create<IString>("String"));
			impExp->export_module_to_string(transaction, ("mdl" + moduleParameters.moduleName).c_str(), moduleSource.get(), context.get());
			VTX_INFO("Exported MDL source code:\n {}", moduleSource->get_c_str());
		}
		transaction->commit();
	}

	FunctionInfo getTextureConstant(const std::string& texturePath, const IType_texture::Shape shape , const float gamma)
	{
		const std::string textureFolder = utl::getFolder(texturePath);
		const std::string textureName   = "/" + utl::getFile(texturePath);
		mdl::addSearchPath(textureFolder);

		const State* state = getState();
		const Handle<IMdl_factory>& factory = state->factory;
		ITransaction* transaction = getGlobalTransaction();
		const Handle<IExpression_factory>& ef = state->expressionFactory;

		const Handle<IValue_texture>   argValue(factory->create_texture(transaction, textureName.c_str(), shape, gamma, nullptr, true, nullptr));
		VTX_INFO("Texture {} loaded with gamma : {} ", textureName, argValue->get_gamma());
		const Handle<IExpression>      argExpr(ef->create_constant(argValue.get()));
		FunctionInfo                      textureFunction;
		textureFunction.functionType = FunctionInfo::MDL_CONSTANT;
		textureFunction.expression = argExpr;
		return textureFunction;
	};

	FunctionInfo createFunction(ModuleCreationParameters& moduleParameters, MdlFunctionId functionId, std::map<std::string, FunctionInfo> arguments)
	{
		const State* state = getState();
		const Handle<IExpression_factory>& ef = state->expressionFactory;

		FunctionInfo functionInfo = mdlFunctionDb.functions[functionId];

		functionInfo.getParameters(moduleParameters.potentialParameterCount);

		const Handle<IExpression_list> callArguments(ef->create_expression_list());

		for (auto [name, param] : functionInfo.parameters)
		{
			if(name=="handle")
			{
				continue;
			}
			if (auto it = arguments.find(name); it != arguments.end())
			{
				if (arguments[name].functionType == FunctionInfo::MDL_EXPRESSION)
				{
					callArguments->add_expression(name.c_str(), arguments[name].expression.get());
				}
				else if(functionId != MDL_MATERIAL_SURFACE && functionId !=MDL_MATERIAL && functionId!=MDL_DIFFUSE_REFLECTION)
				{
					param.defaultValue = arguments[name].expression;
					moduleParameters.addParameter(param, callArguments);
					moduleParameters.addDefault(param);
				}
			}
			else if (functionId != MDL_MATERIAL_SURFACE && functionId != MDL_MATERIAL && functionId!=MDL_DIFFUSE_REFLECTION)
			{
				moduleParameters.addParameter(param, callArguments);
				moduleParameters.addDefault(param);
			}
			
		}
		functionInfo.expression = ef->create_direct_call(GET_SIGNATURE(functionInfo), callArguments.get());

		return functionInfo;
	}
	
}

