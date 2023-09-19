#include "CudaLinker.h"
#include <cuda.h>

#include "Device/CUDAChecks.h"


namespace vtx::mdl
{


    std::string extractFunctionPrototype(const std::string& ptxCode, const std::string& functionName, const std::string& newModifier = "") {
        // Search for the function name
        const std::string searchStr = ".func " + functionName + "(";
        const size_t      namePos = ptxCode.find(searchStr);

        // If found, find the start of the line before the modifier and extract until the closing parenthesis
        if (namePos != std::string::npos) {
            const size_t startPos = ptxCode.rfind('\n', namePos);
            const size_t endPos = ptxCode.find(')', namePos + searchStr.length());

            if (startPos != std::string::npos && endPos != std::string::npos) {
                std::string extractedPrototype = ptxCode.substr(startPos + 1, endPos - startPos);

                // If a new modifier is specified, replace the original modifier with the new one
                if (!newModifier.empty()) {
                    size_t modifierEndPos = extractedPrototype.find(".func");
                    if (modifierEndPos != std::string::npos) {
                        extractedPrototype.replace(0, modifierEndPos, newModifier + " ");
                    }
                }

                return extractedPrototype;
            }
        }

        // Return an empty string if the function is not found
        return "";
    }
    
    using namespace mi;
    using namespace base;
    using namespace neuraylib;
    // Helper function to create PTX source code for a non-empty 32-bit value array.
    void printArrayU32(std::string& str, std::string const& name, unsigned count, std::string const& content)
    {
        str += ".visible .const .align 4 .u32 " + name + "[";
        if (count == 0) {
            // PTX does not allow empty arrays, so use a dummy entry
            str += "1] = { 0 };\n";
        }
        else {
            str += to_string(count) + "] = { " + content + " };\n";
        }
    }

    // Helper function to create PTX source code for a non-empty function pointer array.
    void printArrayFunc(std::string& str, std::string const& name, unsigned count, std::string const& content)
    {
        str += ".visible .const .align 8 .u64 " + name + "[";
        if (count == 0) {
            // PTX does not allow empty arrays, so use a dummy entry
            str += "1] = { dummy_func };\n";
        }
        else {
            str += to_string(count) + "] = { " + content + " };\n";
        }
    }

    // Generate PTX array containing the references to all generated functions.
    std::string generateFuncArrayPtx(const std::vector<Handle<const ITarget_code>>& targetCodes)
    {
        // Create PTX header and mdl_expr_functions_count constant
        std::string src =
            ".version 4.0\n"
            ".target sm_20\n"
            ".address_size 64\n";

        // Workaround needed to let CUDA linker resolve the function pointers in the arrays.
        // Also used for "empty" function arrays.
        src += ".func dummy_func() { ret; }\n";

        std::string tcOffsets;
        std::string functionNames;
        std::string tcIndices;
        std::string abIndices;
        unsigned fCount = 0;

        // Iterate over all target codes
        for (size_t tc_index = 0, num = targetCodes.size(); tc_index < num; ++tc_index)
        {
			Handle<const ITarget_code> const& target_code = targetCodes[tc_index];

            std::string newEvaluateMaterialNameFunction = "evaluateMaterial_" + to_string(tc_index);

            // in case of multiple target codes, we need to address the functions by a pair of
            // target_code_index and function_index.
            // the elements in the resulting function array can then be index by offset + func_index.
            if (!tcOffsets.empty())
            {
	            tcOffsets += ", ";
            }
            tcOffsets += to_string(fCount);

            bool original = false;
            // Collect all names and prototypes of callable functions within the current target code
            //for (size_t func_index = 0, func_count = target_code->get_callable_function_count(); func_index < func_count; ++func_index)
            //{
            //    // add to function list
            //    if (!tcIndices.empty())
            //    {
            //        tcIndices += ", ";
            //        functionNames += ", ";
            //        abIndices += ", ";
            //    }

            //    // target code index in case of multiple link units
            //    tcIndices += to_string(tc_index);

            //    // name of the function
            //    functionNames += target_code->get_callable_function(func_index);

            //    // Get argument block index and translate to 1 based list index (-> 0 = not-used)
            //    Size ab_index = target_code->get_callable_function_argument_block_index(func_index);
            //    abIndices += to_string(ab_index == Size(~0) ? 0 : (ab_index + 1));
            //    fCount++;

            //    // Add prototype declaration
            //    src += target_code->get_callable_function_prototype(func_index, ITarget_code::SL_PTX);
            //    src += '\n';
            //}
            if (!fCount==0)
            {
                functionNames += ", ";
            }
            std::string newCode   = utl::replaceFunctionNameInPTX(target_code->get_code(), "__replace__EvaluateMaterial", newEvaluateMaterialNameFunction);
            std::string prototype = extractFunctionPrototype(newCode, newEvaluateMaterialNameFunction, ".extern");
            src += prototype + ";\n";
            functionNames += newEvaluateMaterialNameFunction;
            fCount++;
        }

        // infos per target code (link unit)
        //src += std::string(".visible .const .align 4 .u32 mdl_target_code_count = ") + to_string(targetCodes.size()) + ";\n";
        //printArrayU32(src, std::string("mdl_target_code_offsets"), unsigned(targetCodes.size()), tcOffsets);

        // infos per function
        src += std::string(".visible .const .align 4 .u32 mdl_functions_count = ") + to_string(fCount) + ";\n";
        printArrayFunc(src, std::string("mdl_functions"), fCount, functionNames);
        //printArrayU32(src, std::string("mdl_arg_block_indices"), fCount, abIndices);
        //printArrayU32(src, std::string("mdl_target_code_indices"), fCount, tcIndices);

        VTX_INFO("Generated Mdl Function Array Ptx:\n{}",src);
        return src;
    }



    // Build a linked CUDA kernel containing our kernel and all the generated code, making it
    // available to the kernel via an added "mdl_expr_functions" array.
    CUmodule buildLinkedKernel(std::vector<Handle<const ITarget_code>> const& targetCodes, const char* ptxFile, const char* kernelFunctionName, CUfunction* outKernelFunction)
    {
        // Generate PTX array containing the references to all generated functions.
        // The linker will resolve them to addresses.

        std::string ptx_func_array_src = generateFuncArrayPtx(targetCodes);
#ifdef DUMP_PTX
        std::cout << "Dumping CUDA PTX code for the \"mdl_expr_functions\" array:\n\n"
            << ptx_func_array_src << std::endl;
#endif

        // Link all generated code, our generated PTX array and our kernel together

		CUlinkState  cudaLinkState;
		CUmodule     cudaModule;
		void*        linkedCubin = nullptr;
		size_t       linkedCubinSize = 0;
		char         errorLog[8192], infoLog[8192];
		CUjit_option options[4];
		void*        optionVals[4];

        // Setup the linker

		// Pass a buffer for info messages
		options[0]    = CU_JIT_INFO_LOG_BUFFER;
		optionVals[0] = infoLog;
		// Pass the size of the info buffer
		options[1]    = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
		optionVals[1] = reinterpret_cast<void*>(uintptr_t(sizeof(infoLog)));
		// Pass a buffer for error messages
		options[2]    = CU_JIT_ERROR_LOG_BUFFER;
		optionVals[2] = errorLog;
		// Pass the size of the error buffer
		options[3]    = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
		optionVals[3] = reinterpret_cast<void*>(uintptr_t(sizeof(errorLog)));

        CU_CHECK(cuLinkCreate(4, options, optionVals, &cudaLinkState));

        CUresult linkResult = CUDA_SUCCESS;
        do {

            // Add all code generated by the MDL PTX backend
            for (size_t i = 0, numTargetCodes = targetCodes.size(); i < numTargetCodes; ++i) {
				const char* code = targetCodes[i]->get_code();

                std::string newCode = utl::replaceFunctionNameInPTX(code, "__replace__EvaluateMaterial", "evaluateMaterial_" + std::to_string(i));
                newCode             = utl::replaceFunctionNameInPTX(newCode, "__direct_callable__EvaluateMaterial", "unusedDirectCallable_" + std::to_string(i));
                linkResult          = cuLinkAddData(
                    cudaLinkState, CU_JIT_INPUT_PTX,
                    newCode.data(), //const_cast<char*>(),
                    newCode.length() + 1,
                    nullptr, 0, nullptr, nullptr);
                VTX_ASSERT_CLOSE(linkResult == CUDA_SUCCESS, "PTX linker error:\n {} \n info Log {}", errorLog, infoLog);

                if (linkResult != CUDA_SUCCESS) break;
            }
            if (linkResult != CUDA_SUCCESS) break;


            // Add the "mdl_expr_functions" array PTX module
            linkResult = cuLinkAddData(
                cudaLinkState, CU_JIT_INPUT_PTX,
                const_cast<char*>(ptx_func_array_src.c_str()),
                ptx_func_array_src.size(),
                nullptr, 0, nullptr, nullptr);
            VTX_ASSERT_CLOSE(linkResult == CUDA_SUCCESS, "PTX linker error:\n {} \n info Log {}", errorLog, infoLog);

            if (linkResult != CUDA_SUCCESS) break;

            // Add our kernel
            linkResult = cuLinkAddFile(
                cudaLinkState, CU_JIT_INPUT_PTX,
                ptxFile, 0, nullptr, nullptr);
            if (linkResult != CUDA_SUCCESS) break;

            // Link everything to a cubin
            linkResult = cuLinkComplete(cudaLinkState, &linkedCubin, &linkedCubinSize);
            VTX_ASSERT_CLOSE(linkResult == CUDA_SUCCESS, "PTX linker error:\n {} \n info Log {}", errorLog, infoLog);

        } while (false);
        CU_CHECK(linkResult);
        
        VTX_INFO("CUDA link completed.");
        if (infoLog[0])
            VTX_INFO("CUDA linker output:\n {}", infoLog);

        // Load the result and get the entrypoint of our kernel
        CU_CHECK(cuModuleLoadData(&cudaModule, linkedCubin));
        CU_CHECK(cuModuleGetFunction(
            outKernelFunction, cudaModule, kernelFunctionName));

        int regs = 0;
        CU_CHECK(cuFuncGetAttribute(&regs, CU_FUNC_ATTRIBUTE_NUM_REGS, *outKernelFunction));
        int lmem = 0;
        CU_CHECK(cuFuncGetAttribute(&lmem, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, *outKernelFunction));
        VTX_INFO("Kernel uses {} registers and {} lmem and has a size of {} bytes.", regs, lmem, linkedCubinSize);

        // Cleanup
        CU_CHECK(cuLinkDestroy(cudaLinkState));

        return cudaModule;
    }


    static MdlCudaLinker* mdlCudaLinker = nullptr;

    MdlCudaLinker& getMdlCudaLinker()
    {
        if (!mdlCudaLinker)
        {
	        mdlCudaLinker = new MdlCudaLinker();
        }

		return *mdlCudaLinker;
    }

	void MdlCudaLinker::submitTargetCode(const mi::base::Handle<const mi::neuraylib::ITarget_code>& targetCode, const std::string& materialName)
    {
        m_targetCodes.push_back(targetCode);
        matNametoTargetCode[materialName] = m_targetCodes.size() - 1;
        isDirty = true;
    }

	int MdlCudaLinker::getMdlFunctionIndices(const std::string& material)
    {
        if (isDirty)
        {
            link();
        }

        return matNametoTargetCode[material];
    }

    void MdlCudaLinker::link()
    {
        VTX_INFO("Linking CUDA MDL code.");
        outModule = buildLinkedKernel(m_targetCodes, ptxFile.c_str(), kernelFunctionName.c_str(), &outKernelFunction);
        isDirty = false;
    }

	CUfunction& MdlCudaLinker::getKernelFunction()
    {
        if(isDirty)
        {
            link();
        }
	    return outKernelFunction;
    }
}

