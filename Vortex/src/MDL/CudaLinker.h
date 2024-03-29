﻿#pragma once
#include <cuda.h>
#include "MdlWrapper.h"
#include <map>
namespace vtx::mdl
{
    //NOTE TO SELF
    // Basically what we do is take the code generated by MDL after inlining of the material functions into the uber function in MaterialDirectCallable.cu
    // compiled by clang.
    // Then we create on the fly a PTX file that contains an array of function pointers to the generated functions.
    // This array is reference as external inside rendererFunctions.h which is included in shadeKernel.cu
    // Then we link the shadeKernel ptx, the generated ptx and the Mdl uber function ptx together. Obtaining a CUFunction pointer we can call.


    // Return a textual representation of the given value.
    template <typename T>
    std::string to_string(T val)
    {
        std::ostringstream stream;
        stream << val;
        return stream.str();
    }

    //------------------------------------------------------------------------------
    //
    // Material execution code
    //
    //------------------------------------------------------------------------------

    // Helper function to create PTX source code for a non-empty 32-bit value array.
    void printArrayU32(std::string& str, std::string const& name, unsigned count, std::string const& content);

    // Helper function to create PTX source code for a non-empty function pointer array.
    void printArrayFunc(std::string& str, std::string const& name, unsigned count, std::string const& content);

    // Generate PTX array containing the references to all generated functions.
    std::string generateFuncArrayPtx(const std::vector<mi::base::Handle<const mi::neuraylib::ITarget_code> >& targetCodes);

    // Build a linked CUDA kernel containing our kernel and all the generated code, making it
    // available to the kernel via an added "mdl_expr_functions" array.
    CUmodule buildLinkedKernel(
        std::vector<mi::base::Handle<const mi::neuraylib::ITarget_code> > const& targetCodes,
        const char* ptxFile,
        const char* kernelFunctionName,
        CUfunction* outKernelFunction);

    class MdlCudaLinker
    {
    public:
        void submitTargetCode(const mi::base::Handle<const mi::neuraylib::ITarget_code>& targetCode, const std::string& materialName);
        int  getMdlFunctionIndices(const std::string& material);
		void link();
        CUfunction& getKernelFunction();

        std::string ptxFile; //input
        std::string kernelFunctionName; //input
    private:
        std::map<std::string, int> matNametoTargetCode;
        std::vector<mi::base::Handle<const mi::neuraylib::ITarget_code> > m_targetCodes;
        CUmodule    outModule;
        CUfunction  outKernelFunction = nullptr;
        bool        isDirty = true;
    };

    MdlCudaLinker& getMdlCudaLinker();
}