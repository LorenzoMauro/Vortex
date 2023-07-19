#pragma once
#ifndef NETWORK_TOOLS_H
#define NETWORK_TOOLS_H

#include "Core/Log.h"
#include <torch/torch.h>
#include "Device/CUDAChecks.h"

#define MAX_CONCENTRATION 50.0f
#define M_PI_F (float)M_PI
#define EPS 1e-6f


namespace vtx::network
{
    std::vector<torch::Tensor> downloadTensors(const torch::ITensorListRef& tensorList);

    void printTensors(const std::vector<std::string>& names, const torch::ITensorListRef& tensorRefList, const std::string& info = "");

    std::vector<std::string> splitVariadicNames(const std::string& variadicString);


#ifdef DEBUG_TENSORS
#define PRINT_TENSORS(info,...) \
	vtx::network::printTensors(vtx::network::splitVariadicNames({#__VA_ARGS__}), {__VA_ARGS__}, info); \
    CUDA_SYNC_CHECK()
#else
#define PRINT_TENSORS(info, ...)
#endif

#ifdef CHECK_ANOMALY
#define ANOMALY_SWITCH torch::autograd::DetectAnomalyGuard detect_anomaly
#else
#define ANOMALY_SWITCH
#endif


    void copyNetworkParameters(const std::shared_ptr<torch::nn::Module>& sourceNetwork, const std::shared_ptr<torch::nn::Module>& targetNetwork);

    void polyakUpdate(const std::shared_ptr<torch::nn::Module>& sourceNetwork, const std::shared_ptr<torch::nn::Module>& targetNetwork, const float& polyakFactor);
}

#endif