#pragma once
#ifndef NETWORK_TOOLS_H
#define NETWORK_TOOLS_H

#include "Core/Log.h"
#include <torch/torch.h>
#include "Device/CUDAChecks.h"

#define MAX_CONCENTRATION 50.0f
#define M_PI_F (float)M_PI
#define EPS 1e-6f


//#define DEBUG_TENSORS
//#define CHECK_ANOMALY

#ifdef DEBUG
#define CHECK_ANOMALY
#endif
inline static constexpr bool falseValue = false;
namespace vtx::network
{
    std::vector<torch::Tensor> downloadTensors(const torch::ITensorListRef& tensorList);

    void printTensors(const std::vector<std::string>& names, const torch::ITensorListRef& tensorRefList, const std::string& info = "");

    std::vector<std::string> splitVariadicNames(const std::string& variadicString);

    void copyNetworkParameters(const std::shared_ptr<torch::nn::Module>& sourceNetwork, const std::shared_ptr<torch::nn::Module>& targetNetwork);

    void polyakUpdate(const std::shared_ptr<torch::nn::Module>& sourceNetwork, const std::shared_ptr<torch::nn::Module>& targetNetwork, const float& polyakFactor);

    bool checkTensorAnomaly(const torch::Tensor& tensor, const std::string& tensorName = "", const std::string& fileName ="", const int& line =-1);
    bool checkTensorHasZero(const torch::Tensor& tensor, const std::string& tensorName = "", const std::string& fileName ="", const int& line =-1);

#define PRINT_TENSOR_SIZE_ALWAYS(tensor) \
    std::cout << #tensor << "\n" << tensor.sizes() << std::endl; \
	CUDA_SYNC_CHECK()

#define PRINT_TENSOR_ALWAYS(info, ...)\
    vtx::network::printTensors(vtx::network::splitVariadicNames({#__VA_ARGS__}), {__VA_ARGS__}, info); \
	CUDA_SYNC_CHECK()

#define CHECK_TENSOR_ANOMALY_ALWAYS(tensor) \
	vtx::network::checkTensorAnomaly(tensor, #tensor)

#ifdef DEBUG_TENSORS
#define PRINT_TENSORS(info,...) \
	vtx::network::printTensors(vtx::network::splitVariadicNames({#__VA_ARGS__}), {__VA_ARGS__}, info); \
    CUDA_SYNC_CHECK()
#else
#define PRINT_TENSORS(info, ...)
#endif

#ifdef CHECK_ANOMALY
#define ANOMALY_SWITCH torch::autograd::DetectAnomalyGuard detect_anomaly

#define CHECK_TENSOR_ANOMALY(tensor) \
vtx::network::checkTensorAnomaly(tensor, #tensor, __FILE__, __LINE__)

#define CHECK_TENSOR_HAS_ZERO(tensor) \
vtx::network::checkTensorHasZero(tensor, #tensor, __FILE__, __LINE__)

#else
#define ANOMALY_SWITCH

#define CHECK_TENSOR_ANOMALY(tensor) falseValue

#define CHECK_TENSOR_HAS_ZERO(tensor) falseValue

#endif

}

#endif