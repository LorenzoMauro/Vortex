#include "tools.h"

namespace vtx::network
{
    std::vector<torch::Tensor> downloadTensors(const torch::ITensorListRef& tensorList)
    {
        const torch::Tensor concatenated = torch::cat({ tensorList }, /*dim=*/0);
        const int numberOfTensors = tensorList.size();
        std::vector<torch::Tensor> result;
        const torch::Tensor hostTensor = concatenated.to(torch::kCPU);
        for (int i = 0; i < numberOfTensors; ++i)
        {
            result.push_back(hostTensor[i]);
        }
        return result;
    }

    void printTensors(const std::vector<std::string>& names, const torch::ITensorListRef& tensorRefList, const std::string& info)
    {
        VTX_INFO("{}", info);
        int i = 0;
        for (auto& tensor : tensorRefList)
        {
            std::cout << names[i] << "\n" << tensor << std::endl;
            i++;
        }
    }

    std::vector<std::string> splitVariadicNames(const std::string& variadicString)
    {
        std::vector<std::string> result;
        // split string by comma
        std::stringstream ss(variadicString);
        std::string token;
        while (std::getline(ss, token, ','))
        {
            result.push_back(token);
        }
        return result;
    }

    void copyNetworkParameters(const std::shared_ptr<torch::nn::Module>& sourceNetwork, const std::shared_ptr<torch::nn::Module>& targetNetwork) {
        torch::OrderedDict<std::string, torch::Tensor> sourceParams = sourceNetwork->named_parameters();
        torch::OrderedDict<std::string, torch::Tensor> targetParams = targetNetwork->named_parameters();

        for (auto& pair : sourceParams) {
            if (targetParams.contains(pair.key())) {
                targetParams[pair.key()] = pair.value().clone();
            }
            else
            {
                VTX_WARN("Polyak Update did not find parameter: {}", pair.key());
            }
        }
    }

    void polyakUpdate(const std::shared_ptr<torch::nn::Module>& sourceNetwork, const std::shared_ptr<torch::nn::Module>& targetNetwork, const float& polyakFactor) {
        torch::OrderedDict<std::string, torch::Tensor> sourceParams = sourceNetwork->named_parameters();
        torch::OrderedDict<std::string, torch::Tensor> targetParams = targetNetwork->named_parameters();

        for (auto& pair : sourceParams) {
            if (targetParams.contains(pair.key())) {
                torch::Tensor& sourceTensor = pair.value();
                torch::Tensor& targetTensor = targetParams[pair.key()];
                auto newTensor = sourceTensor * polyakFactor + targetTensor * (1.0f - polyakFactor);
                targetTensor = newTensor.clone();
            }
            else
            {
                VTX_WARN("Polyak Update did not find parameter: {}", pair.key());
            }
        }
    }

}

