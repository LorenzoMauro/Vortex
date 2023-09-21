#pragma once
#ifndef FULLY_CONNECTED_NETWORK_H
#define FULLY_CONNECTED_NETWORK_H
#include <torch/torch.h>
#include "ActivationMap.h"

namespace vtx::network
{
    struct FcNetworkImpl : torch::nn::Module {

        FcNetworkImpl() = default;

        FcNetworkImpl(
			const int64_t        inputDim,
			const int64_t        outputDim,
			const int64_t        hiddenSize,
            const int            numberOfHiddenLayers,
            const ActivationType hiddenActivation = AT_RELU,
            const ActivationType outputActivation = AT_NONE)
        {
            std::vector<int64_t> layerSize;
            std::vector<ActivationType> activationTypes;
            layerSize.push_back(inputDim);

            for (int i = 0; i < numberOfHiddenLayers - 1; i++)
            {
                layerSize.push_back(hiddenSize);
                activationTypes.push_back(hiddenActivation);
            }

            layerSize.push_back(outputDim);
            activationTypes.push_back(outputActivation);

            init(layerSize, activationTypes);
        }

        FcNetworkImpl(
            const std::vector<int64_t>& layerSizes,
            std::vector<ActivationType>& activationTypes
        )
        {
	        init(layerSizes, activationTypes);
        }

        void init(const std::vector<int64_t>& layerSizes,std::vector<ActivationType>& activationTypesIn)
        {
            fcLayers.clear();

            for (int i = 0; i < layerSizes.size() - 1; i++)
            {
                fcLayers.emplace_back(layerSizes[i], layerSizes[i + 1]);
                register_module("fc" + std::to_string(i), fcLayers[i]);
                activationTypes.emplace_back(activationTypesIn[i]);
            }
        }

        torch::Tensor forward(const torch::Tensor& x) {

            torch::Tensor output = x;

            for (int i = 0; i < fcLayers.size(); i++)
            {
                output = activationFunction[activationTypes[i]](fcLayers[i]->forward(output));

            }
            return output;
        }

        std::vector<torch::nn::Linear> fcLayers = {nullptr};
        std::vector<ActivationType> activationTypes;
    };
    TORCH_MODULE(FcNetwork);
}


#endif