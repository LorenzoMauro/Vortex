#pragma once
#ifndef ACTIVATION_MAP_H
#define ACTIVATION_MAP_H
#include <torch/torch.h>

namespace vtx::network
{
    enum ActivationType
    {
        AT_RELU,
        AT_TANH,
        AT_SIGMOID,
        AT_SOFTMAX,
        AT_NONE
    };

    static inline std::map<ActivationType, std::function<torch::Tensor(const torch::Tensor&)>> activationFunction = {
    { ActivationType::AT_RELU, [](const torch::Tensor& t) { return torch::relu(t); } },
    { ActivationType::AT_TANH, [](const torch::Tensor& t) { return torch::tanh(t); } },
    { ActivationType::AT_SIGMOID, [](const torch::Tensor& t) { return torch::sigmoid(t); } },
    { ActivationType::AT_SOFTMAX, [](const torch::Tensor& t) { return torch::softmax(t, -1); } },
    { ActivationType::AT_NONE, [](const torch::Tensor& t) { return t; } }
    };
}

#endif