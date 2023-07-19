#pragma once
#ifndef SPHERICALGAUSSIAN_H
#define SPHERICALGAUSSIAN_H
#include "NeuralNetworks/tools.h"

namespace vtx::network
{
    class SphericalGaussian {
    public:
        static torch::Tensor sample(const torch::Tensor& loc, const torch::Tensor& scale, torch::Device device) {

            int64_t batchSize = loc.size(0);

            torch::Tensor uniform = torch::rand({ batchSize, 1 }, torch::TensorOptions().device(device));
            torch::Tensor w = 1.0f + torch::log(uniform + (1.0f - uniform) * torch::exp(-2.0f * scale) + EPS) / (scale + EPS);

            torch::Tensor angleUniform = torch::rand({ batchSize, 1 }, torch::TensorOptions().device(device));
            angleUniform = angleUniform * 2.0f * M_PI_F;
            torch::Tensor v = torch::cat({ torch::cos(angleUniform), torch::sin(angleUniform) }, -1);

            const torch::Tensor w_ = torch::sqrt(torch::clamp(1 - w.pow(2), EPS, 1));

            const torch::Tensor x = torch::cat({ w, w_ * v }, -1);

            // Householder rotation
            const torch::Tensor e1 = torch::tensor({ 1.0, 0.0, 0.0 }, loc.options()).expand_as(loc);
            auto       u = e1 - loc;
            u = u / (u.norm(2, 1, true) + EPS);
            auto sample = x - 2 * (x * u).sum(-1, true) * u;

            PRINT_TENSORS("VON MISES SAMPLE", loc, scale, uniform, w, w_, v, x, e1, u, sample);

            return sample;
        }

        static torch::Tensor prob(const torch::Tensor& x, const torch::Tensor& loc, const torch::Tensor& scale)
        {
            torch::Tensor p = 1.0f / (2.0f * M_PI_F * (1.0f - torch::exp(-2.0f * scale))) * torch::exp(scale * (loc * x).sum(-1).unsqueeze(-1) - 1.0f);
            PRINT_TENSORS("VON MISES PROB", x, loc, scale, p);
            return p;
        }

        static torch::Tensor logLikelihood(const torch::Tensor& x, const torch::Tensor& loc, const torch::Tensor& scale) {

            torch::Tensor logP = torch::log(prob(x, loc, scale) + EPS);
            PRINT_TENSORS("VON MISES LOG LIKELIHOOD", x, loc, scale, logP);
            return logP;
        }
    };
}
#endif