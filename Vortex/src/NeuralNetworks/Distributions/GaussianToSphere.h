#pragma once
#ifndef GUASSIAN_TO_SPHERE_H
#define GUASSIAN_TO_SPHERE_H
#include "NeuralNetworks/tools.h"

namespace vtx::network
{
    class GaussianToSphere
    {
    public:
        static torch::Tensor sampleToAction(const torch::Tensor& sample)
        {
            const torch::Tensor squashedSample = torch::tanh(sample);
            const torch::Tensor angleX = squashedSample.index({ at::indexing::Slice(), 0 });
            const torch::Tensor angleY = squashedSample.index({ at::indexing::Slice(), 1 });
            const torch::Tensor phi = (angleX + 1) * 0.5f * M_PI_F;
            const torch::Tensor theta = (angleY + 1) * M_PI_F;

            const torch::Tensor sinPhi = torch::sin(phi);
            const torch::Tensor x = sinPhi * torch::cos(theta);
            const torch::Tensor y = sinPhi * torch::sin(theta);
            const torch::Tensor z = torch::cos(phi);

            torch::Tensor cartesianCoordinates = torch::stack({ x, y, z }, /*dim=*/-1);

            PRINT_TENSORS("GAUSSIAN TO SPHERE", sample, squashedSample, angleX, angleY, phi, theta, sinPhi, x, y, z, cartesianCoordinates);

            return cartesianCoordinates;
        }

        static torch::Tensor sampleLogLikelihood(const torch::Tensor& sample, const torch::Tensor& mean, const torch::Tensor& logCov)
        {
            //mean and logCov represent a 2d gaussian distribution for spherical angles, evaluate the loglikelihood of the action which is a cartesian direction

            constexpr float epsilon = 1e-6f;

            const torch::Tensor pi = torch::full({ }, M_PI_F);

            const torch::Tensor gaussianLogL = -0.5 * (torch::log(2.0f * pi) + logCov + torch::pow(sample - mean, 2) / (torch::exp(logCov) + epsilon));

            const torch::Tensor squashedSample = torch::tanh(sample);
            const torch::Tensor actionX = squashedSample.index({ at::indexing::Slice(), 0 });
            const torch::Tensor actionY = squashedSample.index({ at::indexing::Slice(), 1 });

            const torch::Tensor absJacobianDeterminant = torch::abs(0.5 * torch::pow(pi, 2) * (1 - torch::pow(actionX, 2)) * (1 - torch::pow(actionY, 2)) * torch::cos(0.5 * pi * actionX));

            const torch::Tensor adjustedLogLikelihood = torch::sum(gaussianLogL, -1) - torch::log(absJacobianDeterminant + epsilon);

            const torch::Tensor output = adjustedLogLikelihood.unsqueeze(-1);

            PRINT_TENSORS("GAUSSIAN TO SPHERE LOG LIKELIHOOD", sample, mean, logCov, pi, gaussianLogL, squashedSample, actionX, actionY, absJacobianDeterminant, adjustedLogLikelihood, output);

            return output;
        }
    };
}


#endif