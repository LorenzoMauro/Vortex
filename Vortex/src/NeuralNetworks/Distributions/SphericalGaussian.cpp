#include "SphericalGaussian.h"

#include "NeuralNetworks/NeuralNetworkGraphs.h"
#include "NeuralNetworks/tools.h"

namespace vtx::distribution
{
    torch::Tensor SphericalGaussian::sample(const torch::Tensor& loc, const torch::Tensor& scale) {

        int64_t batchSize = loc.size(0);

        torch::Device       device = loc.device();
        const torch::Tensor uniform = torch::rand({ batchSize, 1 }, torch::TensorOptions().device(device));
        torch::Tensor       w = 1.0f + torch::log(uniform + (1.0f - uniform) * torch::exp(-2.0f * scale) + EPS) / (scale + EPS);

        torch::Tensor angleUniform = torch::rand({ batchSize, 1 }, torch::TensorOptions().device(device));
        angleUniform = angleUniform * 2.0f * M_PI_F;
        const torch::Tensor v = torch::cat({ torch::cos(angleUniform), torch::sin(angleUniform) }, -1);

        const torch::Tensor w_ = torch::sqrt(clamp(1 - w.pow(2), EPS, 1));

        const torch::Tensor x = torch::cat({ w, w_ * v }, -1);

        // Householder rotation
        const torch::Tensor e1 = torch::tensor({ 1.0, 0.0, 0.0 }, loc.options()).expand_as(loc);
        auto       u = e1 - loc;
        u = u / (u.norm(2, 1, true) + EPS);
        auto sample = x - 2 * (x * u).sum(-1, true) * u;

        PRINT_TENSORS("VON MISES SAMPLE", loc, scale, uniform, w, w_, v, x, e1, u, sample);

        return sample;
    }

    torch::Tensor SphericalGaussian::prob(const torch::Tensor& x, const torch::Tensor& loc, const torch::Tensor& scale)
    {
        torch::Tensor p = scale / (2.0f * M_PI_F * (1.0f - torch::exp(-2.0f * scale))) * torch::exp(scale * ((loc * x).sum(-1).unsqueeze(-1) - 1.0f));
        PRINT_TENSORS("VON MISES PROB", x, loc, scale, p);
        return p;
    }

    torch::Tensor SphericalGaussian::logLikelihood(const torch::Tensor& x, const torch::Tensor& loc, const torch::Tensor& scale) {

        torch::Tensor logP = torch::log(prob(x, loc, scale) + EPS);
        PRINT_TENSORS("VON MISES LOG LIKELIHOOD", x, loc, scale, logP);
        return logP;
    }

    std::tuple<torch::Tensor, torch::Tensor> SphericalGaussian::splitParams(const torch::Tensor& params)
    {
        torch::Tensor loc = params.narrow(params.dim() - 1, 0, 3);
        torch::Tensor scale = params.narrow(params.dim() - 1, 3, 1);
        return { loc, scale };
    }

    torch::Tensor SphericalGaussian::prob(const torch::Tensor& x, const torch::Tensor& params)
    {
        PRINT_TENSORS("VON MISES PROB", x, params);
        auto [loc, scale] = splitParams(params);
        PRINT_TENSORS("VON MISES PROB", x, loc, scale);
        return prob(x, loc, scale);
    }

    torch::Tensor SphericalGaussian::logLikelihood(const torch::Tensor& x, const torch::Tensor& params)
    {
        auto [loc, scale] = splitParams(params);
        return logLikelihood(x, loc, scale);
    }

    torch::Tensor SphericalGaussian::sample(const torch::Tensor& params)
    {
        auto [loc, scale] = splitParams(params);
        return sample(loc, scale);
    }

    torch::Tensor SphericalGaussian::finalizeRawParams(const torch::Tensor& params)
    {
        // The params comes from the network output but are not yet "activated" properly
        // The dimension is Batch Size x Distribution Parameters Count
        // Extracting means and k for all mixtures at once
        auto [loc, scale] = splitParams(params);
        PRINT_TENSORS("VON MISES RAW PARAMS", loc, scale);
        torch::Tensor locNorm = linalg_vector_norm(loc, 2, -1, true);
        loc                 = loc / locNorm;
        scale               = softplus(scale) + EPS;
        PRINT_TENSORS("VON MISES PARAMS", loc, scale);
        // Stacking them along the last dimension
        torch::Tensor elaboratedParams = torch::cat({ loc, scale }, params.dim() - 1);
        return elaboratedParams;
    }

    void SphericalGaussian::setGraphData(
        const torch::Tensor& params,
        const torch::Tensor& mixtureWeights,
        network::GraphsData& graphData,
        const bool isTraining,
        const int depth = 0
    )
    {
        auto [_, scale] = splitParams(params);
        const torch::Tensor scaleWeightedMean = (scale * mixtureWeights.unsqueeze(-1)).sum(1).mean().unsqueeze(-1);
        const std::vector<torch::Tensor> hostTensors = network::downloadTensors(
            {
                scaleWeightedMean
            }
        );

        if(isTraining)
        {
			graphData.addData(network::G_SPHERICAL_GAUSSIAN_T_K, hostTensors[0].item<float>());
        }
        else
        {
	        graphData.addData(network::G_SPHERICAL_GAUSSIAN_I_K, hostTensors[0].item<float>(), depth);
        }
    }

}