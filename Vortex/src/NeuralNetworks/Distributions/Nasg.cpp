#include "Nasg.h"
#include "TransformUtils.h"
#include "NeuralNetworks/NeuralNetworkGraphs.h"

namespace vtx::distribution
{
	torch::Tensor Nasg::normalizationFactor(const torch::Tensor& lambda, const torch::Tensor& a)
	{
		torch::Tensor result = 2.0f * (float)M_PI * (1.0f - torch::exp(-2.0f * lambda)) / (lambda * torch::sqrt(1.0f + a));
		CHECK_TENSOR_HAS_ZERO(result);
		CHECK_TENSOR_ANOMALY(result);
		if(CHECK_TENSOR_HAS_ZERO(lambda))
		{
			PRINT_TENSORS("Normalization", lambda, a, result);
		}
		return result;
	}

	torch::Tensor lambdaAActivation(const torch::Tensor& rawParams, const int& startIndex)
	{
		const torch::Tensor s1 = rawParams.narrow(-1, startIndex, 2);
		torch::Tensor       lambdaA = softplus(s1) + EPS;//torch::exp(s1) + EPS;
		return lambdaA;
	}

	std::tuple<torch::Tensor, torch::Tensor> axisFromTrig(
		const torch::Tensor& cosThetaEuler, const torch::Tensor& sinThetaEuler,
		const torch::Tensor& cosPhiEuler, const torch::Tensor& sinPhiEuler,
		const torch::Tensor& cosPsiEuler, const torch::Tensor& sinPsiEuler
	)
	{
		const torch::Tensor zAxis = TransformUtils::zAxisFromTrig(
			cosThetaEuler, sinThetaEuler,
			cosPhiEuler, sinPhiEuler
		);
		const torch::Tensor xAxis = TransformUtils::xAxisFromTrig(
			cosThetaEuler, sinThetaEuler,
			cosPhiEuler, sinPhiEuler,
			cosPsiEuler, sinPsiEuler
		);

		return { xAxis, zAxis };
	}
	std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> trigonometricParametrization(const torch::Tensor& rawParams)
	{
		const torch::IntArrayRef rawParamsSize = rawParams.sizes();

		// 1. Splitting s0 and s1 from rawParams
		const torch::Tensor s0 = rawParams.narrow(-1, 0, 5);
		// 2. Applying transformation on s0 to get trigonometric parameters
		torch::Tensor trigParams = 2 * sigmoid(s0) - 1;

		const torch::Tensor cosTheta = trigParams.narrow(-1, 0, 1);
		const torch::Tensor sinTheta = torch::sqrt(1 - torch::pow(cosTheta, 2) + EPS);

		// Extract parts of trigParams
		const torch::Tensor cosPhi_sinPhi_cosPsi_sinPsi = trigParams.narrow(-1, 1, 4);

		// Combine tensors, including the newly computed sinTheta
		trigParams = torch::cat({ cosTheta, sinTheta, cosPhi_sinPhi_cosPsi_sinPsi }, -1);
		trigParams = trigParams.view({ rawParamsSize.at(0), rawParamsSize.at(1), 3, 2 });

		// Normalize the trigonometric values so that they refer to real angles

		trigParams = trigParams / linalg_vector_norm(trigParams, 2, -1, true).expand_as(trigParams);
		//trigParams = normalize(trigParams, torch::nn::functional::NormalizeFuncOptions().dim(trigParams.dim() - 1));
		trigParams = trigParams.view({ rawParamsSize.at(0), rawParamsSize.at(1), 6 });

		const torch::Tensor cosThetaEuler = trigParams.narrow(-1, 0, 1);
		const torch::Tensor sinThetaEuler = trigParams.narrow(-1, 1, 1);
		const torch::Tensor cosPhiEuler = trigParams.narrow(-1, 2, 1);
		const torch::Tensor sinPhiEuler = trigParams.narrow(-1, 3, 1);
		const torch::Tensor cosPsiEuler = trigParams.narrow(-1, 4, 1);
		const torch::Tensor sinPsiEuler = trigParams.narrow(-1, 5, 1);
		//PRINT_TENSOR_ALWAYS("Trig", cosThetaEuler, sinThetaEuler, cosPhiEuler, sinPhiEuler, cosPsiEuler, sinPsiEuler);

		auto [xAxis, zAxis] = axisFromTrig(cosThetaEuler, sinThetaEuler, cosPhiEuler, sinPhiEuler, cosPsiEuler, sinPsiEuler);
		const torch::Tensor lambdaA = lambdaAActivation(rawParams, 5);

		return { xAxis, zAxis, lambdaA };
	}

	std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> eulerAngleParametrization(const torch::Tensor& rawParams)
	{
		torch::Tensor s0 = rawParams.narrow(-1, 0, 3);

		s0 = sigmoid(s0);
		const torch::Tensor thetaEuler = s0.narrow(-1, 0, 1) * M_PI;
		const torch::Tensor phiEuler = s0.narrow(-1, 1, 1) * 2 * M_PI;
		const torch::Tensor psiEuler = s0.narrow(-1, 2, 1) * 2 * M_PI;

		const torch::Tensor cosThetaEuler = torch::cos(thetaEuler);
		const torch::Tensor sinThetaEuler = torch::sin(thetaEuler);
		const torch::Tensor cosPhiEuler = torch::cos(phiEuler);
		const torch::Tensor sinPhiEuler = torch::sin(phiEuler);
		const torch::Tensor cosPsiEuler = torch::cos(psiEuler);
		const torch::Tensor sinPsiEuler = torch::sin(psiEuler);
		auto [xAxis, zAxis] = axisFromTrig(cosThetaEuler, sinThetaEuler, cosPhiEuler, sinPhiEuler, cosPsiEuler, sinPsiEuler);
		const torch::Tensor lambdaA = lambdaAActivation(rawParams, 3);

		return { xAxis, zAxis, lambdaA };
	}

	std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> axisAngleParametrization(const torch::Tensor& rawParams)
	{
		// Normalize z using the given approach
		torch::Tensor zAxis = rawParams.narrow(-1, 0, 3) + EPS;
		zAxis = normalize(zAxis, torch::nn::functional::NormalizeFuncOptions().dim(-1));

		torch::Tensor gamma = rawParams.narrow(-1, 3, 1);
		gamma = sigmoid(gamma) * 2.0f * M_PI;
		// Choose an initial orthogonal x using cross product
		torch::Tensor globalX = torch::tensor({ 1.0, 0.0, 0.0 }, zAxis.options()).expand_as(zAxis);
		torch::Tensor xInit = cross(zAxis, globalX, -1);

		// Just to handle the extremely rare case where z might be [0, 1, 0] or very close.
		// Add a tiny bit to x_init to ensure it's not zero.
		xInit = xInit + EPS;
		xInit = normalize(xInit, torch::nn::functional::NormalizeFuncOptions().dim(-1));
		// Normalize x_init again using the same approach

		// Apply Rodrigues' rotation formula
		const torch::Tensor cosGamma = torch::cos(gamma);
		const torch::Tensor sinGamma = torch::sin(gamma);

		const torch::Tensor term1 = xInit * cosGamma;
		const torch::Tensor term2 = cross(zAxis, xInit, -1) * sinGamma;
		const torch::Tensor term3 = zAxis * (sum(zAxis * xInit, -1, true)) * (1 - cosGamma);

		torch::Tensor xAxis = term1 + term2 + term3;
		xAxis = normalize(xAxis, torch::nn::functional::NormalizeFuncOptions().dim(-1));

		auto xDotZ = sum(zAxis * xAxis, -1, true);
		const torch::Tensor lambdaA = lambdaAActivation(rawParams, 4);

		return { xAxis, zAxis, lambdaA };
	}

	torch::Tensor Nasg::finalizeRawParams(const torch::Tensor& rawParams, const network::DistributionType& type)
	{
		CHECK_TENSOR_ANOMALY(rawParams);
		torch::Tensor xAxis, zAxis, lambdaA;
		switch (type)
		{
		case network::D_NASG_TRIG:
			{
				std::tie(xAxis, zAxis, lambdaA) = trigonometricParametrization(rawParams);
			}break;
		case network::D_NASG_ANGLE:
			{
				std::tie(xAxis, zAxis, lambdaA) = eulerAngleParametrization(rawParams);
			}break;
		case network::D_NASG_AXIS_ANGLE:
			{
				std::tie(xAxis, zAxis, lambdaA) = axisAngleParametrization(rawParams);
			}break;
		default: ;
		}
		CHECK_TENSOR_ANOMALY(xAxis);
		CHECK_TENSOR_ANOMALY(zAxis);
		CHECK_TENSOR_ANOMALY(lambdaA);
		torch::Tensor params = torch::cat({ xAxis, zAxis, lambdaA }, -1);
		return params;
	}

	std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> Nasg::splitParams(
		const torch::Tensor& params)
	{
		torch::Tensor xAxis  = params.narrow(-1, 0, 3);
		torch::Tensor zAxis  = params.narrow(-1, 3, 3);
		torch::Tensor lambda = params.narrow(-1, 6, 1);
		torch::Tensor a      = params.narrow(-1, 7, 1);

		CHECK_TENSOR_ANOMALY(xAxis);
		CHECK_TENSOR_ANOMALY(zAxis);
		CHECK_TENSOR_ANOMALY(lambda);
		CHECK_TENSOR_ANOMALY(a);
		return { xAxis, zAxis, lambda, a};
	}

	torch::Tensor Nasg::prob(
		const torch::Tensor& sample,
		const torch::Tensor& zAxis,
		const torch::Tensor& xAxis,
		const torch::Tensor& lambda,
		const torch::Tensor& a
	)
	{
		const torch::Tensor vDotZ = (sample * zAxis).sum(-1, true).clamp(-1.0f + EPS, 1.0f-EPS);
		const torch::Tensor vDotX       = (sample * xAxis).sum(-1, true);
		const torch::Tensor scaleValue  = (vDotZ + 1.0f) / 2.0f;
		const torch::Tensor denominator = 1.0f - torch::pow(vDotZ, 2.0f);
		const torch::Tensor powerValue  = a * torch::pow(vDotX, 2.0f) / denominator;

		const torch::Tensor pow1 = torch::pow(scaleValue, 1.0f + powerValue);
		const torch::Tensor pow2 = torch::pow(scaleValue, powerValue);
		const torch::Tensor g = torch::exp(2.0f * lambda * (pow1 - 1.0f)) * pow2;
		const torch::Tensor normalization = normalizationFactor(lambda, a);
		torch::Tensor       finalResult   = g / normalization;

		//PRINT_TENSOR_ALWAYS("Nasg Prob computation:", sample, zAxis, xAxis, lambda, a, vDotZ, vDotX, scaleValue, denominator, powerValue, pow1, pow2, g, normalization, finalResult);
		if(CHECK_TENSOR_ANOMALY(finalResult))
		{
			CHECK_TENSOR_ANOMALY(vDotZ);
			CHECK_TENSOR_ANOMALY(vDotX);
			CHECK_TENSOR_ANOMALY(scaleValue);
			CHECK_TENSOR_ANOMALY(denominator);
			CHECK_TENSOR_ANOMALY(powerValue);
			CHECK_TENSOR_ANOMALY(pow1);
			CHECK_TENSOR_ANOMALY(pow2);
			CHECK_TENSOR_ANOMALY(g);
			VTX_ASSERT_CLOSE(false,"Nasg Prob Anomaly");
		}

		return finalResult;
	}

	torch::Tensor Nasg::sample(
		const torch::Tensor& transform,
		const torch::Tensor& lambda,
		const torch::Tensor& a
	)
	{
		// Sample epsilons
		const torch::Tensor uniform0 = torch::rand_like(lambda);
		const torch::Tensor uniform1 = torch::rand_like(lambda);
		const torch::Tensor uniform2 = torch::rand_like(lambda);

		// Map epsilons
		const torch::Tensor s   = torch::exp(-2.0f * lambda) + uniform0 * (1.0f - torch::exp(-2.0f * lambda));
		const torch::Tensor rho = -M_PI_2 + uniform1 * M_PI;

		// To Sample Both the East and West Hemispheres by using the sign function
		const torch::Tensor modifiedUniform2 = (0.5f + 0.5f * sign(uniform2 - 0.5f)) * M_PI;

		// Compute theta and phi based on sampling
		const torch::Tensor phi   = torch::atan(torch::sqrt(1.0f + a) * torch::tan(rho)) + modifiedUniform2; // 0 to 2PI
		const torch::Tensor theta = torch::acos(2.0f * torch::pow(torch::log(s) / (2.0f * lambda) + 1.0f,(1.0f + a - a * torch::pow(torch::cos(rho), 2.0f)) / (1.0f + a)) -1.0f); // 0 to PI

		const torch::Tensor v = TransformUtils::cartesianFromSpherical(theta, phi);

		PRINT_TENSOR_SIZE_ALWAYS(transform);
		torch::Tensor result = matmul(v.unsqueeze(2), transform.unsqueeze(1)).squeeze(2);

		if(CHECK_TENSOR_ANOMALY(result))
		{
			CHECK_TENSOR_ANOMALY(uniform0);
			CHECK_TENSOR_ANOMALY(uniform1);
			CHECK_TENSOR_ANOMALY(uniform2);
			CHECK_TENSOR_ANOMALY(s);
			CHECK_TENSOR_ANOMALY(rho);
			CHECK_TENSOR_ANOMALY(modifiedUniform2);
			CHECK_TENSOR_ANOMALY(phi);
			CHECK_TENSOR_ANOMALY(theta);
			CHECK_TENSOR_ANOMALY(v);
			VTX_ASSERT_CLOSE(false,"Nasg Sample Anomaly");
		}
		return result;
	}

	torch::Tensor Nasg::prob(
		const torch::Tensor& sample,
		const torch::Tensor& params
	)
	{
		auto          [xBasisVector, zBasisVector, lambda, a] = splitParams(params);
		torch::Tensor result                                  = prob(sample, zBasisVector, xBasisVector, lambda, a);
		return result;
	}

	torch::Tensor Nasg::sample(
		const torch::Tensor& params
	)
	{
		auto [xBasisVector, zBasisVector, lambda, a] = splitParams(params);
		const torch::Tensor yAxis = cross(zBasisVector, xBasisVector, -1);
		const torch::Tensor transform = torch::cat({xBasisVector, yAxis, zBasisVector}, -1);

		torch::Tensor result = sample(transform, lambda, a);
		return result;
	}

	void Nasg::setGraphData(
		const torch::Tensor& params,
		const torch::Tensor& mixtureWeights,
		network::GraphsData& graphData,
		const bool           isTraining,
		const int            depth)
	{
		auto [xBasisVector, zBasisVector, lambda, a] = splitParams(params);
		const torch::Tensor lambdaWeightedMean = ((lambda * mixtureWeights.unsqueeze(-1)).sum(1)).mean();
		const torch::Tensor aWeightedMean = ((a * mixtureWeights.unsqueeze(-1)).sum(1)).mean();
		const std::vector<torch::Tensor> hostTensors = network::downloadTensors(
			{
				lambdaWeightedMean.unsqueeze(-1),
				aWeightedMean.unsqueeze(-1)
			}
		);

		if (isTraining)
		{
			graphData.addData(network::G_NASG_T_LAMBDA, hostTensors[0].item<float>());
			graphData.addData(network::G_NASG_T_A, hostTensors[1].item<float>());
		}
		else
		{
			graphData.addData(network::G_NASG_I_LAMBDA, hostTensors[0].item<float>(), depth);
			graphData.addData(network::G_NASG_I_A, hostTensors[1].item<float>(), depth);
		}
	}
}
