#pragma once
#ifndef TRANSFORM_UTILS_H
#define TRANSFORM_UTILS_H
#ifndef CUDA_INTERFACE
#include <torch/torch.h>
#include "NeuralNetworks/tools.h"
#else
#include "cuda_runtime.h"
#include "Core/Math.h"
#include "Device/DevicePrograms/randomNumberGenerator.h"
#endif

namespace vtx
{
	class TransformUtils
	{
	public:

#ifdef CUDA_INTERFACE
		__forceinline__ __device__ static math::vec3f xAxisFromTrig(
			const float& cosThetaEuler,
			const float& sinThetaEuler,
			const float& sinPhiEuler,
			const float& cosPhiEuler,
			const float& sinPsiEuler,
			const float& cosPsiEuler
		)
		{
			return {
				cosThetaEuler * cosPhiEuler * cosPsiEuler - sinPhiEuler * sinPsiEuler,
				cosThetaEuler * sinPhiEuler * cosPsiEuler + cosPhiEuler * sinPsiEuler,
				-sinThetaEuler * cosPsiEuler
			};
		}

		__forceinline__ __device__ static math::vec3f zAxisFromTrig(
			const float& cosThetaEuler,
			const float& sinThetaEuler,
			const float& sinPhiEuler,
			const float& cosPhiEuler
		)
		{
			return {
				sinThetaEuler * cosPhiEuler,
				sinThetaEuler * sinPhiEuler,
				cosThetaEuler };
		}
#else
		static torch::Tensor xAxisFromTrig(
			const torch::Tensor& cosThetaEuler,
			const torch::Tensor& sinThetaEuler,
			const torch::Tensor& cosPhiEuler,
			const torch::Tensor& sinPhiEuler,
			const torch::Tensor& cosPsiEuler,
			const torch::Tensor& sinPsiEuler);

		static torch::Tensor zAxisFromTrig(
			const torch::Tensor& cosThetaEuler,
			const torch::Tensor& sinThetaEuler,
			const torch::Tensor& cosPhiEuler,
			const torch::Tensor& sinPhiEuler);

		static torch::Tensor transformFromTrig(
			const torch::Tensor& cosThetaEuler,
			const torch::Tensor& sinThetaEuler,
			const torch::Tensor& cosPhiEuler,
			const torch::Tensor& sinPhiEuler,
			const torch::Tensor& cosPsiEuler,
			const torch::Tensor& sinPsiEuler);

		static torch::Tensor transformFromAngles(
			const torch::Tensor& thetaEuler,
			const torch::Tensor& phiEuler,
			const torch::Tensor& psiEuler
		);

		static torch::Tensor cartesianFromSpherical(
			const torch::Tensor& theta,
			const torch::Tensor& phi
		);
#endif
	};
}

#endif