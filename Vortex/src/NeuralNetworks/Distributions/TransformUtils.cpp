#include "TransformUtils.h"

namespace vtx
{
	torch::Tensor TransformUtils::xAxisFromTrig(
		const torch::Tensor& cosThetaEuler,
		const torch::Tensor& sinThetaEuler,
		const torch::Tensor& cosPhiEuler,
		const torch::Tensor& sinPhiEuler,
		const torch::Tensor& cosPsiEuler,
		const torch::Tensor& sinPsiEuler)
	{
		torch::Tensor xBasisVector = torch::cat(
			{
				cosThetaEuler * cosPhiEuler * cosPsiEuler - sinPhiEuler * sinPsiEuler,
				cosThetaEuler * sinPhiEuler * cosPsiEuler + cosPhiEuler * sinPsiEuler,
				-sinThetaEuler * cosPsiEuler
			}, -1);
		return xBasisVector;
	}

	torch::Tensor TransformUtils::zAxisFromTrig(
		const torch::Tensor& cosThetaEuler,
		const torch::Tensor& sinThetaEuler, 
		const torch::Tensor& cosPhiEuler,
		const torch::Tensor& sinPhiEuler)
	{
		torch::Tensor zBasisVector = torch::cat(
			{
				sinThetaEuler * cosPhiEuler,
				sinThetaEuler * sinPhiEuler,
				cosThetaEuler
			}, -1);
		return zBasisVector;
	}
	torch::Tensor TransformUtils::transformFromTrig(
		const torch::Tensor& cosThetaEuler, const torch::Tensor& sinThetaEuler,
		const torch::Tensor& cosPhiEuler, const torch::Tensor& sinPhiEuler,
		const torch::Tensor& cosPsiEuler, const torch::Tensor& sinPsiEuler)
	{
		const torch::Tensor xAxis = xAxisFromTrig(
			cosThetaEuler, sinThetaEuler,
			cosPhiEuler, sinPhiEuler,
			cosPsiEuler, sinPsiEuler);

		const torch::Tensor zAxis = zAxisFromTrig(
			cosThetaEuler, sinThetaEuler,
			cosPhiEuler, sinPhiEuler);

		const torch::Tensor yAxis = torch::cross(zAxis, xAxis, -1);

		torch::Tensor transform = torch::stack({ xAxis, yAxis, zAxis }, -1);
		return transform;
	}

	torch::Tensor TransformUtils::transformFromAngles(
		const torch::Tensor& thetaEuler,
		const torch::Tensor& phiEuler,
		const torch::Tensor& psiEuler)
	{
		const torch::Tensor cosThetaEuler = torch::cos(thetaEuler);
		const torch::Tensor cosPhiEuler = torch::cos(phiEuler);
		const torch::Tensor sinPhiEuler = torch::sin(phiEuler);
		const torch::Tensor cosPsiEuler = torch::cos(psiEuler);
		const torch::Tensor sinPsiEuler = torch::sin(psiEuler);
		const torch::Tensor sinThetaEuler = torch::sin(thetaEuler);
		// Transformation using the Euler angles
		torch::Tensor transform = transformFromTrig(
			cosThetaEuler,
			sinThetaEuler,
			cosPhiEuler,
			sinPhiEuler,
			cosPsiEuler,
			sinPsiEuler);

		return transform;
	}

	// Theta is the polar angle, phi is the azimuthal angle
	torch::Tensor TransformUtils::cartesianFromSpherical(
		const torch::Tensor& theta,
		const torch::Tensor& phi)
	{
		const torch::Tensor sinTheta = torch::sin(theta);
		const torch::Tensor v = torch::stack({
														sinTheta* torch::cos(phi),
														sinTheta* torch::sin(phi),
														torch::cos(theta)
			}, 1);
		return v;
	}
}