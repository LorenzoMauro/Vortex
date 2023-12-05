#pragma once
#include <tiny-cuda-nn/cpp_api.h>
#include <torch/torch.h>

inline c10::ScalarType torchPrecision(const tcnn::cpp::Precision precision)
{
	if(precision == tcnn::cpp::Precision::Fp16)
	{
		return torch::kHalf;
	}
	if(precision == tcnn::cpp::Precision::Fp32)
	{
		return torch::kFloat;
	}
	printf("Unknown precision tcnn->torch");
	return {};
}
