#pragma once
#ifndef VTX_ERROR_TYPES_H
#define VTX_ERROR_TYPES_H

namespace vtx
{
	enum class ErrorType
	{
		MSE,
		MAPE,

		ERROR_TYPE_COUNT
	};

	struct Errors
	{
		float mse = 0.0f;
		float mape = 0.0f;

		float* dMseMap = nullptr;
		float* dMapeMap = nullptr;
	};
	
}

#endif