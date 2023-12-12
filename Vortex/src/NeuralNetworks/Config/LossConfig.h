#pragma once
#include <map>
#include <string>

namespace vtx::network::config
{
	enum LossType
	{
		L_KL_DIV,
		L_KL_DIV_MC_ESTIMATION,
		L_PEARSON_DIV,
		L_PEARSON_DIV_MC_ESTIMATION,
		L_COUNT
	};

	inline static const char* lossNames[] = {
		"KL Divergence",
		"KL Divergence MC Estimation",
		"Pearson Divergence",
		"Pearson Divergence MC Estimation"
	};

	inline static std::map<std::string, LossType> lossNameToEnum =
	{
			{"KL Divergence", L_KL_DIV},
			{"KL Divergence MC Estimation", L_KL_DIV_MC_ESTIMATION},
			{"Pearson Divergence", L_PEARSON_DIV},
			{"Pearson Divergence MC Estimation", L_PEARSON_DIV_MC_ESTIMATION}
	};

	enum LossReduction
	{
		SUM,
		MEAN,
		ABS,
		COUNT
	};

	inline static const char* lossReductionNames[] = {
		"Sum",
		"Mean",
		"Abs"
	};

	inline static std::map<std::string, LossReduction> lossReductionNameToEnum =
	{
			{"Sum", SUM},
			{"Mean", MEAN},
			{"Abs", ABS}
	};
}
