#pragma once
#include <map>
#include <string>

namespace vtx::network::config
{
	enum SamplingStrategy
	{
		SS_ALL,
		SS_PATHS_WITH_CONTRIBUTION,
		SS_LIGHT_SAMPLES,

		SS_COUNT
	};

	inline static const char* samplingStrategyNames[] = {
		"All",
		"Paths with Contribution",
		"Light Samples"
	};

	inline static std::map<std::string, SamplingStrategy> samplingStrategyNameToEnum =
	{
		{"All", SS_ALL},
		{"Paths with Contribution", SS_PATHS_WITH_CONTRIBUTION},
		{"Light Samples", SS_LIGHT_SAMPLES}
	};

	struct BatchGenerationConfig
	{
		SamplingStrategy strategy          = SS_PATHS_WITH_CONTRIBUTION;
		bool             weightByMis       = true;
		float            lightSamplingProb = 0.0f;
		bool             isUpdated         = true;
		bool              limitToFirstBounce = true;
	};

}
