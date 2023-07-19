#ifndef NETWORK_GRAPHS_H
#define NETWORK_GRAPHS_H
#pragma once
#include <map>
#include <string>
#include <vector>


namespace vtx::network
{
	enum GraphType
	{
		G_POLICY_LOSS,
		G_Q1_LOSS,
		G_Q2_LOSS,
		G_ALPHA_LOSS,
		G_DATASET_REWARDS,
		G_Q1_VALUES,
		G_Q2_VALUES,
		G_ALPHA_VALUES,
		G_INFERENCE_CONCENTRATION
	};

	static const std::map<GraphType, std::string> graphNames = {
			{ G_POLICY_LOSS, "Policy Loss" },
			{ G_Q1_LOSS, "Q1 Loss" },
			{ G_Q2_LOSS, "Q2 Loss" },
			{ G_ALPHA_LOSS, "Alpha Loss" },
			{ G_DATASET_REWARDS, "Dataset Rewards" },
			{ G_Q1_VALUES, "Q1 Values" },
			{ G_Q2_VALUES, "Q2 Values" },
			{ G_ALPHA_VALUES, "Alpha Values" },
			{ G_INFERENCE_CONCENTRATION, "Inference Concentration" }
	};

	struct GraphsData
	{

		std::map<GraphType, std::vector<float>> graphs;

		void reset(const std::vector<GraphType>& graphTypes)
		{
			for (auto& graphType : graphTypes)
			{
				graphs[graphType].clear();
			}
		}
	};
}

#endif