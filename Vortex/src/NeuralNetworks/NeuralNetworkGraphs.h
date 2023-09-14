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
		G_INFERENCE_CONCENTRATION,

		G_NGP_T_SAMPLING_FRACTION,
		G_NGP_I_SAMPLING_FRACTION,

		G_NGP_TAU,

		G_NGP_T_TARGET_P,
		G_NGP_T_NEURAL_P,
		G_NGP_T_BSDF_P,
		G_NGP_T_BLENDED_P,

		G_NGP_T_LOSS_Q,
		G_NGP_T_LOSS_BLENDED_Q,
		G_NGP_T_LOSS,

		G_NASG_T_A,
		G_NASG_I_A,
		G_NASG_T_LAMBDA,
		G_NASG_I_LAMBDA,

		G_SPHERICAL_GAUSSIAN_T_K,
		G_SPHERICAL_GAUSSIAN_I_K
	};

	static const std::map<GraphType, std::string> graphNames = {
		{G_POLICY_LOSS, "Policy Loss"},
		{G_Q1_LOSS, "Q1 Loss"},
		{G_Q2_LOSS, "Q2 Loss"},
		{G_ALPHA_LOSS, "Alpha Loss"},
		{G_DATASET_REWARDS, "Dataset Rewards"},
		{G_Q1_VALUES, "Q1 Values"},
		{G_Q2_VALUES, "Q2 Values"},
		{G_ALPHA_VALUES, "Alpha Values"},
		{G_INFERENCE_CONCENTRATION, "Inference Concentration"},

		{G_NGP_T_LOSS,				"NGP Training Loss"},
		{G_NGP_T_TARGET_P,			"NGP Training Target Probability"},
		{G_NGP_T_LOSS_Q,			"NGP Training KL Divergence Q"},
		{G_NGP_T_LOSS_BLENDED_Q,	"NGP Training KL Divergence Blended Q"},
		{G_NGP_T_NEURAL_P,			"NGP Training Q"},
		{G_NGP_T_BLENDED_P,			"NGP Training Blended Q"},
		{G_NGP_T_SAMPLING_FRACTION,	"NGP Training Sampling Fraction"},
		{G_NGP_I_SAMPLING_FRACTION,	"NGP Inference Sampling Fraction"}
	};

	struct GraphsData
	{

		std::map<GraphType, std::vector<float>> graphs;

		void addData(const GraphType& graphType, const float& data, const int& depth=0)
		{
			if(depth == 0)
			{
				graphs[graphType].push_back(data);
			}
			else
			{
				const float prevValue = graphs[graphType].back();
				const float newMeanValue = (prevValue * ((float)depth - 1.0f) + (float)data) / (float)depth;
				// update value
				graphs[graphType].back() = newMeanValue;
			}
		}
		void reset()
		{
			for(auto& [graphType, graph] : graphs)
			{
				graph.clear();
			}
		}
	};
}

#endif