#pragma once
#include "LaunchParams.h"
#include "Core/Math.h"
#include "NoiseData.h"
#include "Scene/Nodes/RendererSettings.h"


namespace vtx
{
	void noiseComputation(const LaunchParams* deviceParams, const int& rendererNodeId);

	void switchOutput(const LaunchParams* launchParams, int width, int height, math::vec3f* beauty = nullptr);

	void removeFireflies(const LaunchParams* launchParams, int kernelSize, float threshold, int width, int height);

	void toneMapRadianceKernel(const LaunchParams* launchParams, const int width, const int height, const char* name);
}
