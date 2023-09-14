#pragma once
#include <yaml-cpp/yaml.h>

#include "Core/Math.h"
#include "Scene/Nodes/RendererSettings.h"
#include "Serialization/MathSerializer.h"

namespace YAML {

	template<>
	struct convert<vtx::AdaptiveSamplingSettings> {
		static Node encode(const vtx::AdaptiveSamplingSettings& rhs) {
			Node node;
			node["active"] = rhs.active;
			node["noiseKernelSize"] = rhs.noiseKernelSize;
			node["minAdaptiveSamples"] = rhs.minAdaptiveSamples;
			node["minPixelSamples"] = rhs.minPixelSamples;
			node["maxPixelSamples"] = rhs.maxPixelSamples;
			node["albedoNormalNoiseInfluence"] = rhs.albedoNormalNoiseInfluence;
			node["noiseCutOff"] = rhs.noiseCutOff;
			return node;
		}

		static bool decode(const Node& node, vtx::AdaptiveSamplingSettings& rhs) {
			if (!node.IsMap()) {
				return false;
			}

			rhs.active = node["active"].as<bool>();
			rhs.noiseKernelSize = node["noiseKernelSize"].as<int>();
			rhs.minAdaptiveSamples = node["minAdaptiveSamples"].as<int>();
			rhs.minPixelSamples = node["minPixelSamples"].as<int>();
			rhs.maxPixelSamples = node["maxPixelSamples"].as<int>();
			rhs.albedoNormalNoiseInfluence = node["albedoNormalNoiseInfluence"].as<float>();
			rhs.noiseCutOff = node["noiseCutOff"].as<float>();
			rhs.isUpdated = true;

			return true;
		}
	};

	template<>
	struct convert<vtx::FireflySettings> {
		static Node encode(const vtx::FireflySettings& rhs) {
			Node node;
			node["active"] = rhs.active;
			node["threshold"] = rhs.threshold;
			node["kernelSize"] = rhs.kernelSize;
			return node;
		}

		static bool decode(const Node& node, vtx::FireflySettings& rhs) {
			if (!node.IsMap()) {
				return false;
			}

			rhs.active = node["active"].as<bool>();
			rhs.threshold = node["threshold"].as<float>();
			rhs.kernelSize = node["kernelSize"].as<int>();
			rhs.isUpdated = true;
			return true;
		}
	};

	template<>
	struct convert<vtx::DenoiserSettings> {
		static Node encode(const vtx::DenoiserSettings& rhs) {
			Node node;
			node["active"] = rhs.active;
			node["denoiserStart"] = rhs.denoiserStart;
			node["denoiserBlend"] = rhs.denoiserBlend;
			return node;
		}

		static bool decode(const Node& node, vtx::DenoiserSettings& rhs) {
			if (!node.IsMap()) {
				return false;
			}

			rhs.active = node["active"].as<bool>();
			rhs.denoiserStart = node["denoiserStart"].as<int>();
			rhs.denoiserBlend = node["denoiserBlend"].as<float>();
			rhs.isUpdated = true;
			return true;
		}
	};

	template<>
	struct convert<vtx::WavefrontSettings> {
		static Node encode(const vtx::WavefrontSettings& rhs) {
			Node node;
			node["active"] = rhs.active;
			node["fitWavefront"] = rhs.fitWavefront;
			node["optixShade"] = rhs.optixShade;
			node["parallelShade"] = rhs.parallelShade;
			node["longPathPercentage"] = rhs.longPathPercentage;
			node["useLongPathKernel"] = rhs.useLongPathKernel;
			return node;
		}

		static bool decode(const Node& node, vtx::WavefrontSettings& rhs) {
			if (!node.IsMap()) {
				return false;
			}

			rhs.active = node["active"].as<bool>();
			rhs.fitWavefront = node["fitWavefront"].as<bool>();
			rhs.optixShade = node["optixShade"].as<bool>();
			rhs.parallelShade = node["parallelShade"].as<bool>();
			rhs.longPathPercentage = node["longPathPercentage"].as<float>();
			rhs.useLongPathKernel = node["useLongPathKernel"].as<bool>();
			rhs.isUpdated = true;
			return true;
		}
	};

	template<>
	struct convert<vtx::ToneMapperSettings> {
		static Node encode(const vtx::ToneMapperSettings& rhs) {
			Node node;
			node["whitePoint"] = rhs.whitePoint; // Calls the encode function for vtx::math::vec3f
			node["invWhitePoint"] = rhs.invWhitePoint;
			node["colorBalance"] = rhs.colorBalance;
			node["burnHighlights"] = rhs.burnHighlights;
			node["crushBlacks"] = rhs.crushBlacks;
			node["saturation"] = rhs.saturation;
			node["gamma"] = rhs.gamma;
			node["invGamma"] = rhs.invGamma;
			return node;
		}

		static bool decode(const Node& node, vtx::ToneMapperSettings& rhs) {
			if (!node.IsMap()) {
				return false;
			}

			rhs.whitePoint = node["whitePoint"].as<vtx::math::vec3f>();
			rhs.invWhitePoint = node["invWhitePoint"].as<vtx::math::vec3f>();
			rhs.colorBalance = node["colorBalance"].as<vtx::math::vec3f>();
			rhs.burnHighlights = node["burnHighlights"].as<float>();
			rhs.crushBlacks = node["crushBlacks"].as<float>();
			rhs.saturation = node["saturation"].as<float>();
			rhs.gamma = node["gamma"].as<float>();
			rhs.invGamma = node["invGamma"].as<float>();
			rhs.isUpdated = true;
			return true;
		}
	};

	template<>
	struct convert<vtx::DisplayBuffer> {
		static Node encode(const vtx::DisplayBuffer& rhs) {
			return Node(vtx::displayBufferNames[rhs]);
		}

		static bool decode(const Node& node, vtx::DisplayBuffer& rhs) {
			const auto name = node.as<std::string>();
			for (int i = 0; i < vtx::FB_COUNT; i++) {
				if (name == vtx::displayBufferNames[i]) {
					rhs = static_cast<vtx::DisplayBuffer>(i);
					return true;
				}
			}
			return false;
		}
	};

	template<>
	struct convert<vtx::SamplingTechnique> {
		static Node encode(const vtx::SamplingTechnique& rhs) {
			return Node(vtx::samplingTechniqueNames[rhs]);
		}

		static bool decode(const Node& node, vtx::SamplingTechnique& rhs) {
			const auto name = node.as<std::string>();
			for (int i = 0; i < vtx::S_COUNT; i++)
			{
				if (name == vtx::samplingTechniqueNames[i])
				{
					rhs = static_cast<vtx::SamplingTechnique>(i);
					return true;
				}
			}
			return false;
		}
	};

	template<>
	struct convert<vtx::RendererSettings>
	{
		static Node encode(const vtx::RendererSettings& rhs)
		{
			Node node;
			node["iteration"] = rhs.iteration;
			node["maxBounces"] = rhs.maxBounces;
			node["maxSamples"] = rhs.maxSamples;
			node["accumulate"] = rhs.accumulate;
			node["samplingTechnique"] = rhs.samplingTechnique;
			node["displayBuffer"] = rhs.displayBuffer;
			node["minClamp"] = rhs.minClamp;
			node["maxClamp"] = rhs.maxClamp;
			node["useRussianRoulette"] = rhs.useRussianRoulette;
			node["runOnSeparateThread"] = rhs.runOnSeparateThread;
			node["adaptiveSamplingSettings"] = rhs.adaptiveSamplingSettings;
			node["fireflySettings"] = rhs.fireflySettings;
			node["denoiserSettings"] = rhs.denoiserSettings;
			node["toneMapperSettings"] = rhs.toneMapperSettings;
			return node;
		}

		static bool decode(const Node& node, vtx::RendererSettings& rhs)
		{
			if (!node.IsMap())
			{
				return false;
			}
			rhs.iteration = node["iteration"].as<int>();
			rhs.maxBounces = node["maxBounces"].as<int>();
			rhs.maxSamples = node["maxSamples"].as<int>();
			rhs.accumulate = node["accumulate"].as<bool>();
			rhs.samplingTechnique = node["samplingTechnique"].as<vtx::SamplingTechnique>();
			rhs.displayBuffer = node["displayBuffer"].as<vtx::DisplayBuffer>();
			rhs.minClamp = node["minClamp"].as<float>();
			rhs.maxClamp = node["maxClamp"].as<float>();
			rhs.useRussianRoulette = node["useRussianRoulette"].as<bool>();
			rhs.runOnSeparateThread = node["runOnSeparateThread"].as<bool>();
			rhs.adaptiveSamplingSettings = node["adaptiveSamplingSettings"].as<vtx::AdaptiveSamplingSettings>();
			rhs.fireflySettings = node["fireflySettings"].as<vtx::FireflySettings>();
			rhs.denoiserSettings = node["denoiserSettings"].as<vtx::DenoiserSettings>();
			rhs.toneMapperSettings = node["toneMapperSettings"].as<vtx::ToneMapperSettings>();
			rhs.isUpdated = true;
			rhs.isMaxBounceChanged = true;
			return true;
		}
	};

}