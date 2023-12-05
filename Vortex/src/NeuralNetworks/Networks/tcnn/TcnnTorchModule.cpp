#include "TcnnTorchModule.h"

#include "TcnnAutoGrad.h"
#include "tcnnUtils.h"
#include "NeuralNetworks/NetworkSettings.h"

namespace torchTcnn
{
    TcnnModuleImpl::TcnnModuleImpl(int _inputDim, int _outputDim, const nlohmann::json& encodingConfig, const nlohmann::json& networkConfig, int _seed) :
        inputDims(_inputDim),
        outputDims(_outputDim),
        seed(_seed),
        nativeTcnnModule(createNetworkWithInputEncoding(inputDims, outputDims, encodingConfig, networkConfig))
    {
        dType = torchPrecision(nativeTcnnModule.paramPrecision());
        seed = _seed;
        params = register_parameter("params", nativeTcnnModule.initialParams(seed), true);
        loss_scale = tcnn::cpp::default_loss_scale(nativeTcnnModule.paramPrecision());
		//overflowLayer = OverflowLayer();
		//register_module("overflowLayer", overflowLayer);
    }

    torch::Tensor TcnnModuleImpl::forward(torch::Tensor x) {
        if (!x.is_cuda()) {
            std::cerr << "Warning: input must be a CUDA tensor, but isn't. This indicates suboptimal performance." << std::endl;
            x = x.to(torch::kCUDA);
        }

        int64_t         batchSize = x.size(0);
        const int       batchSizeGranularity = int(tcnn::cpp::batch_size_granularity());
        const long long paddedBatchSize = (batchSize + batchSizeGranularity - 1) / batchSizeGranularity *
            batchSizeGranularity;

        const torch::Tensor xPadded = (batchSize == paddedBatchSize) ? x : torch::nn::functional::pad(x, torch::nn::functional::PadFuncOptions({ 0, 0, 0, paddedBatchSize - batchSize }));
        torch::Tensor output = ModuleFunction::apply(
            nativeTcnnModule,
            xPadded.to(torch::kFloat).contiguous(),
            params.to(torchPrecision(nativeTcnnModule.paramPrecision())).contiguous(),
            loss_scale
        );
        output = output.slice(0, 0, batchSize).slice(1, 0, outputDims);
		//output = overflowLayer->forward(output);
        return output;
    }

	nlohmann::json getNetworkSettings(vtx::network::PathGuidingNetworkSettings* settings)
	{
		nlohmann::json config;
		config["otype"] = "FullyFusedMLP";
		config["activation"] = "ReLU";
		config["output_activation"] = "None";
		config["n_neurons"] = settings->hiddenDim;
		config["n_hidden_layers"] = settings->numHiddenLayers;
		return config;
	}

	nlohmann::json getEncodingSettings(vtx::network::TcnnEncodingConfig* inputSettings) {
		nlohmann::json config;
		config["otype"] = vtx::network::TcnnEncodingTypeName[static_cast<int>(inputSettings->otype)];

		switch (inputSettings->otype) {
		case vtx::network::TcnnEncodingType::Composite:
			// Assuming you have logic to handle Composite type
			break;
		case vtx::network::TcnnEncodingType::Frequency:
			config["n_frequencies"] = inputSettings->frequencyEncoding.n_frequencies;
			break;
		case vtx::network::TcnnEncodingType::Grid:
			config["type"] = vtx::network::GridTypeName[static_cast<int>(inputSettings->gridEncoding.type)]; // Assuming GridTypeName is similar to TcnnEncodingTypeName
			config["n_levels"]             = inputSettings->gridEncoding.n_levels;
			config["n_features_per_level"] = inputSettings->gridEncoding.n_features_per_level;
			config["log2_hashmap_size"]    = inputSettings->gridEncoding.log2_hashmap_size;
			config["base_resolution"]      = inputSettings->gridEncoding.base_resolution;
			config["per_level_scale"]      = inputSettings->gridEncoding.per_level_scale;
			config["interpolation"]        = vtx::network::InterpolationTypeName[static_cast<int>(inputSettings->gridEncoding.interpolation)]; // Assuming a similar mapping for InterpolationType
			break;
		case vtx::network::TcnnEncodingType::Identity:
			config["scale"] = inputSettings->identityEncoding.scale;
			config["offset"] = inputSettings->identityEncoding.offset;
			break;
		case vtx::network::TcnnEncodingType::OneBlob:
			config["n_bins"] = inputSettings->oneBlobEncoding.n_bins;
			break;
		case vtx::network::TcnnEncodingType::SphericalHarmonics:
			config["degree"] = inputSettings->sphericalHarmonicsEncoding.degree;
			break;
		case vtx::network::TcnnEncodingType::TriangleWave:
			config["n_frequencies"] = inputSettings->triangleWaveEncoding.n_frequencies;
			break;
		default:
			throw std::runtime_error("Unsupported encoding type");
		}

		return config;
	}

	nlohmann::json getCompositeEncodingSettings(vtx::network::InputSettings* inputSettings)
	{
		nlohmann::json config;
		config["otype"] = "Composite";
		nlohmann::json positionEncoding = getEncodingSettings(&inputSettings->tcnnCompositeEncodingConfig.positionEncoding);
		positionEncoding["n_dims_to_encode"] = 3;
		nlohmann::json directionEncoding = getEncodingSettings(&inputSettings->tcnnCompositeEncodingConfig.positionEncoding);
		directionEncoding["n_dims_to_encode"] = 3;
		nlohmann::json normalEncoding = getEncodingSettings(&inputSettings->tcnnCompositeEncodingConfig.positionEncoding);
		normalEncoding["n_dims_to_encode"] = 3;
		config["nested"][0] = positionEncoding;
		config["nested"][1] = directionEncoding;
		config["nested"][2] = normalEncoding;
		return config;
	}

	torchTcnn::TcnnModule build(int outputDim, vtx::network::PathGuidingNetworkSettings* settings, vtx::network::InputSettings* inputSettings)
	{
		// Assuming your module requires input dimension, output dimension, encoding and network configuration, and a seed
		int seed = 1337;   // Example seed
		return std::move(torchTcnn::TcnnModule(9, outputDim, getCompositeEncodingSettings(inputSettings), getNetworkSettings(settings), seed));
	}
}
