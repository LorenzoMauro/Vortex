#pragma once
#include "NativeModule.h"

namespace vtx
{
	namespace network
	{
		namespace config
		{
			struct AuxNetEncodingConfig;
			struct MlpSettings;
			struct NetworkSettings;
            struct EncodingConfig;
            struct MainNetEncodingConfig;
		}
	}
}

/* Input encoding, followed by a neural network.

This module is more efficient than invoking individual `Encoding`
and `Network` modules in sequence.

Takes a `torch.float` input tensor of shape `[:, n_input_dims]` and maps
it to a tensor of shape `[:, n_output_dims]`.

The output tensor can be either of type `torch.float` or `torch.half`,
depending on which performs better on the system.
*/
namespace torchTcnn
{
    struct TcnnModuleImpl : public torch::nn::Module {
        TcnnModuleImpl() = default;

        TcnnModuleImpl(int _inputDim, int _outputDim, const nlohmann::json& encodingConfig, const nlohmann::json& networkConfig, int _seed = 1337);

        torch::Tensor forward(torch::Tensor x);

        torch::Tensor params;
        torch::Dtype dType;
        // Other variables
        int inputDims, outputDims;
        int seed;
        NativeModule nativeTcnnModule;
        float loss_scale;
        //OverflowLayer overflowLayer;

    };
    TORCH_MODULE(TcnnModule);
    nlohmann::json getNetworkSettings(const vtx::network::config::MlpSettings& settings);
    nlohmann::json getEncodingSettings(const vtx::network::config::EncodingConfig& inputSettings);
    nlohmann::json getEncodingSettings(const vtx::network::config::MainNetEncodingConfig& inputSettings);
    nlohmann::json getEncodingSettings(int rawFeatureSize, const vtx::network::config::AuxNetEncodingConfig& inputSettings);


}

