#pragma once
#include<torch/torch.h>

#ifdef snprintf
#undef snprintf
#endif

#include <json/json.hpp>
#include <tiny-cuda-nn/cpp_api.h>

namespace torchTcnn
{
	class NativeModule {
	public:
		NativeModule() = default;

		NativeModule(tcnn::cpp::Module* module) : m_module{ module } {}

		std::tuple<tcnn::cpp::Context, torch::Tensor> fwd(torch::Tensor input, torch::Tensor params);

		std::tuple<torch::Tensor, torch::Tensor> bwd(const tcnn::cpp::Context& ctx, torch::Tensor input, torch::Tensor params, torch::Tensor output, torch::Tensor dLdOutput);

		std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> bwdBwdInput(const tcnn::cpp::Context& ctx, torch::Tensor input, torch::Tensor params, torch::Tensor dLDdLdInput, torch::Tensor dLdOutput);

		torch::Tensor initialParams(size_t seed);

		uint32_t nInputDims() const;

		uint32_t nParams() const;
		tcnn::cpp::Precision paramPrecision() const;
		c10::ScalarType c10ParamPrecision() const;

		uint32_t nOutputDims() const;
		tcnn::cpp::Precision outputPrecision() const;
		c10::ScalarType c10OutputPrecision() const;

		nlohmann::json hyperparams() const;
		std::string name() const;


	private:
		std::unique_ptr<tcnn::cpp::Module> m_module;
	};

	NativeModule createNetworkWithInputEncoding(uint32_t nInputDims, uint32_t nOutputDims, const nlohmann::json& encoding, const nlohmann::json& network);

	NativeModule createNetwork(uint32_t nInputDims, uint32_t nOutputDims, const nlohmann::json& network);

	NativeModule createEncoding(uint32_t nInputDims, const nlohmann::json& encoding, tcnn::cpp::Precision requestedPrecision);


}
