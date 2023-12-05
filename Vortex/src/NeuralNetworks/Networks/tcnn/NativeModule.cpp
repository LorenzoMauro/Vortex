#include "NativeModule.h"
#include <tiny-cuda-nn/cpp_api.h>
#include <c10/cuda/CUDAGuard.h>

#include "NeuralNetworks/tools.h"

#define STRINGIFY(x) #x
#define STR(x) STRINGIFY(x)
#define FILE_LINE __FILE__ ":" STR(__LINE__)
#define CHECK_THROW(x) \
	do { if (!(x)) throw std::runtime_error(std::string(FILE_LINE " check failed " #x)); } while(0)

#define CHECK_INPUT(x) CHECK_THROW(x.device().is_cuda()); CHECK_THROW(x.is_contiguous())

namespace torchTcnn
{
	c10::ScalarType torch_type(tcnn::cpp::Precision precision) {
		switch (precision) {
		case tcnn::cpp::Precision::Fp32: return torch::kFloat32;
		case tcnn::cpp::Precision::Fp16: return torch::kHalf;
		default: throw std::runtime_error{"Unknown precision tcnn->torch"};
		}
	}

	void* void_data_ptr(torch::Tensor& tensor) {
		switch (tensor.scalar_type()) {
		case torch::kFloat32: return tensor.data_ptr<float>();
		case torch::kHalf: return tensor.data_ptr<torch::Half>();
		default: throw std::runtime_error{"Unknown precision torch->void"};
		}
	}

	std::tuple<tcnn::cpp::Context, torch::Tensor> NativeModule::fwd(torch::Tensor input, torch::Tensor params) {
		CHECK_INPUT(input);
		CHECK_INPUT(params);

		// Types
		CHECK_THROW(input.scalar_type() == torch::kFloat32);
		CHECK_THROW(params.scalar_type() == c10ParamPrecision());

		// Sizes
		CHECK_THROW(input.size(1) == nInputDims());
		CHECK_THROW(params.size(0) == nParams());

		// Device
		at::Device device = input.device();
		CHECK_THROW(device == params.device());

		const at::cuda::CUDAGuard device_guard{device};
		cudaStream_t stream = at::cuda::getCurrentCUDAStream();

		uint32_t batch_size = input.size(0);

		torch::Tensor output = torch::empty({ batch_size, nOutputDims() }, torch::TensorOptions().dtype(c10OutputPrecision()).device(device));

		tcnn::cpp::Context ctx;
		if (!input.requires_grad() && !params.requires_grad()) {
			m_module->inference(stream, batch_size, input.data_ptr<float>(), void_data_ptr(output), void_data_ptr(params));
		}
		else {
			ctx = m_module->forward(stream, batch_size, input.data_ptr<float>(), void_data_ptr(output), void_data_ptr(params), input.requires_grad());
		}

		return { std::move(ctx), output };
	}

	std::tuple<torch::Tensor, torch::Tensor> NativeModule::bwd(const tcnn::cpp::Context& ctx, torch::Tensor input, torch::Tensor params, torch::Tensor output, torch::Tensor dLdOutput) {
		if (!ctx.ctx) {
			throw std::runtime_error{"Module::bwd: called with invalid context. fwd likely (mistakenly) ran in inference mode."};
		}

		CHECK_INPUT(input);
		CHECK_INPUT(params);
		CHECK_INPUT(output);
		CHECK_INPUT(dLdOutput);

		// Types
		CHECK_THROW(input.scalar_type() == torch::kFloat32);
		CHECK_THROW(params.scalar_type() == c10ParamPrecision());
		CHECK_THROW(output.scalar_type() == c10OutputPrecision());
		CHECK_THROW(dLdOutput.scalar_type() == c10OutputPrecision());

		// Sizes
		CHECK_THROW(input.size(1) == nInputDims());
		CHECK_THROW(output.size(1) == nOutputDims());
		CHECK_THROW(params.size(0) == nParams());
		CHECK_THROW(output.size(0) == input.size(0));
		CHECK_THROW(dLdOutput.size(0) == input.size(0));

		// Device
		at::Device device = input.device();
		CHECK_THROW(device == params.device());
		CHECK_THROW(device == output.device());
		CHECK_THROW(device == dLdOutput.device());

		const at::cuda::CUDAGuard device_guard{device};
		cudaStream_t stream = at::cuda::getCurrentCUDAStream();

		uint32_t batch_size = input.size(0);

		torch::Tensor dL_dinput;
		if (input.requires_grad()) {
			dL_dinput = torch::empty({ batch_size, input.size(1) }, torch::TensorOptions().dtype(torch::kFloat32).device(device));
		}

		torch::Tensor dL_dparams;
		if (params.requires_grad()) {
			dL_dparams = torch::empty({ nParams() }, torch::TensorOptions().dtype(c10ParamPrecision()).device(device));
		}

		if (input.requires_grad() || params.requires_grad()) {
			m_module->backward(
				stream,
				ctx,
				batch_size,
				input.requires_grad() ? dL_dinput.data_ptr<float>() : nullptr,
				void_data_ptr(dLdOutput),
				params.requires_grad() ? void_data_ptr(dL_dparams) : nullptr,
				input.data_ptr<float>(),
				void_data_ptr(output),
				void_data_ptr(params)
			);
		}

		return { dL_dinput, dL_dparams };
	}

	std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> NativeModule::bwdBwdInput(const tcnn::cpp::Context& ctx, torch::Tensor input, torch::Tensor params, torch::Tensor dLDdLdInput, torch::Tensor dLdOutput) {
		// from: dL_ddLdinput
		// to:   dL_ddLdoutput, dL_dparams, dL_dinput

		if (!ctx.ctx) {
			throw std::runtime_error{"Module::bwd_bwd_input: called with invalid context. fwd likely (mistakenly) ran in inference mode."};
		}

		CHECK_INPUT(input);
		CHECK_INPUT(params);
		CHECK_INPUT(dLDdLdInput);
		CHECK_INPUT(dLdOutput);

		// Types
		CHECK_THROW(input.scalar_type() == torch::kFloat32);
		CHECK_THROW(dLDdLdInput.scalar_type() == torch::kFloat32);
		CHECK_THROW(params.scalar_type() == c10ParamPrecision());
		CHECK_THROW(dLdOutput.scalar_type() == c10OutputPrecision());

		// Sizes
		CHECK_THROW(input.size(1) == nInputDims());
		CHECK_THROW(dLdOutput.size(1) == nOutputDims());
		CHECK_THROW(dLDdLdInput.size(1) == nInputDims());
		CHECK_THROW(params.size(0) == nParams());
		CHECK_THROW(dLdOutput.size(0) == input.size(0));
		CHECK_THROW(dLDdLdInput.size(0) == input.size(0));

		// Device
		at::Device device = input.device();
		CHECK_THROW(device == params.device());
		CHECK_THROW(device == dLDdLdInput.device());
		CHECK_THROW(device == dLdOutput.device());

		const at::cuda::CUDAGuard device_guard{device};
		cudaStream_t stream = at::cuda::getCurrentCUDAStream();

		uint32_t batch_size = input.size(0);

		torch::Tensor dL_ddLdoutput;
		if (dLdOutput.requires_grad()) {
			dL_ddLdoutput = torch::zeros({ batch_size, nOutputDims() }, torch::TensorOptions().dtype(c10OutputPrecision()).device(device));
		}

		torch::Tensor dL_dparams;
		if (params.requires_grad()) {
			dL_dparams = torch::zeros({ nParams() }, torch::TensorOptions().dtype(c10ParamPrecision()).device(device));
		}

		torch::Tensor dL_dinput;
		if (input.requires_grad()) {
			dL_dinput = torch::zeros({ batch_size, nInputDims() }, torch::TensorOptions().dtype(torch::kFloat32).device(device));
		}

		if (dLdOutput.requires_grad() || params.requires_grad()) {
			m_module->backward_backward_input(
				stream,
				ctx,
				batch_size,
				dLDdLdInput.data_ptr<float>(),
				input.data_ptr<float>(),
				(params.requires_grad() || input.requires_grad()) ? void_data_ptr(dLdOutput) : nullptr,
				params.requires_grad() ? void_data_ptr(dL_dparams) : nullptr,
				dLdOutput.requires_grad() ? void_data_ptr(dL_ddLdoutput) : nullptr,
				input.requires_grad() ? dL_dinput.data_ptr<float>() : nullptr,
				void_data_ptr(params)
			);
		}

		return { dL_ddLdoutput, dL_dparams, dL_dinput };
	}

	torch::Tensor NativeModule::initialParams(size_t seed) {
		torch::Tensor output = torch::zeros({ nParams() }, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
		m_module->initialize_params(seed, output.data_ptr<float>());
		return output;
	}

	uint32_t NativeModule::nInputDims() const { return m_module->n_input_dims(); }

	uint32_t NativeModule::nParams() const { return (uint32_t)m_module->n_params(); }

	tcnn::cpp::Precision NativeModule::paramPrecision() const { return m_module->param_precision(); }

	c10::ScalarType NativeModule::c10ParamPrecision() const { return torch_type(paramPrecision()); }

	uint32_t NativeModule::nOutputDims() const { return m_module->n_output_dims(); }

	tcnn::cpp::Precision NativeModule::outputPrecision() const { return m_module->output_precision(); }

	c10::ScalarType NativeModule::c10OutputPrecision() const { return torch_type(outputPrecision()); }

	nlohmann::json NativeModule::hyperparams() const { return m_module->hyperparams(); }

	std::string NativeModule::name() const { return m_module->name(); }

	NativeModule createNetworkWithInputEncoding(uint32_t nInputDims, uint32_t nOutputDims, const nlohmann::json& encoding, const nlohmann::json& network) {
		return NativeModule{ tcnn::cpp::create_network_with_input_encoding(nInputDims, nOutputDims, encoding, network) };
	}

	NativeModule createNetwork(uint32_t nInputDims, uint32_t nOutputDims, const nlohmann::json& network) {
		return NativeModule{ tcnn::cpp::create_network(nInputDims, nOutputDims, network) };
	}

	NativeModule createEncoding(uint32_t nInputDims, const nlohmann::json& encoding, tcnn::cpp::Precision requestedPrecision) {
		return NativeModule{ tcnn::cpp::create_encoding(nInputDims, encoding, requestedPrecision) };
	}

}
