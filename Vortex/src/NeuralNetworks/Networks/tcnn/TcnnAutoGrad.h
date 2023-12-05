#pragma once
#include "NativeModule.h"

namespace torchTcnn
{
	

	class ModuleFunctionBackward : public torch::autograd::Function<ModuleFunctionBackward>
	{
	public:
		static torch::autograd::variable_list forward(torch::autograd::AutogradContext* ctx,
													  torch::autograd::AutogradContext* ctxFwd,
													  const torch::Tensor&              dOutput,
													  const torch::Tensor&              input,
													  const torch::Tensor&              params,
													  const torch::Tensor&              output);

		static torch::autograd::tensor_list backward(torch::autograd::AutogradContext* ctx,
			const torch::autograd::tensor_list& gradOutputs);
	};

	class ModuleFunction : public torch::autograd::Function<ModuleFunction>
	{
	public:
		static torch::Tensor forward(
			torch::autograd::AutogradContext* ctx,
			torchTcnn::NativeModule&          native_tcnn_module,
			torch::Tensor                     input,
			torch::Tensor                     params,
			float                             loss_scale);

		static torch::autograd::tensor_list backward(
			torch::autograd::AutogradContext* ctx,
			const torch::autograd::tensor_list& grad_outputs
		);
	};
}

