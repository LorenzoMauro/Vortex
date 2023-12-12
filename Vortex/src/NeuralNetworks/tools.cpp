#include "tools.h"

#define PRINT_TENSOR_IN_STACK true
//#define PRINT_TENSOR_IN_STACK false
namespace vtx::network
{
    std::vector<torch::Tensor> downloadTensors(const torch::ITensorListRef& tensorList)
    {
        const torch::Tensor concatenated = torch::cat({ tensorList }, /*dim=*/0);
        const int numberOfTensors = tensorList.size();
        std::vector<torch::Tensor> result;
        const torch::Tensor hostTensor = concatenated.to(torch::kCPU);
        for (int i = 0; i < numberOfTensors; ++i)
        {
            result.push_back(hostTensor[i]);
        }
        return result;
    }

    void printTensors(const std::vector<std::string>& names, const torch::ITensorListRef& tensorRefList, const std::string& info)
    {
        VTX_INFO("{}", info);
        int i = 0;
        for (auto& tensor : tensorRefList)
        {
            std::cout << names[i] << "\n" << tensor << std::endl;
            i++;
        }
    }

    std::vector<std::string> splitVariadicNames(const std::string& variadicString)
    {
        std::vector<std::string> result;
        // split string by comma
        std::stringstream ss(variadicString);
        std::string token;
        while (std::getline(ss, token, ','))
        {
            result.push_back(token);
        }
        return result;
    }

    bool checkTensorAnomaly(const torch::Tensor& tensor, const std::string& tensorName, const std::string& fileName, const int& line)
    {
        bool isAnomaly = false;
        if (tensor.isnan().any().item<bool>())
        {
            VTX_ERROR("Tensor " + tensorName + " has NaN values!" + "File: " + fileName + "Line: " + std::to_string(line));
            isAnomaly = true;
        }
        else if (tensor.isinf().any().item<bool>())
        {
            VTX_ERROR("Tensor " + tensorName + " has Inf values!" + "File: " + fileName + "Line: " + std::to_string(line));
            isAnomaly = true;
        }
        return isAnomaly;
    }

    bool checkTensorHasZero(const torch::Tensor& tensor, const std::string& tensorName, const std::string& fileName, const int& line)
    {
        if (tensor.eq(0).any().item<bool>())
        {
	        VTX_ERROR("Tensor " + tensorName + " has zero values!" + "File: " + fileName + "Line: " + std::to_string(line));
			return true;
		}
        return false;
    }

    void copyNetworkParameters(const std::shared_ptr<torch::nn::Module>& sourceNetwork, const std::shared_ptr<torch::nn::Module>& targetNetwork) {
        torch::OrderedDict<std::string, torch::Tensor> sourceParams = sourceNetwork->named_parameters();
        torch::OrderedDict<std::string, torch::Tensor> targetParams = targetNetwork->named_parameters();

        for (auto& pair : sourceParams) {
            if (targetParams.contains(pair.key())) {
                targetParams[pair.key()] = pair.value().clone();
            }
            else
            {
                VTX_WARN("Polyak Update did not find parameter: {}", pair.key());
            }
        }
    }

    void polyakUpdate(const std::shared_ptr<torch::nn::Module>& sourceNetwork, const std::shared_ptr<torch::nn::Module>& targetNetwork, const float& polyakFactor) {
        torch::OrderedDict<std::string, torch::Tensor> sourceParams = sourceNetwork->named_parameters();
        torch::OrderedDict<std::string, torch::Tensor> targetParams = targetNetwork->named_parameters();

        for (auto& pair : sourceParams) {
            if (targetParams.contains(pair.key())) {
                torch::Tensor& sourceTensor = pair.value();
                torch::Tensor& targetTensor = targetParams[pair.key()];
                auto newTensor = sourceTensor * polyakFactor + targetTensor * (1.0f - polyakFactor);
                targetTensor = newTensor.clone();
            }
            else
            {
                VTX_WARN("Polyak Update did not find parameter: {}", pair.key());
            }
        }
    }

    static TensorDebugger* tensorCallStackInstance = nullptr;

    TensorDebugger& TensorDebugger::get()
    {
        if (tensorCallStackInstance == nullptr)
        {
            tensorCallStackInstance = new TensorDebugger();
        }
        return *tensorCallStackInstance;
    }

    std::tuple<bool, bool, bool> hasValues(const torch::Tensor& tensor)
    {
	    const bool hasNan = tensor.isnan().any().item<bool>();
		const bool hasInf = tensor.isinf().any().item<bool>();
		const bool hasZero = tensor.eq(0).any().item<bool>();
		return std::make_tuple(hasNan, hasInf, hasZero);
	}

    void printTensorInfo(const std::string& scope, const std::string& name, const torch::Tensor& tensor, bool printTensor = false, bool gradient = false)
    {
	    const auto [hasNan, hasInf, hasZero] = hasValues(tensor);
        torch::Tensor t = tensor;
        if (gradient)
        {
            t = tensor.grad();
            if(!t.defined()) return;
            std::cout << "GRADIENT ";
		}

        std::cout << "Tensor : " << name << " Size: " << t.sizes() << " Type: " << t.dtype();
		if (hasNan)
		{
            std::cout << " has NaN values!";
            
		}
		if (hasInf)
		{
            std::cout << " has Inf values!";
		}
		if (hasZero)
		{
			std::cout << " has zero values!";
		}
        std::cout << "\n\tScope: " << scope << "\n";
        if (printTensor && (hasInf || hasNan || hasZero))
        {
            if (t.size(0) > 11) {
                std::cout << t.slice(0, 0, 10) << std::endl;
            }
            else {
                std::cout << t << std::endl;
            }
		}
        std::cout << std::endl;
	}

    void printGradientStack()
    {
        VTX_WARN("GRADIENT STACK:");
	    const auto& tensors = TensorDebugger::get().tensors;
		for (int i = 0; i< tensors.size(); i++)
		{
			const auto& [scope, name, tensor] = tensors[i];
			if(tensor.is_leaf() && tensor.grad().defined())
			{
                const torch::Tensor& grad = tensor.grad();
                printTensorInfo(scope, name, grad, false, true);
			}
		}
    }

	bool TensorDebugger::analyzeGradients()
    {
		const auto& tensors = TensorDebugger::get().tensors;
        for (int i = 0; i< tensors.size(); i++)
        {
            const auto& [scope, name, tensor] = tensors[i];
            if(tensor.is_leaf() && tensor.grad().defined())
            {
                const torch::Tensor& grad = tensor.grad();
                auto [hasNan, hasInf, hasZero] = hasValues(grad);
                if (hasNan || hasInf)
                {
                    VTX_ERROR("{} GRADIENT has {} values!", name, (hasNan) ? "NaN" : "Inf");
                    return true;
				}
            }
        }
        return false;
    }

    void TensorDebugger::push(const std::string& scope, const std::string& name, const torch::Tensor& tensor)
    {
        CUDA_SYNC_CHECK();
        if (!tensor.defined()) return;

        get().tensors.emplace_back(scope, name, tensor);
		auto [hasNan, hasInf, hasZero] = hasValues(tensor);
        if (hasNan || hasInf)
        {
            VTX_ERROR("{} has {} values!", name, (hasNan) ? "NaN" : "Inf");
            TensorDebugger::printStack();
            __debugbreak();
        }
    }

    void TensorDebugger::clear()
    {
		get().tensors.clear();
    }

    void TensorDebugger::printStack()
    {
        VTX_WARN("Tensor Call Stack:");
        const auto& tensors = TensorDebugger::get().tensors;
        for(int i=0; i<tensors.size(); i++)
        {
            auto [scope, name, tensor] = tensors[i];
            printTensorInfo(scope, name, tensor, PRINT_TENSOR_IN_STACK, false);
		}
        
    }

    TensorDebugger::~TensorDebugger()
    {
		get().clear();
	    delete tensorCallStackInstance;
    }

}

