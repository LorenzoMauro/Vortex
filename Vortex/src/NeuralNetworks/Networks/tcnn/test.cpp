#include "test.h"
#include "TcnnTorchModule.h"
#define DEBUG_TESNROS
#include "NeuralNetworks/tools.h"
#include "NeuralNetworks/Networks/TcnnSettings.h"

void testTccnTorch() {
    return;
    torch::autograd::DetectAnomalyGuard detect_anomaly;
    // Configuration setup
	vtx::network::TcnnEncodingConfig encodingConfig;
    encodingConfig.otype = vtx::network::TcnnEncodingType::Identity;
    nlohmann::json config;
    config["encoding"]                         = torchTcnn::getEncodingSettings(&encodingConfig);
    config["network"]["otype"]                 = "FullyFusedMLP";
    config["network"]["activation"]            = "ReLU";
    config["network"]["output_activation"]     = "None";
    config["network"]["n_neurons"]             = 64;
    config["network"]["n_hidden_layers"]       = 2;

    // Assuming your module requires input dimension, output dimension, encoding and network configuration, and a seed
    int inputDim = 3; // Example input dimension
    int outputDim = 1; // Example output dimension
    int seed = 1337;   // Example seed

    // Create the TCNN Module
    torchTcnn::TcnnModule tcnnModule(inputDim, outputDim, config["encoding"], config["network"], seed);
    int batchSize = tcnn::cpp::batch_size_granularity();

    // Set the device to CUDA
    torch::Device device(torch::kCUDA);
    tcnnModule->to(device);

    // Create a test input tensor and target tensor
    auto testInput = torch::rand({ batchSize, inputDim }, torch::dtype(torch::kFloat32)).to(device);
    auto target = torch::rand({ batchSize, outputDim }, torch::dtype(torch::kFloat32)).to(device); // Random target for demo

    // Define a loss function and an optimizer
    auto optimizer = torch::optim::Adam(tcnnModule->parameters(), torch::optim::AdamOptions(1e-3));

    // Training loop
    for (int epoch = 0; epoch < 1000; ++epoch) { // Example: 10 epochs
        // Forward pass
        optimizer.zero_grad();
        auto output = tcnnModule->forward(testInput);
        torch::Tensor loss = target - output;
        loss = loss * loss;
        loss = loss.mean();
        loss.backward();
        optimizer.step();

        // Calculate loss
        //auto loss = lossFunction(output, target);
        //std::cout << "Loss: " << loss << std::endl;
        //std::cout << "Output: " << output << std::endl;
        //std::cout << "Target: " << target << std::endl;
        //std::cout << "Input: " << testInput << std::endl;

        //// Backward pass and optimize
        ///*loss.backward();
        //optimizer.step();*/

        std::cout << "Epoch [" << epoch + 1 << "/10], Loss: " << loss.item<float>() << std::endl;
    }
}
