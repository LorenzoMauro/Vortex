#pragma once
#include <memory>
#include "Layers/GuiLayer.h"
#include "NeuralNetworks/Experiment.h"

namespace vtx {
	namespace graph
	{
		class Renderer;
	}

	class ExperimentsLayer : public Layer {
    public:

        ExperimentsLayer();

        virtual void OnAttach();

        virtual void OnDetach();

        virtual void OnUpdate(float ts);

        virtual void OnUIRender();

        void startNewRender(SamplingTechnique technique);

        void mapeComputation();

        void generateGroundTruth();

        void runCurrentSettingsExperiment();

        void stopExperiment();

        void storeGroundTruth();

    public:
        ExperimentsManager* em;

        CUDABuffer mapeBuffer;

        std::shared_ptr<graph::Renderer> renderer;
        network::Network* network = nullptr;

        bool toggleMisExperiment = true;
        bool toggleBsdfExperiment = true;
    };
};
