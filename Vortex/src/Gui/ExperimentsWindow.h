#pragma once
#include <memory>
#include "Gui/GuiWindow.h"
#include "NeuralNetworks/Experiment.h"

namespace vtx {
	namespace graph
	{
		class Renderer;
	}

	class ExperimentsWindow : public Window {
    public:

        ExperimentsWindow();

        virtual void OnUpdate(float ts);

        virtual void renderMainContent() override;
        virtual void renderToolBar() override;

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
        float lossesContentPercentage = 0.75f;
    };
};
