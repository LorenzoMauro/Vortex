#include "NeuralNetwork.h"

#include "Core/Options.h"
#include "Device/DevicePrograms/LaunchParams.h" 
#include "Device/UploadCode/UploadBuffers.h"

#include "Device/UploadCode/UploadData.h"
#include "Networks/Sac.h"
#include "Networks/Npg.h"


namespace vtx::network
{
	Network::Network()
    {
        settings = getOptions()->networkSettings;
        initNetworks();
    }

    Network::~Network() = default;

    void Network::initNetworks()
    {
        if (settings.type == NetworkType::NT_SAC)
        {
            settings.pathGuidingSettings.produceSamplingFraction = false;
            impl = std::make_unique<Sac>(&settings);
        }
        else if (settings.type == NetworkType::NT_NGP)
        {
            settings.pathGuidingSettings.produceSamplingFraction = true;
            impl = std::make_unique<Npg>(&settings);
		}
		else
		{
			VTX_ERROR("Network settings.type not supported");
        }
        isInitialized = true;
    }

    void Network::train()
    {
        if (settings.active && settings.doTraining)
        {
            impl->train();
            CUDA_SYNC_CHECK();
        }
    }

    bool Network::doInference()
    {
        return UPLOAD_DATA->settings.renderer.iteration >= settings.inferenceIterationStart && settings.active && settings.doInference;
    }

    NetworkSettings& Network::getNeuralNetSettings()
    {
        return settings;
    }

    void Network::inference(const int& depth)
    {
        if(doInference())
        {
            impl->inference(depth);
            CUDA_SYNC_CHECK();
        }
    }

    void Network::reset()
    {
        settings.doTraining = true;
        impl->reset();
    }

    GraphsData& Network::getGraphs()
    {
    	return impl->getGraphs();
	}
}

