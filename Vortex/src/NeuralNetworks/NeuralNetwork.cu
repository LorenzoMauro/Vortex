#include "NeuralNetwork.h"

#include "Device/DevicePrograms/LaunchParams.h" 
#include "Device/UploadCode/UploadBuffers.h"
#include "Device/Wrappers/dWrapper.h"

#include "Device/UploadCode/UploadData.h"
#include "Networks/Sac.h"


namespace vtx::network
{
	Network::Network()
    {
        type = getOptions()->networkType;
        initNetworks();
    }

    Network::~Network() = default;

    void Network::initNetworks()
    {
        if (type == NetworkType::NT_SAC)
        {
            impl = std::make_unique<Sac>();
        }
        else if (type == NetworkType::NT_NGP)
        {
        }
    }

    void Network::train()
    {
        if (UPLOAD_DATA->settings.useNetwork)
        {
            impl->train();
        }
    }

    bool Network::doInference()
    {
        NetworkSettings& settings = impl->getSettings();
        return UPLOAD_DATA->settings.iteration >= settings.inferenceIterationStart && UPLOAD_DATA->settings.useNetwork && settings.doInference;
    }

    NetworkSettings& Network::getNeuralNetSettings()
    {
        return impl->getSettings();
    }

    void Network::inference()
    {
        if(doInference())
        {
            impl->inference();
        }
    }

    void Network::reset()
    {
        impl->reset();
    }

    GraphsData& Network::getGraphs()
    {
    	return impl->getGraphs();
	}
}

