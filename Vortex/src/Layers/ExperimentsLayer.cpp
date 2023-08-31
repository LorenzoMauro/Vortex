#include "ExperimentsLayer.h"

#include "Core/CustomImGui/CustomImGui.h"
#include "Device/CudaFunctions/cudaFunctions.h"
#include "Device/UploadCode/UploadData.h"
#include "GuiElements/PlottingWrapper.h"
#include "Scene/Nodes/Renderer.h"
#include "ImGuiFileDialog.h"
#include "Scene/Scene.h"
#include "Serialization/Serializer.h"

namespace vtx
{
	ExperimentsLayer::ExperimentsLayer()
	{
		std::shared_ptr<graph::Scene> scene = graph::Scene::getScene();
		renderer = scene->renderer;
	}
	void ExperimentsLayer::OnAttach()
	{
		network = &renderer->waveFrontIntegrator.network;
		em = &network->experimentManager;
	}

	void ExperimentsLayer::OnDetach()
	{
	}

	void ExperimentsLayer::OnUpdate(float ts)
	{

		switch (em->stage)
		{
		case STAGE_REFERENCE_GENERATION:
		{
			generateGroundTruth();
		}
		break;
		case STAGE_MAPE_COMPUTATION:
		{
			mapeComputation();
		}
		break;
		default:
			break;
		}
	}

	void ExperimentsLayer::OnUIRender()
	{
		ImGui::Begin("Experiment");

		const float availableWidth = ImGui::GetContentRegionAvail().x;
		const float plotWidth = availableWidth * 0.7f; // 75% for the plot
		const float settingsWidth = availableWidth * 0.3f; // 25% for the settings

		ImGui::BeginChild("Experiment Plot Child", ImVec2(plotWidth, 0), true);

		gui::PlotInfo MapePlot;
		MapePlot.title = "Mape";
		MapePlot.xLabel = "samples";
		MapePlot.yLabel = "Mape";
		for (auto& experiment : em->experiments)
		{
			if (experiment.rendererSettings.samplingTechnique == S_MIS && !toggleMisExperiment)
			{
				continue;
			}
			if (experiment.rendererSettings.samplingTechnique == S_BSDF && !toggleBsdfExperiment)
			{
				continue;
			}
			if (experiment.displayExperiment)
			{
				MapePlot.addPlot(experiment.mape, experiment.name);
			}
		}

		gui::gridPlot({ MapePlot });

		ImGui::EndChild();


		ImGui::SameLine();  // Position the next child window on the same line to the right


		ImGui::BeginChild("Settings Child", ImVec2(settingsWidth, 0), true);
		ImGui::PushItemWidth(settingsWidth);

		auto stageName = experimentStageNames[em->stage];
		vtxImGui::halfSpaceWidget("Stage", vtxImGui::booleanText, "%s", stageName.c_str());

		vtxImGui::halfSpaceWidget("Current Step", vtxImGui::booleanText, "%d", em->currentExperimentStep);


		vtxImGui::halfSpaceWidget("Max Samples", ImGui::DragInt, (hiddenLabel + "_Max Samples").c_str(), &(em->maxSamples), 1, 0, 10000, "%d", 0);
		vtxImGui::halfSpaceWidget("Width", ImGui::DragInt, (hiddenLabel + "_Max Samples").c_str(), &(em->width), 1, 0, 4000, "%d", 0);
		vtxImGui::halfSpaceWidget("Height", ImGui::DragInt, (hiddenLabel + "_Max Samples").c_str(), &(em->height), 1, 0, 4000, "%d", 0);

		const float halfItemWidth = ImGui::CalcItemWidth() * 0.5f;
		ImVec2 buttonSize = ImVec2(halfItemWidth, 0);
		//ImGui::PushItemWidth(halfItemWidth);

		if (ImGui::Button("Generate Ground Truth", buttonSize))
		{
			em->stage = STAGE_REFERENCE_GENERATION;
		}

		if (em->isGroundTruthReady)
		{
			//ImGui::PushItemWidth(halfItemWidth);
			if (ImGui::Button("Run Current Settings", buttonSize))
			{
				runCurrentSettingsExperiment();
				// To unlock the renderer when the iteration is finished
				renderer->settings.iteration = -1;
				renderer->settings.isUpdated = true;
			}

			ImGui::SameLine();
			//ImGui::PushItemWidth(halfItemWidth);
			if (ImGui::Button("Stop Experiment", buttonSize))
			{
				stopExperiment();
			}
		}


		if (ImGui::Button("Save Experiments", buttonSize))
		{
			vtxImGui::openFileDialog("saveExperiments", "Save Experiments", ".yaml");
		}
		{
			auto [result, filePathName, filePath] = vtxImGui::fileDialog("saveExperiments");
			if (result)
			{
				serializer::serializeExperimentManger(filePathName, *em);
			}
		}
		ImGui::SameLine();

		if (ImGui::Button("Load Experiments", buttonSize))
		{
			vtxImGui::openFileDialog("loadExperiments", "Load Experiments", ".yaml");
		}
		{
			auto [result, filePathName, filePath] = vtxImGui::fileDialog("loadExperiments");
			if (result)
			{
				*em = serializer::deserializeExperimentManager(filePathName);
			}
		}

#ifdef DEBUG_UI
		if (ImGui::Button("Store Ground Truth"))
		{
			vtxImGui::openFileDialog("saveGT", "Save Ground Truth", ".*,.png,.jpg,.bmp,.hdr");
		}
		{
			auto [result, filePathName, filePath] = vtxImGui::fileDialog("saveGT");
			if (result)
			{
				em->saveGroundTruth(filePathName);
			}
		}

		if (ImGui::Button("Load Ground Truth"))
		{
			vtxImGui::openFileDialog("loadGT", "Load Ground Truth", ".*,.png,.jpg,.bmp,.hdr");
		}
		{
			auto [result, filePathName, filePath] = vtxImGui::fileDialog("loadGT");
			if (result)
			{
				em->loadGroundTruth(filePathName);
			}
		}
#endif

		if (ImGui::Button("Bsdf Experiments", buttonSize))
		{
			toggleBsdfExperiment = !toggleBsdfExperiment;
		}

		ImGui::SameLine();

		if (ImGui::Button("Mis Sampling", buttonSize))
		{
			toggleMisExperiment = !toggleMisExperiment;
		}

		std::vector<int> toRemove;
		int i = -1;
		for(auto& experiment : em->experiments)
		{
			i++;
			if (experiment.rendererSettings.samplingTechnique == S_MIS && !toggleMisExperiment)
			{
				continue;
			}
			if (experiment.rendererSettings.samplingTechnique == S_BSDF && !toggleBsdfExperiment)
			{
				continue;
			}
			ImGui::Separator();
			if (ImGui::CollapsingHeader(("Experiment Nr. " + std::to_string(i)).c_str()))
			{
				vtxImGui::halfSpaceWidget("Experiment Nr. ", vtxImGui::booleanText, "%d", i);


				vtxImGui::halfSpaceWidget("Sampling Technique ", vtxImGui::booleanText, "%s", samplingTechniqueNames[experiment.rendererSettings.samplingTechnique]);
				if (experiment.networkSettings.active)
				{
					vtxImGui::halfSpaceWidget("Network ", vtxImGui::booleanText, "%s", network::networkNames[experiment.networkSettings.type]);
				}

				vtxImGui::halfSpaceWidget("Store", ImGui::Checkbox, (hiddenLabel + "_Store" + std::to_string(i)).c_str(), &(experiment.storeExperiment));
				vtxImGui::halfSpaceWidget("Display", ImGui::Checkbox, (hiddenLabel + "_Display" + std::to_string(i)).c_str(), &(experiment.displayExperiment));

				if (ImGui::Button("Delete", buttonSize))
				{
					toRemove.push_back(i);
				}
			}
		}

		for (auto& i : toRemove)
		{
			em->experiments.erase(em->experiments.begin() + i);
			em->currentExperiment = std::min(0, em->currentExperiment - 1);
		}

		ImGui::EndChild();

		ImGui::End();
	}

	void ExperimentsLayer::startNewRender(SamplingTechnique technique)
	{
		renderer->settings.samplingTechnique = technique;
		renderer->settings.iteration = -1;
		renderer->settings.maxSamples = em->maxSamples;
		renderer->waveFrontIntegrator.settings.active = true;
		renderer->settings.isUpdated = true;

		renderer->isSizeLocked = false;
		renderer->resize(em->width, em->height);
		renderer->isSizeLocked = true;

		renderer->camera->lockCamera = false;
		renderer->camera->resize(em->width, em->height);
		renderer->camera->lockCamera = true;

	}

	void ExperimentsLayer::mapeComputation()
	{
		if (em->currentExperiment >= em->experiments.size())
		{
			return;
		}
		Experiment& experiment = em->experiments[em->currentExperiment];
		if (em->currentExperimentStep == 0)
		{
			startNewRender(experiment.rendererSettings.samplingTechnique);

			if (experiment.networkSettings.active)
			{

				renderer->waveFrontIntegrator.network.settings.active= true;

				bool initNetwork = false;
				if (network->settings.type != experiment.networkSettings.type)
				{
					network->settings.type = experiment.networkSettings.type;
					initNetwork = true;
				}
				if (network->isInitialized == false)
				{
					initNetwork = true;
				}
				if (initNetwork)
				{
					network->initNetworks();
				}
				else
				{
					network->reset();
				}
				network::NetworkSettings& networkSettings = network->getNeuralNetSettings();
				networkSettings.inferenceIterationStart = experiment.networkSettings.inferenceIterationStart;
				networkSettings.doInference = true;
				networkSettings.clearOnInferenceStart = false;
			}
			else
			{
				renderer->waveFrontIntegrator.network.settings.active = false;
			}
			em->currentExperimentStep++;
			return;
		}

		const float mape = cuda::computeMape(em->groundTruth, UPLOAD_DATA->frameBufferData.tmRadiance, em->width, em->height);
		experiment.mape.push_back(mape);

		if (renderer->settings.iteration == em->maxSamples)
		{
			em->currentExperiment += 1;
			em->currentExperimentStep = 0;
			return;
		}
		em->currentExperimentStep++;
	}
	void ExperimentsLayer::generateGroundTruth()
	{
		if (em->currentExperimentStep == 0)
		{
			startNewRender(S_MIS);
			renderer->waveFrontIntegrator.network.settings.active = false;
			renderer->waveFrontIntegrator.network.settings.doInference = false;
			renderer->waveFrontIntegrator.network.settings.isUpdated = true;
		}

		if (renderer->settings.iteration == em->maxSamples)
		{
			storeGroundTruth();
			em->stage = STAGE_NONE;
			em->currentExperimentStep = 0;
			return;
		}
		em->currentExperimentStep++;
	}
	void ExperimentsLayer::runCurrentSettingsExperiment()
	{
		em->experiments.emplace_back();

		Experiment& mapeExperiment = em->experiments.back();
		mapeExperiment.rendererSettings = renderer->settings;
		mapeExperiment.networkSettings = network->settings;
		mapeExperiment.wavefrontSettings = renderer->waveFrontIntegrator.settings;
		mapeExperiment.storeExperiment = true;
		em->currentExperiment = em->experiments.size() - 1;
		mapeExperiment.constructName(em->currentExperiment);
		em->currentExperimentStep = 0;
		em->stage = STAGE_MAPE_COMPUTATION;
	}

	void ExperimentsLayer::stopExperiment()
	{
		// HACK
		if (em->stage == STAGE_REFERENCE_GENERATION)
		{
			storeGroundTruth();
		}
		em->stage = STAGE_NONE;
		em->currentExperimentStep = 0;
		renderer->isSizeLocked = false;
		renderer->camera->lockCamera = false;
	}

	void ExperimentsLayer::storeGroundTruth()
	{
		const size_t imageSize = (size_t)em->width * (size_t)em->height * sizeof(math::vec3f);

		em->groundTruthBuffer.resize(imageSize);
		em->groundTruth = em->groundTruthBuffer.castedPointer<math::vec3f>();
		const void* groundTruthRendered = UPLOAD_DATA->frameBufferData.tmRadiance;

		cudaError_t        error = cudaMemcpy((void*)em->groundTruth, groundTruthRendered, imageSize, cudaMemcpyDeviceToDevice);
		em->isGroundTruthReady = true;

		em->currentExperimentStep = 0;
	}
}

