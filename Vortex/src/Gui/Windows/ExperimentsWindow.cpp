#include "ExperimentsWindow.h"

#include "Core/FileDialog.h"
#include "Core/LoadingSaving.h"
#include "Core/CustomImGui/CustomImGui.h"
#include "Device/CudaFunctions/cudaFunctions.h"
#include "Device/UploadCode/DeviceDataCoordinator.h"
#include "Gui/PlottingWrapper.h"
#include "Gui/GuiProvider.h"
#include "Scene/Nodes/Renderer.h"
#include "Scene/Scene.h"
#include "Serialization/Serializer.h"

namespace vtx
{
#define em renderer->waveFrontIntegrator.network.experimentManager
#define net renderer->waveFrontIntegrator.network

	ExperimentsWindow::ExperimentsWindow() : renderer(graph::Scene::get()->renderer)
	{
		name = "Neural Path Guiding Experiments";
	}

	void ExperimentsWindow::OnUpdate(float ts)
	{

		switch (em.stage)
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
	void ExperimentsWindow::mainContent()
	{
		gui::PlotInfo MapePlot;
		MapePlot.title = "Mape";
		MapePlot.xLabel = "samples";
		MapePlot.yLabel = "Mape";

		for (auto& experiment : em.experiments)
		{
			if (experiment.rendererSettings.samplingTechnique == S_MIS && !toggleMisExperiment)
			{
				continue;
			}
			if (experiment.rendererSettings.samplingTechnique == S_BSDF && !toggleBsdfExperiment)
			{
				continue;
			}
			if (experiment.displayExperiment || displayAll)
			{
				MapePlot.addPlot(experiment.mape, experiment.name);
			}
		}

		std::vector<gui::PlotInfo> neuralNetworkPlots = gui::GuiProvider::getPlots(renderer->waveFrontIntegrator.network);
		if (!neuralNetworkPlots.empty())
		{
			const float totalAvailableHeight = ImGui::GetContentRegionAvail().y - resizerSize - ImGui::GetStyle().ItemSpacing.y * 3.0f;// -childPaddingHeight;
			const float MapePlotHeight = totalAvailableHeight * (1.0f - lossesContentPercentage);
			const float LossesPlotHeight = totalAvailableHeight * lossesContentPercentage;

			const float plotsWindowWidth = ImGui::GetContentRegionAvail().x;
			// First child window
			ImGui::BeginChild("Child1", ImVec2(plotsWindowWidth, MapePlotHeight), false);
			{
				gui::gridPlot({ MapePlot });
			}
			ImGui::EndChild();

			// Resizing handle
			vtxImGui::childWindowResizerButton(lossesContentPercentage, resizerSize, false);

			// Second child window
			ImGui::BeginChild("Child2", ImVec2(plotsWindowWidth, LossesPlotHeight), false);
			{
				gui::gridPlot(neuralNetworkPlots);
			}
			ImGui::EndChild();
		}
		else
		{
			gui::gridPlot({ MapePlot });
		}
	}


	void ExperimentsWindow::toolBarContent()
	{
		const float halfItemWidth = ImGui::CalcItemWidth() * 0.5f;
		ImVec2 buttonSize = ImVec2(halfItemWidth, 0);

		if(vtxImGui::halfSpaceCheckbox("Perform Batch Experiments", &performBatchExperiment) && performBatchExperiment)
		{
			em.saveFilePath = FileDialogs::saveFileDialog({ "*.vtx", "*.xml" });
		}

		if(ImGui::Button("Load Experiment Manager"))
		{
			em.saveFilePath = FileDialogs::openFileDialog({ "*.vtx", "*.xml" });
			serializer::deserializeExperimentManager(em.saveFilePath);
			return;
		}

		auto stageName = experimentStageNames[em.stage];
		vtxImGui::halfSpaceWidget("Stage", vtxImGui::booleanText, "%s", stageName.c_str());

		vtxImGui::halfSpaceWidget("Current Step", vtxImGui::booleanText, "%d", em.currentExperimentStep);


		vtxImGui::halfSpaceWidget("GT Max Samples", ImGui::DragInt, (hiddenLabel + "_Max Samples").c_str(), &(em.gtSamples), 1, 0, 10000, "%d", 0);
		vtxImGui::halfSpaceWidget("TEST Max Samples", ImGui::DragInt, (hiddenLabel + "_Max Samples").c_str(), &(em.testSamples), 1, 0, 10000, "%d", 0);
		vtxImGui::halfSpaceWidget("Width", ImGui::DragInt, (hiddenLabel + "_Max Samples").c_str(), &(em.width), 1, 0, 4000, "%d", 0);
		vtxImGui::halfSpaceWidget("Height", ImGui::DragInt, (hiddenLabel + "_Max Samples").c_str(), &(em.height), 1, 0, 4000, "%d", 0);


		if (ImGui::Button("Generate Ground Truth", buttonSize))
		{
			em.stage = STAGE_REFERENCE_GENERATION;
		}

		if (em.isGroundTruthReady)
		{
			//ImGui::PushItemWidth(halfItemWidth);
			if (ImGui::Button("Run Current Settings", buttonSize))
			{
				runCurrentSettingsExperiment();
				// To unlock the renderer when the iteration is finished
				renderer->restart();
			}

			ImGui::SameLine();
			//ImGui::PushItemWidth(halfItemWidth);
			if (ImGui::Button("Stop Experiment", buttonSize))
			{
				stopExperiment();
			}
		}

		if (ImGui::Button("Bsdf Experiments", buttonSize))
		{
			toggleBsdfExperiment = !toggleBsdfExperiment;
		}

		ImGui::SameLine();

		if (ImGui::Button("Mis Sampling", buttonSize))
		{
			toggleMisExperiment = !toggleMisExperiment;
		}

		vtxImGui::halfSpaceCheckbox("Display All Tests", &displayAll);

		vtxImGui::halfSpaceWidget("Experiments in Queue", vtxImGui::booleanText, "%d", em.experimentQueue.size());

		std::vector<int> toRemove;
		int i = -1;
		for (auto& experiment : em.experiments)
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
			ImGui::PushID(i);
			if (ImGui::CollapsingHeader((experiment.name).c_str()))
			{
				gui::GuiProvider::drawEditGui(experiment);

				if (ImGui::Button("Delete", buttonSize))
				{
					toRemove.push_back(i);
				}
			}
			ImGui::PopID();
		}

		for (auto& i : toRemove)
		{
			em.experiments.erase(em.experiments.begin() + i);
			em.currentExperiment = std::min(0, em.currentExperiment - 1);
		}
	}

	void ExperimentsWindow::startNewRender(const SamplingTechnique technique)
	{
		renderer->restart();
		renderer->settings.samplingTechnique = technique;
		renderer->settings.maxSamples = em.testSamples;
		renderer->waveFrontIntegrator.settings.active = true;
		renderer->settings.isUpdated = true;

		renderer->isSizeLocked = false;
		renderer->resize(em.width, em.height);
		renderer->isSizeLocked = true;

		renderer->camera->lockCamera = false;
		renderer->camera->resize(em.width, em.height);
		renderer->camera->lockCamera = true;

	}

	void ExperimentsWindow::mapeComputation()
	{
		if (em.currentExperiment >= em.experiments.size())
		{
			return;
		}
		Experiment& experiment = em.experiments[em.currentExperiment];
		if (em.currentExperimentStep == 0)
		{
			startNewRender(experiment.rendererSettings.samplingTechnique);

			if (experiment.networkSettings.active)
			{

				renderer->waveFrontIntegrator.network.settings.active= true;

				bool initNetwork = false;
				/*if (net.settings.type != experiment.networkSettings.type)
				{
					net.settings.type = experiment.networkSettings.type;
					initNetwork = true;
				}*/
				if (net.isInitialized == false)
				{
					initNetwork = true;
				}
				if (initNetwork)
				{
					net.initNetworks();
				}
				else
				{
					net.reset();
				}
				network::config::NetworkSettings& networkSettings = net.getNeuralNetSettings();
				networkSettings.inferenceIterationStart = experiment.networkSettings.inferenceIterationStart;
				networkSettings.doInference = true;
				networkSettings.clearOnInferenceStart = false;
			}
			else
			{
				renderer->waveFrontIntegrator.network.settings.active = false;
			}
			em.currentExperimentStep++;
			return;
		}

		const float mape = cuda::computeMape(em.groundTruth, onDeviceData->launchParamsData.getHostImage().frameBuffer.tmRadiance, em.width, em.height);
		experiment.mape.push_back(mape);

		if (renderer->settings.iteration == em.testSamples)
		{
			em.currentExperiment += 1;
			em.currentExperimentStep = 0;
			return;
		}
		em.currentExperimentStep++;
	}

	void ExperimentsWindow::generateGroundTruth()
	{
		if (em.currentExperimentStep == 0)
		{
			startNewRender(S_MIS);
			renderer->waveFrontIntegrator.network.settings.active = false;
			renderer->waveFrontIntegrator.network.settings.doInference = false;
			renderer->waveFrontIntegrator.network.settings.isUpdated = true;
		}

		if (renderer->settings.iteration == em.testSamples)
		{
			storeGroundTruth();
			em.stage = STAGE_NONE;
			em.currentExperimentStep = 0;
			return;
		}
		em.currentExperimentStep++;
	}
	void ExperimentsWindow::runCurrentSettingsExperiment()
	{
		em.experiments.emplace_back();

		Experiment& mapeExperiment = em.experiments.back();
		mapeExperiment.rendererSettings = renderer->settings;
		mapeExperiment.networkSettings = net.settings;
		mapeExperiment.wavefrontSettings = renderer->waveFrontIntegrator.settings;
		mapeExperiment.storeExperiment = true;
		em.currentExperiment = em.experiments.size() - 1;
		mapeExperiment.constructName(em.currentExperiment);
		em.currentExperimentStep = 0;
		em.stage = STAGE_MAPE_COMPUTATION;
	}

	void ExperimentsWindow::stopExperiment()
	{
		// HACK
		if (em.stage == STAGE_REFERENCE_GENERATION)
		{
			storeGroundTruth();
		}
		em.stage = STAGE_NONE;
		em.currentExperimentStep = 0;
		renderer->isSizeLocked = false;
		renderer->camera->lockCamera = false;
	}

	void ExperimentsWindow::storeGroundTruth()
	{
		const size_t imageSize = (size_t)em.width * (size_t)em.height * sizeof(math::vec3f);

		em.groundTruthBuffer.resize(imageSize);
		em.groundTruth = em.groundTruthBuffer.castedPointer<math::vec3f>();
		const void* groundTruthRendered = onDeviceData->launchParamsData.getHostImage().frameBuffer.tmRadiance;

		cudaError_t        error = cudaMemcpy((void*)em.groundTruth, groundTruthRendered, imageSize, cudaMemcpyDeviceToDevice);
		em.isGroundTruthReady = true;

		em.currentExperimentStep = 0;
	}


	
}

