#include "PlottingWrapper.h"

#include "Core/Math.h"

namespace vtx::gui
{
	static ColorMap colorMap;

	void PlotInfo::addPlot(const DataType& _data, std::string _name)
	{
		data.push_back(_data);
		color.push_back(colorMap.getColor());

		_name = (_name.empty()) ? "Plot_" + std::to_string(name.size()) : _name;
		name.push_back(_name);
	}

	ColorMap::ColorMap()
	{
		seed = std::chrono::system_clock::now().time_since_epoch().count();
		generator = std::default_random_engine(seed);
	}

	ImU32 ColorMap::getColor()
	{
		requestedCount++;
		if (colorMap.find(requestedCount) == colorMap.end())
		{
			colorMap[requestedCount] = ImGui::ColorConvertFloat4ToU32(getRandomPastelColor());
		}
		return colorMap[requestedCount];
	}

	ImVec4 ColorMap::getRandomPastelColor()
	{
		const float h = (generator() % 360) / 360.0f; // Generate a random hue
		const float s = (generator() % 100) / 200.0f + 0.5f; // Generate a random saturation between 0.5 and 1.0
		const float v = 0.9f; // Fix value/brightness at a high level
		float r = 0.0f;
		float g = 0.0f;
		float b = 0.0f;
		ImGui::ColorConvertHSVtoRGB(h, s, v, r, g, b);
		return { r,g,b,1.0f };
	}

	void plotLines(const PlotInfo& lines, const ImVec2& quadrantSize)
	{
		ImGui::BeginChild(lines.title.c_str(), quadrantSize, false);
		if (ImPlot::BeginPlot(lines.title.c_str(), quadrantSize)) {
			ImPlotAxisFlags flags = ImPlotAxisFlags_AutoFit;
			flags |= ImPlotAxisFlags_NoLabel;
			ImPlot::SetupAxis(ImAxis_X1, lines.xLabel.c_str(), flags);
			ImPlot::SetupAxis(ImAxis_Y1, lines.yLabel.c_str(), flags);

			for (int i = 0; i < lines.color.size(); i++)
			{
				ImPlot::PushStyleColor(ImPlotCol_Line, lines.color[i]);
				std::visit([&](auto&& arg) {
					using T = std::decay_t<decltype(arg)>;
					if constexpr (std::is_same_v<T, std::vector<int>>)
					{
						ImPlot::PlotLine(lines.name[i].c_str(), arg.data(), arg.size());
					}
					else if constexpr (std::is_same_v<T, std::vector<float>>)
					{
						ImPlot::PlotLine(lines.name[i].c_str(), arg.data(), arg.size());
					}
					}, lines.data[i]);
				ImPlot::PopStyleColor();
			}

			ImPlot::EndPlot();
		}
		ImGui::EndChild();
	}

	void gridPlot(const std::vector<PlotInfo>& plots)
	{
		colorMap.requestedCount = 0;
		const int numberOfPlots = plots.size();
		int xNumberPlots = std::ceil(std::sqrt((float)numberOfPlots));
		int yNumberPlots = std::ceil((float)numberOfPlots / xNumberPlots);


		ImGuiStyle& style = ImGui::GetStyle();
		math::vec2f windowSize = math::vec2f(ImGui::GetContentRegionAvail().x, ImGui::GetContentRegionAvail().y);
		math::vec2f itemSpacing = math::vec2f(style.ItemSpacing.x, style.ItemSpacing.y);
		math::vec2f windowPadding = math::vec2f(style.WindowPadding.x, style.WindowPadding.y);

		math::vec2f quadrantSize = math::vec2f(windowSize) / math::vec2f(xNumberPlots, yNumberPlots) - itemSpacing * math::vec2f(xNumberPlots - 1, yNumberPlots - 1); //- windowPadding 

		//VTX_INFO("Window Size: {}-{}\n Item Spacing: {}-{}\n Window Padding: {}-{}\n Quadrant Size: {}-{}\n", windowSize.x, windowSize.y, itemSpacing.x, itemSpacing.y, windowPadding.x, windowPadding.y, quadrantSize.x, quadrantSize.y);

		for (int y = 0; y < yNumberPlots; y++) {
			for (int x = 0; x < xNumberPlots; x++) {
				int index = x + y * xNumberPlots;

				if (index < numberOfPlots) // make sure index is within the bounds
				{
					math::vec2f actualSize = quadrantSize;
					if (index == numberOfPlots - 1) // if last plot
					{
						int remainingXSpaces = xNumberPlots - (x + 1);
						int remainingYSpaces = yNumberPlots - (y + 1);

						actualSize.x += (float)remainingXSpaces * (quadrantSize.x + itemSpacing.x);
						actualSize.y += (float)remainingYSpaces * (quadrantSize.y + itemSpacing.y);

						//VTX_INFO("Remaining X Spaces: {}\n Remaining Y Spaces: {}\n Quadrant Size = {}-{}", remainingXSpaces, remainingYSpaces, quadrantSize.x, quadrantSize.y);
					}

					plotLines(plots[index], { actualSize.x, actualSize.y });
					if (x < xNumberPlots - 1)
					{
						ImGui::SameLine();
					}
				}
			}
		}
	}
}

