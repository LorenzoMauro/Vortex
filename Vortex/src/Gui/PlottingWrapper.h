#pragma once
#include <chrono>
#include <implot.h>
#include <map>
#include <random>
#include <variant>

namespace vtx::gui
{
	using DataType = std::variant<std::vector<int>, std::vector<float>>;

	struct ColorMap
	{
		ColorMap();

		ImU32 getColor();

		ImVec4 getRandomPastelColor();

		unsigned                   seed;
		std::default_random_engine generator;
		std::map<int, ImU32>       colorMap;
		int                        requestedCount = 0;
	};

	struct PlotInfo
	{
		std::vector<DataType> data;
		std::vector<ImU32> color;
		std::vector<std::string> name;

		std::string xLabel = "X";
		std::string yLabel = "Y";
		std::string title = "Plot";

		void addPlot(const DataType& _data, std::string _name = "");
		bool logScale = false;
	};

	void plotLines(const PlotInfo& lines, const ImVec2& quadrantSize = ImGui::GetContentRegionAvail());

	void gridPlot(const std::vector<PlotInfo>& plots);
}