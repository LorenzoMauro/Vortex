#pragma once
#include <algorithm>
#include <vector>
#include <string>
#include "backends/imgui_impl_vulkan.h"
#include "IconsFontAwesome6.h"
#include "Walnut/Application.h"
#include <glm/gtc/matrix_transform.hpp>


namespace Utils {
	
	static uint32_t ConvertToRGBA(glm::vec4 color) {
		// Convert a vec4 color to a uint32_t color
		color = clamp(color, 0.0f, 1.0f);
		uint8_t r = (uint8_t)(color.r * 255.0f);
		uint8_t g = (uint8_t)(color.g * 255.0f);
		uint8_t b = (uint8_t)(color.b * 255.0f);
		uint8_t a = (uint8_t)(color.a * 255.0f);
		uint32_t return_color = (a << 24) | (b << 16) | (g << 8) | r;
		return return_color;
	}

	static std::string FindAvailableName(std::string Name, std::vector<std::string> Names) {
		std::vector<int> usedSuffixes;
		
		for (std::string s : Names) {
			if (s == Name) {
				usedSuffixes.push_back(0);
			}
			else if (s.find(Name + ".") == 0) {
				// The name is followed by a period, so check if it is followed by a 3-digit number
				std::string suffix = s.substr(Name.size() + 1);
				if (suffix.size() >= 3 && suffix.find_first_not_of("0123456789") == std::string::npos) {
					// The suffix is a 3-digit number, so increment sameNameCount
					usedSuffixes.push_back(std::stoi(suffix));
				}
			}
		}

		//find first available suffix
		std::sort(usedSuffixes.begin(), usedSuffixes.end());
		int suffix = usedSuffixes.size();
		for (int i = 0; i < usedSuffixes.size(); i++) {
			if (usedSuffixes[i] != i) {
				suffix = i;
				break;
			}
		}
		
		if (suffix == 0) {
			return Name;
		}
		else {
			std::string formattedCount = std::to_string(suffix);
			while (formattedCount.size() < 3) {
				formattedCount = "0" + formattedCount;
			}
			return Name + "." + formattedCount;
		}
	}

	static void AddIcons(char* path) {
		ImGuiIO& io = ImGui::GetIO();

		ImFontConfig config;
		config.MergeMode = true;
		config.GlyphMinAdvanceX = 13.0f; // Use if you want to make the icon monospaced
		static const ImWchar icon_ranges[] = { ICON_MIN_FA, ICON_MAX_FA, 0 };
		io.Fonts->AddFontFromFileTTF(path, 13.0f, &config, icon_ranges);

		// Upload Fonts
		{
			
			VkCommandBuffer command_buffer = Walnut::Application::GetCommandBuffer(true);
			VkDevice g_Device = Walnut::Application::GetDevice();
			ImGui_ImplVulkan_CreateFontsTexture(command_buffer);
			Walnut::Application::FlushCommandBuffer(command_buffer);
			auto err = vkDeviceWaitIdle(g_Device);
			check_vk_result(err);
			ImGui_ImplVulkan_DestroyFontUploadObjects();
		}
	}
	
}