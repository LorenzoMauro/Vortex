#include "FileDialog.h"
#include <Windows.h>
#include <shobjidl.h>

#include "Application.h"
#include "Gui/ViewportWindow.h"
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>

namespace vtx
{

    std::string convertExtensions(const std::vector<std::string>& extensions) {
        std::string filter;
        for (const auto& ext : extensions) {
            filter += "*." + ext + '\0' + "*." + ext + '\0';
        }
        return filter;
    }

    std::string FileDialogs::openFileDialog(const std::vector<std::string>& extensions, FileDialogType type) {
        if (type == FileDialogType::Modern) {
            return fileDialog(true, extensions);
        }
        else {
            return openFile(convertExtensions(extensions).c_str());
        }
    }

    std::string FileDialogs::saveFileDialog(const std::vector<std::string>& extensions, FileDialogType type) {
        if (type == FileDialogType::Modern) {
            return fileDialog(true, extensions);
        }
        else {
            return saveFile(convertExtensions(extensions).c_str());
        }
    }

    std::string FileDialogs::fileDialog(bool open, const std::vector<std::string>& extensions) {
        IFileDialog* pfd = nullptr;
        HRESULT hr;

        std::string resultPath;

        if (open) {
            hr = CoCreateInstance(CLSID_FileOpenDialog, NULL, CLSCTX_ALL, IID_IFileOpenDialog, reinterpret_cast<void**>(&pfd));
        }
        else {
            hr = CoCreateInstance(CLSID_FileSaveDialog, NULL, CLSCTX_ALL, IID_IFileSaveDialog, reinterpret_cast<void**>(&pfd));
        }

        if (SUCCEEDED(hr)) {
            std::vector<COMDLG_FILTERSPEC> filterSpecs;
            for (const auto& ext : extensions) {
                filterSpecs.push_back({ L"File", stringToWstring(ext).c_str() });
            }

            pfd->SetFileTypes(static_cast<UINT>(filterSpecs.size()), filterSpecs.data());
            hr = pfd->Show(NULL);

            if (SUCCEEDED(hr)) {
                IShellItem* pItem = nullptr;
                hr = pfd->GetResult(&pItem);
                pfd->Release();

                if (SUCCEEDED(hr) && pItem) {
                    PWSTR pathWStr;
                    pItem->GetDisplayName(SIGDN_FILESYSPATH, &pathWStr);

                    if (pathWStr) {
                        resultPath = wstringToString(pathWStr);
                        CoTaskMemFree(pathWStr);
                    }
                    pItem->Release();
                }
            }
            else {
                pfd->Release();
            }
        }

        return resultPath;
    }

    std::wstring FileDialogs::stringToWstring(const std::string& str) {
        std::wstring wstr(str.begin(), str.end());
        return wstr;
    }

    std::string FileDialogs::wstringToString(const std::wstring& wstr) {
        std::string str(wstr.begin(), wstr.end());
        return str;
    }

    std::string FileDialogs::openFile(const char* filter)
    {
        OPENFILENAMEA ofn;
        CHAR szFile[260] = { 0 };
        CHAR currentDir[256] = { 0 };
        ZeroMemory(&ofn, sizeof(OPENFILENAME));
        ofn.lStructSize = sizeof(OPENFILENAME);
        ofn.hwndOwner = glfwGetWin32Window((GLFWwindow*)Application::get()->glfwWindow);
        ofn.lpstrFile = szFile;
        ofn.nMaxFile = sizeof(szFile);
        if (GetCurrentDirectoryA(256, currentDir))
            ofn.lpstrInitialDir = currentDir;
        ofn.lpstrFilter = filter;
        ofn.nFilterIndex = 1;
        ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;

        if (GetOpenFileNameA(&ofn) == TRUE)
            return ofn.lpstrFile;

        return std::string();

    }

    std::string FileDialogs::saveFile(const char* filter)
    {
        OPENFILENAMEA ofn;
        CHAR szFile[260] = { 0 };
        CHAR currentDir[256] = { 0 };
        ZeroMemory(&ofn, sizeof(OPENFILENAME));
        ofn.lStructSize = sizeof(OPENFILENAME);
        ofn.hwndOwner = glfwGetWin32Window((GLFWwindow*)Application::get()->glfwWindow);
        ofn.lpstrFile = szFile;
        ofn.nMaxFile = sizeof(szFile);
        if (GetCurrentDirectoryA(256, currentDir))
            ofn.lpstrInitialDir = currentDir;
        ofn.lpstrFilter = filter;
        ofn.nFilterIndex = 1;
        ofn.Flags = OFN_PATHMUSTEXIST | OFN_OVERWRITEPROMPT | OFN_NOCHANGEDIR;

        // Sets the default extension by extracting it from the filter
        ofn.lpstrDefExt = strchr(filter, '\0') + 1;

        if (GetSaveFileNameA(&ofn) == TRUE)
            return ofn.lpstrFile;

        return std::string();
    }


}
