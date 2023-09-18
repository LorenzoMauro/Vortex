
#include <string>
#include <vector>

namespace vtx {


    class FileDialogs {
    public:

        enum class FileDialogType {
            Modern,
            Legacy
        };

		static std::string openFileDialog(const std::vector<std::string>& extensions, FileDialogType type = FileDialogType::Modern);
		static std::string saveFileDialog(const std::vector<std::string>& extensions, FileDialogType type = FileDialogType::Modern);

    private:
        static std::string fileDialog(bool open, const std::vector<std::string>& extensions);

        static std::wstring stringToWstring(const std::string& str);

        static std::string wstringToString(const std::wstring& wstr);
		static std::string        openFile(const char* filter);
		static std::string        saveFile(const char* filter);
	};

} // namespace vtx
