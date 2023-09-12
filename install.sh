set -e
trap 'read -p "An error occurred. Press any key to exit."' ERR

#constants
repoRoot=$(pwd)
extFolder=$repoRoot/ext
logFile="$repoRoot/installationLog.txt"
touch $logFile

source ./utilities.sh

#manual dependencies
echo "Before proceeding make sure to have installed the following dependencies:"
echo "- CUDA 11.7: https://developer.nvidia.com/cuda-toolkit-archive"
echo "- CUDA 8.0: https://developer.nvidia.com/cuda-80-ga2-download-archive"
echo "- Clang 12: https://releases.llvm.org/download.html"
echo "- OptiX 7.7: https://developer.nvidia.com/designworks/optix/downloads/legacy"
echo "- LibTorch: https://pytorch.org/get-started/locally/"
echo "For LibTorch Select Windows for the OS, Libtorch for the Package and CUDA 11.7 for the Compute Platform. Then download the Release and Debug Version and extract wherever you want."
echo "The folders you will be prompted to select later are the "libtorch" folder inside the extracted folder for the Release and Debug Binaries"
read -p "Press any key to continue..."

# Check if script is running from the correct location
[ ! -d "./ext" ] && echo "Run this script from the root of the Vortex repository." && exit 1

cudaToolkitRoot=$(checkAndLogStep "cudaToolkitChecked" "cudaToolkitRoot" "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.7" "Please set CUDA 11.7 folder, usually is set to C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.7")
cuda8Path=$(checkAndLogStep "cuda8Checked" "cuda8Path" "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v8.0" "Please set CUDA 8.0 folder, usually is set to C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v8.0")
optixPath=$(checkAndLogStep "optixPathChecked" "optixPath" "C:\ProgramData\NVIDIA Corporation\OptiX SDK 8.0.0" "Please set Optix Path, usually is set to C:\ProgramData\NVIDIA Corporation\OptiX SDK 8.0.0")
clang12Path=$(checkAndLogStep "clang12Checked" "clang12Path" "C:/Program Files (x86)/LLVM" "Please set Clang 12 folder, usually is set to C:/Program Files (x86)/LLVM")
clang12Path="${clang12Path}/bin/clang.exe"
torchPathRelease=$(checkAndLogStep "libTorchReleaseChecked" "torchPathRelease" "" "Please set LibTorch Release folder. The folder path needs to point to the libtorch folder inside the extract folder from the downloaded compressed file.")
torchPathDebug=$(checkAndLogStep "libTorchDebugChecked" "torchPathDebug" "" "Please set LibTorch Debug folder. The folder path needs to point to the libtorch folder inside the extract folder from the downloaded compressed file.")

if ! stepDoneBefore "vcpkgInstallation"; then
    answer=$(askYesOrNo "Do you want to use an existing vcpkg installation? (Yes for existing, No for new)")
    if [[ $answer -eq 6 ]]; then
        vcpkg_dir=$(askForPath "Choose the existing vcpkg directory")
        echo "User selected existing vcpkg directory: $vcpkg_dir"
    
    elif [[ $answer -eq 7 ]]; then
        # If existing vcpkg directory found
        if [ -d "./ext/vcpkg" ]; then
            echo "Existing vcpkg directory found. Updating..."
            cd ./ext/vcpkg
            git pull origin master
            if [[ $? -ne 0 ]]; then
                echo "Failed to update vcpkg. Please resolve any issues manually."
                exit 1
            fi
            cd -
        else
            echo "Cloning vcpkg into ./ext/"
            git clone https://github.com/Microsoft/vcpkg.git ./ext/vcpkg
        fi
        vcpkg_dir="./ext/vcpkg"
        # Additional logic for bootstrap and install
    fi

    vcpkg_dir=$(realpath "$vcpkg_dir")

    echo "vcpkg_dir is:" $vcpkg_dir
    cd $vcpkg_dir
    ./bootstrap-vcpkg.bat  # On Linux: bootstrap-vcpkg.sh

    # Install dependencies for MDL (static)
    ./vcpkg install boost-any boost-uuid --triplet=x64-windows-static
    ./vcpkg install openimageio --triplet=x64-windows-static

    # Install other dependencies (static-md)
    ./vcpkg install imgui[docking-experimental,opengl3-binding,glfw-binding,win32-binding] --triplet=x64-windows-static-md --recurse
    ./vcpkg install spdlog --triplet=x64-windows-static-md
    ./vcpkg install yaml-cpp --triplet=x64-windows-static-md
    ./vcpkg install assimp --triplet=x64-windows-static-md
    ./vcpkg install glfw3 --triplet=x64-windows-static-md
    ./vcpkg install implot --triplet=x64-windows-static-md

    
    echo "vcpkg Directory set to ${vcpkg_dir}"
    echo "vcpkg_dir=$vcpkg_dir" >> $logFile
    markStepDone "vcpkgInstallation"
else
    vcpkg_dir=$(readValueFromLog "$vcpkg_dir")
    echoGreen "Vcpkg and dependencies already installed"
fi

if ! stepDoneBefore "mdlBuild"; then

    # Navigate to MDL-SDK and build
    # Create a build directory if it doesn't exist

    echo "Configuring MDL Cmake"
    mkdir -p $extFolder/MDL-SDK/build

    # Navigate to the new build directory
    cd $extFolder/MDL-SDK/build

    # Run cmake and specify the parent directory as the source directory
    cmake -DCMAKE_TOOLCHAIN_FILE="$vcpkg_dir/scripts/buildsystems/vcpkg.cmake" \
    -DMDL_BUILD_CORE_EXAMPLES=OFF \
    -DMDL_BUILD_DOCUMENTATION=OFF \
    -DMDL_BUILD_SDK_EXAMPLES=OFF \
    -DMDL_ENABLE_CUDA_EXAMPLES=OFF \
    -DMDL_ENABLE_D3D12_EXAMPLES=OFF \
    -DMDL_ENABLE_OPENGL_EXAMPLES=OFF \
    -DMDL_ENABLE_QT_EXAMPLES=OFF \
    -DMDL_ENABLE_VULKAN_EXAMPLES=OFF \
    ..

    # Build the project
    echo "Building MDL"
    cmake --build . --config Release
    cmake --build . --config Release --target INSTALL
    markStepDone "mdlBuild"
else
    echoGreen "MDL-SDK library already built"
fi

echoGreen "Configuring Vortex Cmake.."
cd $repoRoot
mkdir -p $repoRoot/build
cd $repoRoot/build
#rm -rf ./CMakeCache.txt ./CMakeFiles
cmake -DCMAKE_TOOLCHAIN_FILE="$vcpkg_dir/scripts/buildsystems/vcpkg.cmake" \
    -DTORCH_INSTALL_PREFIX_DEBUG="$torchPathDebug" \
    -DTORCH_INSTALL_PREFIX_RELEASE="$torchPathRelease" \
    -DCUDAToolkit_ROOT="$cudaToolkitRoot" \
    -DCLANG_12_PATH="$clang12Path" \
    -DCUDA_8_PATH="$cuda8Path" \
    -DOPTIX77_PATH="$optixPath" \
    ..
echoGreen "Cmake Configuration Completed!"


answer=$(askYesOrNo "Do you want to build Vortex or open Visual Studio? (Yes for building, No for opening Solution)")

if [[ $answer -eq 6 ]]; then
    echoGreen "Building Vortex..."
    cmake --build . --config Release
    echoGreen "Build complete, Launching Vortex."
    ./Vortex/src/Release/Vortex.exe
elif [[ $answer -eq 7 ]]; then
    echoGreen "Opening Visual Studio solution."
    start .build/Vortex.sln
elif [[ $answer -eq 2 ]]; then
    echoGreen "Exiting."
else
    echo -e "\e[31mInvalid option, exiting.\e[0m"
    exit 1
fi

read -p "Press any key to continue..."

