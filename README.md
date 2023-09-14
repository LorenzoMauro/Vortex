# Vortex

# A Journey from Concept to Reality

[Hello there](https://media.giphy.com/media/xTiIzJSKB4l7xTouE8/giphy.gif)!
I'm [Lorenzo Mauro](https://www.linkedin.com/in/lorenzo-mauro-4082a088/), currently diving deep into the finishing stages of a Computer Graphics/Computer Science PhD at the [University of Rome](https://phd.uniroma1.it/web/LORENZO-MAURO_nP1529128_EN.aspx). In the past I've also dipped my toes in concept art at [OnePixelBrush](https://onepixelbrush.com/). Wanna take a walk down memory lane? Here's my [previous portfolio](https://www.artstation.com/lomar).

Vortex is a labor of love, built from the ground up with two primary objectives in mind:

1. **Practical Skill Development**: As someone passionate about computer graphics and eager to make a mark in the industry, I wanted to do more than just work within the confines of existing frameworks to achieve my phd thesis. Diving deep into the nitty-gritty of renderer development provided invaluable insights and hands-on experience beyond the regular PhD curriculum.

2. **Research Endeavor**: Most importantly, Vortex isn't merely a fancy toy-renderer; it's a playground for my academic adventure. This sandbox enabled me to intertwine Path Guiding techniques with the power of neural networks.

While Vortex is a comprehensive project, it's essential to note that its current version is a reflection of focused objectives under tight timelines. I've been the solo developer, so the prioritization was crucial. A few features like dynamic scenes and volumetrics are on my radar for future iterations but at present, the emphasis remains on optimizing and refining the neural network components due to the demands of my PhD work, which is a beast on its own.

If you have feedback, find bugs, or are intrigued by any part of this project, please feel free to reach out 😄.

Happy rendering!

## Table of Contents

- [Introduction](#a-journey-from-concept-to-reality)
- [Technical Details](#technical-details)
  - [Tech Stack & Dependencies](#tech-stack--dependencies)
  - [Architecture](#architecture)
  - [Features & Capabilities](#features--capabilities)
  - [Neural Network Path Guiding](#neural-network-path-guiding)
  - [Future Roadmap](#future-roadmap)
  - [Collaboration](#collaboration)
- [Installation Guide](#installation-guide)
  - [Cloning the Repository](#cloning-the-repository)
  - [Pre-requisites](#pre-requisites)
  - [Scripted Setup](#scripted-setup)
  - [Manual Installation](#manual-installation)
    - [Vcpkg Setup](#vcpkg-setup)
    - [Building MDL](#building-mdl)
    - [Configuring Vortex](#configuring-vortex)

## Technical Details

### Tech Stack & Dependencies
Vortex is a GPU physically-based renderer. It leverages [Nvidia OptiX](https://developer.nvidia.com/rtx/ray-tracing/optix) for ray-tracing acceleration and the [Nvidia MDL-SDK](https://developer.nvidia.com/nvidia-mdl-sdk-get-started) for material shaders. With an [ImGui](https://github.com/ocornut/imgui)-based GUI it is possible to navigate the scene and customize most of the renderer settings. Vortex also supports the import of 3D scenes in standard formats with [Assimp](https://github.com/assimp/assimp), even though the gltfw format with separate textures has been proved the most compatible way of loading scenes. Scene customization and editing options are currently limited, but a shader graph node is available, allowing for material parameter edits.

### Architecture
The core of Vortex's design utilizes the visitor pattern. As such most components are represented as nodes, including the renderer itself.
Several visitors then perform varied operations, with one particular visitor responsible for loading data onto the GPU.

The Renderer has two modes: 
1. A mode reliant on Optix shader binding tables and functions for the architectural framework.
2. A wavefront architecture mode that combines Optix functions for ray tracing with CUDA kernels for shading tasks, and SOA data structures to minimize divergence. 

This wavefront architecture was indispensable for the integration of neural networks within the rendering loop. With this setup, it's feasible to employ a neural network, currently crafted with [LibTorch](https://pytorch.org/cppdocs/), for both training and inference during rendering. An eventual transition to [tiny-cuda](https://github.com/NVlabs/tiny-cuda-nn) is on the horizon, though immediate constraints make libTorch a more practical choice.

### Features & Capabilities
Vortex support for instancing and, thanks to the integration of the mdl-sdk, is compatible with a broad range of materials. An in-house "uber-shader" has also been crafted, drawing inspiration from the Disney principled shader and Blender's equivalent. Futhermore Vortex also implements adaptive sampling, firefly removale, tone mapping and denoising with the Optix Denoiser.

### Neural Network Path Guiding
The highlight of Vortex is its use of neural networks for path guiding. In the wavefront architecture, bounce information is retained. During training, a neural network is fine-tuned to generate a distribution from which ray extension samples are derived. This training minimizes the KL divergence between the sample's distribution value and the light transport equation's returned value. This lets Vortex produce samples that simultaneously consider light distribution and material properties, greatly speeding up variance reduction in challenging scenes.

### Future Roadmap
The immediate focus lies on optimizing the neural network. Improvements over existing methodologies have been identified, particularly in better sample selection for training. A primary identified challenge is ensuring the neural network has sufficient exposure during training to challenging sample paths, like caustics, which current papers don't adequately address. In terms of capabilities, future iterations of Vortex aim to support dynamic scenes, volumetric rendering, and eventually, geometry and material editing functionalities.

### Collaboration
Vortex has been my solo passion project, but if any brave soul out there feels a pull towards it and thinks, "Hey, I could help make this even cooler," then hit me up! I'd be thrilled to join forces and take Vortex to the next level together.

## Installation Guide
**Note on Compatibility**:
This project has been developed for **Windows** and has been primarily tested with **Visual Studio 2022**.
While I've tried my best to make the installation process as smooth as possible, it's been validated on only a few machines. If you come across any bugs or issues, please don't hesitate to let me know. I'd genuinely appreciate the feedback and any information about potential hitches you might encounter.
### Cloning the Repository

If you haven't already cloned the repository, make sure you do so recursively to ensure all submodules are initialized and updated:

```bash
git clone --recursive https://github.com/LorenzoMauro/Vortex.git
```

If you've already cloned the repo but forgot the `--recursive` flag, you can initialize and update the submodules with:

```bash
cd [Vortex-Repo]
git submodule update --init --recursive
```

---
### Pre-requisites

Before diving into the installation process, there are some dependencies you need to manually install:

1. **CUDA 11.7**: [Download here](https://developer.nvidia.com/cuda-toolkit-archive)
2. **CUDA 8.0**: [Download here](https://developer.nvidia.com/cuda-80-ga2-download-archive) (Required by mdl-sdk)
3. **Clang 12**: [Download here](https://releases.llvm.org/download.html)
4. **OptiX 8.0**: [Download here](https://developer.nvidia.com/designworks/optix/download)
5. **LibTorch**:
   - Go to [PyTorch's site](https://pytorch.org/get-started/locally/)
   - Choose:
     - OS: Windows
     - Package: Libtorch
     - Compute Platform: CUDA 11.7
   - Grab both Release and Debug versions. Once downloaded, extract them. You'll find a `libtorch` folder inside each which will be needed later.

### Scripted Setup

For a quick setup, you can use the provided scripts in the repo's root: `install.py`, `install.exe`, `install.sh`.

**What it does**:
   - Checks for required dependencies.
   - Prompts for paths to specific tools or libraries if not found in default spots.
   - Utilizes `vcpkg` to get the necessary libraries.
   - Sets up MDL-SDK for material management.
   - Configures Vortex using CMake.
   - Gives the option to build and launch Vortex, pop it open in Visual Studio or exit.

### Manual Installation
In case you’re a fan of hands-on work, (or the scripts fails, ops!) follow these steps:

1. **Vcpkg Setup**:
   - If you have an existing `vcpkg` installation, update it using git pull origin master, otherwise, from the repository folder, clone it and bootstrap it:
     ```bash
     git clone https://github.com/Microsoft/vcpkg.git ./ext/vcpkg
     cd ext/vcpkg
     ./bootstrap-vcpkg.bat
     ```
   - You can now install the required libraries:
     ```bash
     ./vcpkg install boost-any boost-uuid --triplet=x64-windows-static
     ./vcpkg install openimageio --triplet=x64-windows-static
     ./vcpkg install imgui[docking-experimental,opengl3-binding,glfw-binding,win32-binding] --triplet=x64-windows-static-md --recurse
     ./vcpkg install spdlog yaml-cpp assimp glfw3 implot --triplet=x64-windows-static-md
     ```

2. **Building MDL**:
   - The MDL-SDK will be already cloned in the `./ext` folder as a submodule, time to build it.
   - Navigate to `ext/MDL-SDK`.
   - Create a build directory: `mkdir build && cd build`
   - Configure with cmake:
     ```bash
     cmake  -DCMAKE_TOOLCHAIN_FILE="<vcpkg_intallation_folder>/scripts/buildsystems/vcpkg.cmake" \
            -DMDL_BUILD_CORE_EXAMPLES=OFF \
            -DMDL_BUILD_DOCUMENTATION=OFF \
            -DMDL_BUILD_SDK_EXAMPLES=OFF \
            -DMDL_ENABLE_CUDA_EXAMPLES=OFF \
            -DMDL_ENABLE_D3D12_EXAMPLES=OFF \
            -DMDL_ENABLE_OPENGL_EXAMPLES=OFF \
            -DMDL_ENABLE_QT_EXAMPLES=OFF \
            -DMDL_ENABLE_VULKAN_EXAMPLES=OFF \
            ..
     ```
     or if you prefer using cmake-gui make sure to set all the options as in the command above, we basically don't need to build any example (which require additional dependencies).
   - Build: `cmake --build . --config Release`
   - Install: `cmake --build . --config Release --target INSTALL`

3. **Configuring Vortex**:
   - Navigate back to the root of Vortex.
   - Create a build directory if it doesn't exist and navigate to it.
   - Configure Vortex with cmake which will require you to specify some variables:
	- **`CMAKE_TOOLCHAIN_FILE`**: Path to the `vcpkg.cmake` in your vcpkg installation.
	- **`TORCH_INSTALL_PREFIX_DEBUG`**: Path to the "libtorch" folder in your Debug version of LibTorch.
	- **`TORCH_INSTALL_PREFIX_RELEASE`**: Path to the "libtorch" folder in your Release version of LibTorch.
	- **`CUDAToolkit_ROOT`**: Installation directory of CUDA 11.7.
	- **`CLANG_12_PATH`**: Path to Clang 12's binary (clang.exe).
	- **`CUDA_8_PATH`**: Installation directory of CUDA 8.0.
	- **`OPTIX77_PATH`**: Installation directory of OptiX 7.7.
	- **`-T cuda=<cudapath>`**: This is necessary if CUDA 11.7 is not the latest version of CUDA you have installed, it basically forces Cmake to use 11.7 which is a requirement for libtorch.
     ```bash
     cmake  -T cuda="usually C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.7" \
            -DCMAKE_TOOLCHAIN_FILE="<vcpkg_intallation_folder>/scripts/buildsystems/vcpkg.cmake" \
            -DTORCH_INSTALL_PREFIX_DEBUG="you-extracted-torch-debug-folder/libtorch" \
            -DTORCH_INSTALL_PREFIX_RELEASE="you-extracted-torch-release-folder/libtorch" \
            -DCUDAToolkit_ROOT="usually C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.7" \
            -DCLANG_12_PATH="usually C:/Program Files (x86)/LLVM" \
            -DCUDA_8_PATH="usually C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v8.0" \
            -DOPTIX77_PATH="usually C:\ProgramData\NVIDIA Corporation\OptiX SDK 8.0.0" \
            ..
     ```
- As before, if you prefer you can use cmake-gui, just make sure to set all the options as in the command line
4. **Build and Run**:
   You can do so on the command line:
   - Build Vortex: `cmake --build . --config Release`
   - Run Vortex: `./Vortex/src/Release/Vortex.exe`
   Or in visual studio opening the solution in the build folder.