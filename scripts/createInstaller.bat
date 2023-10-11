cd ..

cmake --build ./build --config Release --target Vortex
cmake --build ./build --config Release --target INSTALL 

del ".\build\install\cudnn64_8.dll"
del ".\build\install\cudnn_ops_train64_8.dll"
del ".\build\install\cusparse64_11.dll"
del ".\build\install\cudnn_adv_infer64_8.dll"
del ".\build\install\cufft64_10.dll"
del ".\build\install\fbjni.dll"
del ".\build\install\cudnn_adv_train64_8.dll"
del ".\build\install\cufftw64_10.dll"
del ".\build\install\libiompstubs5md.dll"
del ".\build\install\cudnn_cnn_infer64_8.dll"
del ".\build\install\cupti64_2022.2.0.dll"
del ".\build\install\nvrtc-builtins64_117.dll"
del ".\build\install\cudnn_cnn_train64_8.dll"
del ".\build\install\cusolver64_11.dll"
del ".\build\install\nvrtc64_112_0.dll"
del ".\build\install\cudnn_ops_infer64_8.dll"
del ".\build\install\cusolverMg64_11.dll"


"E:\nsis-binary-7336-2-beta\makensis.exe" /V4 ./scripts/VortexInstaller.nsi

pause
