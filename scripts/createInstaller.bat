cd ..

cmake --build ./build --config Release --target Vortex
cmake --build ./build --config Release --target INSTALL 

"E:\nsis-binary-7336-2-beta\makensis.exe" /V4 ./scripts/VortexInstaller.nsi

pause
