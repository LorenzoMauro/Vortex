
cd..
call C:\Users\utente\anaconda3\Scripts\activate.bat
Pyinstaller --onefile --distpath "./" --workpath "./pyinstallerTmp/" "setup.py"
del setup.spec
rmdir /s /q "./pyinstallerTmp/"
pause
