@echo off
cd /d "%~dp0..\polyfish-rs" || exit /b 1
powershell.exe -ExecutionPolicy Bypass -File .\run_training_loop.ps1
