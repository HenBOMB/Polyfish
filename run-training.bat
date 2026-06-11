@echo off
cd /d "%~dp0polyfish-rs"
powershell.exe -ExecutionPolicy Bypass -File .\run_training_loop.ps1
