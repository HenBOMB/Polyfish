Set WshShell = CreateObject("WScript.Shell")
' The 0 means "Hide the window", False means "don't wait for it to finish"
WshShell.Run "powershell.exe -ExecutionPolicy Bypass -File auto_train.ps1", 0, False