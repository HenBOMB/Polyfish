Set WshShell = CreateObject("WScript.Shell")
Set FileSystem = CreateObject("Scripting.FileSystemObject")
ScriptDirectory = FileSystem.GetParentFolderName(WScript.ScriptFullName)
Launcher = FileSystem.BuildPath(ScriptDirectory, "auto_train.ps1")
WshShell.Run "powershell.exe -ExecutionPolicy Bypass -File " & Chr(34) & Launcher & Chr(34), 0, False
