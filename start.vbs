Option Explicit

Dim shell, fileSystem, projectRoot, pythonwPath, command
Set shell = CreateObject("WScript.Shell")
Set fileSystem = CreateObject("Scripting.FileSystemObject")

projectRoot = fileSystem.GetParentFolderName(WScript.ScriptFullName)
pythonwPath = projectRoot & "\.venv\Scripts\pythonw.exe"

If Not fileSystem.FileExists(pythonwPath) Then
    MsgBox "Python environment not found: " & pythonwPath, 16, "QQ Speed Dance Tool"
    WScript.Quit 1
End If

shell.CurrentDirectory = projectRoot
command = """" & pythonwPath & """ """ & projectRoot & "\main.py"""
shell.Run command, 0, False
