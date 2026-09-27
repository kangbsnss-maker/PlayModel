Option Explicit
Dim shell, files, root, python, app, log, logdir, launchError, launchDescription
Set shell = CreateObject("WScript.Shell")
Set files = CreateObject("Scripting.FileSystemObject")
root = files.GetParentFolderName(WScript.ScriptFullName)
python = root & "\.venv\Scripts\pythonw.exe"
app = root & "\scripts\local_learning_app.py"
If Not files.FolderExists(root & "\artifacts") Then files.CreateFolder(root & "\artifacts")
logdir = root & "\artifacts\local-learning"
If Not files.FolderExists(logdir) Then files.CreateFolder(logdir)
Set log = files.OpenTextFile(logdir & "\launcher-vbs.log", 8, True)
log.WriteLine CStr(Now) & " start " & app
log.Close
If Not files.FileExists(python) Then
    Set log = files.OpenTextFile(logdir & "\launcher-vbs.log", 8, True)
    log.WriteLine CStr(Now) & " error python_environment_missing " & python
    log.Close
    MsgBox "Python environment missing. See docs/guides/standalone-learning.html", 16, "PlayModel"
    WScript.Quit 1
End If
shell.CurrentDirectory = root
On Error Resume Next
shell.Run Chr(34) & python & Chr(34) & " -X utf8 " & Chr(34) & app & Chr(34), 1, False
If Err.Number <> 0 Then
    launchError = Err.Number
    launchDescription = Err.Description
    Set log = files.OpenTextFile(logdir & "\launcher-vbs.log", 8, True)
    log.WriteLine CStr(Now) & " launch_failed " & CStr(launchError) & " " & launchDescription
    log.Close
    WScript.Quit 1
End If
