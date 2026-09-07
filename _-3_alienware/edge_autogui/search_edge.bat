@echo off
rem Run from this file's folder; caller arguments override the defaults.
python "%~dp0edge_autogui.py" --repeat 20 --wait-min 5 --wait-max 15 --start-delay 0 %*
exit /b %errorlevel%
