@echo off
title AI工业医生启动器
cd /d %~dp0
echo 正在启动，请稍后...
call .venv\Scripts\activate.bat && streamlit run web_app.py
pause