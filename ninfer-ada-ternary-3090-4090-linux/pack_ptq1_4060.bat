@echo off
setlocal
rem pack PTQ1_0 text-only（保留 mtp/vision —— 引擎绑定必选；裁 dflash2）
set "ROOT=%~dp0"
set "PY=D:\dev\python\python.exe"
set "OUT=E:\gguf\qwen3.6-35b-a3b\Ternary-Bonsai-2-27B-PTQ1_0-text.ninfer"
if exist "%OUT%" del /f /q "%OUT%"
"%PY%" "%ROOT%tools\pack_text.py" --keep-mtp --keep-vision --template "E:\gguf\qwen3.6-35b-a3b\qwen3_8_27b.ninfer" --gguf "E:\gguf\qwen3.6-35b-a3b\Ternary-Bonsai-2-27B-PTQ1_0.gguf" --out "%OUT%"
exit /b %ERRORLEVEL%
