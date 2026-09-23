@echo off
setlocal
rem pack PTQ1_0 text-only (skip vision/mtp/dflash2)
set "ROOT=%~dp0"
set "PY=D:\David\python\python.exe"
set "OUT=D:\LLM\ninfer-out\Ternary-Bonsai-2-27B-PTQ1_0-text.ninfer"
if exist "%OUT%" del /f /q "%OUT%"
"%PY%" "%ROOT%tools\pack_text.py" --template "D:\LLM\llama\qwen3_8_27b-v2.ninfer" --gguf "D:\LLM\llama\Ternary-Bonsai-2-27B-PTQ1_0.gguf" --out "%OUT%"
exit /b %ERRORLEVEL%
