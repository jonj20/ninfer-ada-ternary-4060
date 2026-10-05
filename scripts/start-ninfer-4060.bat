@echo off
rem Windows-native launcher for ninfer-serve built by scripts\build_4060_win.bat.
rem Runs the .exe directly - no WSL. This is the single launch entry point: 64K
rem context, thinking on, drafts off by default; -nothink / -drafts switch modes.
rem ASCII-only on purpose: cmd.exe reads .bat with the OEM codepage (936 here), so
rem UTF-8 non-ASCII comments corrupt the parser.
rem
rem Usage: start-ninfer-4060.bat [-ctx N] [-nothink] [-budget N] [-drafts N]
rem                                  [-port N] [-hostkv MIB] [-model PATH]
rem
rem   -ctx N        context limit. Default 65536 (64K), the shipped config.
rem   -nothink      disable thinking. Thinking is ON by default: sampling switches
rem                 to temperature 1.0 / top-p 0.95 / min-p 0.05 / presence-penalty
rem                 0.0 and --no-thinking is dropped. Costs no extra VRAM - the
rem                 startup reservation is identical with and without it.
rem   -budget N     thinking token cap via --default-thinking-budget.
rem                 Thinking effort itself is a per-request field (reasoning_effort),
rem                 not a serve flag, so the WebUI or the JSON body picks low/medium.
rem   -drafts N     MTP draft tokens. Default 0 = speculative decoding OFF, which
rem                 is required for 64K: enabling it costs 262 MiB of headroom
rem                 (budget 1450 -> 1188) and 1088 extra bytes/token, which alone
rem                 pushes the 64K reservation 208 MiB over. -drafts 1 needs a
rem                 much lower -ctx, so re-measure before raising it.
rem   -port N       listen port (default 8080).
rem   -hostkv MIB   host KV arena in MiB (default 0, off).
rem   -model PATH   .ninfer artifact. Overrides the default below.
rem
rem MEMORY - this card is an RTX 4060 Laptop with 8188 MiB. After the weights load
rem the engine reports ~1450 MiB free (1188 with MTP on) and then asks for a
rem reservation. With rk4v4-e8 and drafts off it is linear in context length:
rem
rem     reservation = 17408 bytes/token x max_context + fixed
rem     fixed = 195 MiB with the flags this script already passes
rem              (--device-state-slots 0 --prefill-chunk 128, CUDA graph kept),
rem              470 MiB without them and 550 MiB with MTP enabled.
rem
rem   Measured with exactly the flags below, thinking on or off - identical
rem   either way:
rem
rem     ctx 65536 -> 1283.4 MiB   starts, 166.6 MiB slack   <- default
rem     ctx 76800 -> 1470.4 MiB   refused, 20.4 MiB short
rem     ctx 81920 -> ~1555   MiB  refused, ~105 MiB short
rem
rem   Practical ceiling is about 75500 tokens. 80K cannot fit: KV alone is
rem   1360 MiB plus the 195 MiB fixed block against a 1450 MiB budget. KV pages
rem   are 64 tokens, so pick a multiple of 64. A refusal is not a crash: the log
rem     "requested Engine runtime reservation requires N bytes, but only M ..."
rem   tells you exactly how much you are over by. Cut -ctx accordingly.
rem   Nothing in this script changes the model quantization (rk4v4-e8).
rem
rem RAM: startup pins 1.15 GiB of host state regardless of -hostkv, and the
rem process commits ~10 GiB while keeping only tens of MiB resident. 32 GB of
rem system RAM is comfortable; 16 GB is tight.
rem
rem Usage notes:
rem   - Ctrl+C in this window stops the server (it runs in the foreground).
rem   - Any stale ninfer-serve.exe is killed first, so rerunning is safe.
rem   - WebUI http://localhost:8080/   API http://localhost:8080/v1
rem     Monitor http://localhost:8080/monitor
setlocal enabledelayedexpansion

set "ROOT=%~dp0.."
for %%I in ("%ROOT%") do set "ROOT=%%~fI"
set "APPS=%ROOT%\build_4060\apps"
set "EXE=%APPS%\ninfer-serve.exe"
set "MODEL=E:\gguf\qwen3.6-35b-a3b\Ternary-Bonsai-2-27B-PTQ1_0-vl_mtp_q4q5.ninfer"

set "VCPKG=D:\dev\vcpkg"
set "CUDA=D:\dev\nvkit"

set "CTX="
set "THINK=1"
set "BUDGET="
set "DRAFTS="
set "PORT="
set "HOSTKV="

:parseargs
if "%~1"=="" goto parsed
if /i "%~1"=="-ctx" (set "CTX=%~2" & shift & shift & goto parseargs)
if /i "%~1"=="-think" (set "THINK=1" & shift & goto parseargs)
if /i "%~1"=="-nothink" (set "THINK=" & shift & goto parseargs)
if /i "%~1"=="-budget" (set "BUDGET=%~2" & shift & shift & goto parseargs)
if /i "%~1"=="-drafts" (set "DRAFTS=%~2" & shift & shift & goto parseargs)
if /i "%~1"=="-port" (set "PORT=%~2" & shift & shift & goto parseargs)
if /i "%~1"=="-hostkv" (set "HOSTKV=%~2" & shift & shift & goto parseargs)
if /i "%~1"=="-model" (set "MODEL=%~2" & shift & shift & goto parseargs)
echo Unknown switch: %~1
echo Usage: start-ninfer-4060.bat [-ctx N] [-nothink] [-budget N] [-drafts N] [-port N] [-hostkv MIB] [-model PATH]
exit /b 2
:parsed

if not exist "%EXE%" (
  echo ERROR: %EXE% not found.
  echo Build it first with scripts\build_4060_win.bat
  exit /b 1
)
if not exist "%MODEL%" (
  echo ERROR: model artifact not found:
  echo        %MODEL%
  echo Pass -model PATH to point at a .ninfer file.
  exit /b 1
)

if "%CTX%"=="" set "CTX=65536"
if "%DRAFTS%"=="" set "DRAFTS=0"
if "%PORT%"=="" set "PORT=8080"
if "%HOSTKV%"=="" set "HOSTKV=0"

rem --- sampling and thinking switches -------------------------------
if defined THINK (
  set "SAMPLING=--temperature 1.0 --top-p 0.95 --top-k 20 --min-p 0.05 --presence-penalty 0.0"
  set "THINKFLAG="
  if not "%BUDGET%"=="" set "THINKFLAG=--default-thinking-budget %BUDGET%"
) else (
  set "SAMPLING=--temperature 0.7 --top-p 0.80 --top-k 20 --min-p 0.0 --presence-penalty 1.5"
  set "THINKFLAG=--no-thinking"
)

rem --- speculation --------------------------------------------------
set "SPEC=--spec mtp --draft-tokens %DRAFTS%"
if "%DRAFTS%"=="0" set "SPEC="

rem --- stop any stale server ----------------------------------------
taskkill /f /im ninfer-serve.exe >nul 2>&1
ping -n 3 127.0.0.1 >nul

rem --- runtime DLLs: ffmpeg + curl come from vcpkg, cudart from CUDA --
set "PATH=%VCPKG%\installed\x64-windows\bin;%CUDA%\bin;%PATH%"

echo === ninfer-serve (native Windows) ===
echo   exe     : %EXE%
echo   model   : %MODEL%
if defined THINK (echo   mode    : THINK) else (echo   mode    : no-thinking)
echo   context : %CTX%   (reservation ~ see MEMORY notes in this script)
echo   kv      : rk4v4-e8   lanes 1   host-kv %HOSTKV% MiB
echo   spec    : %SPEC%
echo   port    : %PORT%
echo.
echo   WebUI   http://localhost:%PORT%/
echo   API     http://localhost:%PORT%/v1
echo   Monitor http://localhost:%PORT%/monitor
echo.
echo Log follows. Ctrl+C stops the server.
echo ==========================================
echo.

"%EXE%" "%MODEL%" ^
  --host 0.0.0.0 --port %PORT% ^
  --max-context %CTX% --kv-capacity %CTX% ^
  --kv-dtype rk4v4-e8 ^
  --max-concurrency 1 --default-max-tokens 32768 ^
  --host-kv-mib %HOSTKV% ^
  --max-private-continuations 1 --max-shared-prefixes 0 ^
  --max-long-anchors-per-continuation 0 --auto-long-anchors 0 ^
  --device-state-slots 0 --prefill-chunk 128 ^
  %THINKFLAG% %SAMPLING% %SPEC%

set "RC=%ERRORLEVEL%"
echo.
if not "%RC%"=="0" if not "%RC%"=="1" echo === server exited with code %RC% ===
echo Server stopped.
pause
exit /b %RC%