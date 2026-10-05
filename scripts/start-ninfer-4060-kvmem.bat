@echo off
rem Windows-native launcher that turns on the KVMem bounded KV working set, so a
rem context far larger than VRAM can be served. Runs the .exe directly - no WSL.
rem Built by scripts\build_4060_win.bat; this is the KVMem sibling of
rem start-ninfer-4060.bat, which stays the plain 64K dense launcher.
rem ASCII-only on purpose: cmd.exe reads .bat with the OEM codepage (936 here), so
rem UTF-8 non-ASCII comments corrupt the parser.
rem
rem WHAT THIS BUYS YOU
rem   Default here is a 262144-token logical context with only 65536 tokens of
rem   Device KV. The rest lives in the Host KV tier and comes back through a
rem   per-round retrieval window instead of staying resident. Measured on an RTX
rem   4060 Laptop (8188 MiB) with Ternary-Bonsai-2-27B-PTQ1_0-vl_mtp_q4q5.ninfer:
rem
rem     prompt_n = 258,939 tokens prefilled and generated in one request (364 s),
rem     no context_length_exceeded, while device KV residency fell from 1024
rem     pages (65536 tokens) to 559 and Host KV grew to 3.89 GiB.
rem
rem   That same context placed densely needs 4.25 GiB of KV plus 5.52 GiB of
rem   weights, which does not fit in 8.58 GiB. That gap is the whole point.
rem
rem THE THREE NUMBERS
rem   ctx      262144  logical ceiling (--max-context)
rem   kvc       65536  Device KV pool, the resident working set (--kv-capacity)
rem   budget    32768  tokens the KVMem window keeps Device-resident (--kvmem-budget)
rem
rem   gen_reserve is not a flag: it is derived as kv_capacity - budget = 32768.
rem
rem WHY --kv-ring IS MANDATORY HERE
rem   startup.cpp requires kv_capacity >= max_context unless --kv-ring is set, and
rem   --kvmem-budget does NOT relax that (ring_requested() reads only kv_ring).
rem   --kv-ring is what lets the Device pool (65536) be smaller than the logical
rem   context (262144). It only lowers the pool-size floor; the windowing itself is
rem   driven by --kvmem-budget. Without it startup refuses with
rem   "kv_capacity must be at least max_context".
rem
rem WHY -hostkv IS NOT PASSED BY DEFAULT
rem   With --kv-ring and --kvmem-budget both on, "--host-kv-mib 0" makes startup
rem   auto-size the Host tier by the ring rule (logical - pool = 3072 pages) while
rem   KVMem needs logical - window (3584 pages) and then refuses. Leaving the flag
rem   off uses the ContextCacheOptions default: 8 GiB addressable, pinned lazily.
rem   Pass -hostkv only when you must pin an explicit size (>= 3809 at 256K).
rem
rem RETRIEVAL CAVEAT
rem   -select retrieval scores blocks by mean-key similarity. It is silently
rem   ignored while speculative decoding is on, because target verification still
rem   reads absolute positions; the window degrades to an identity one. Drafts are
rem   off here (see -drafts) precisely so selection stays live.
rem
rem Usage: start-ninfer-4060-kvmem.bat [-ctx N] [-kvc N] [-budget N] [-noring]
rem                                         [-select recency|retrieval] [-nothink]
rem                                         [-drafts N] [-hostkv MIB] [-prefill N]
rem                                         [-port N] [-model PATH]
rem
rem   -ctx N        logical context ceiling (default 262144; multiple of 128)
rem   -kvc N        Device KV pool in tokens (default 65536). Must be >=
rem                  budget + prefill chunk, and < ctx, which is why --kv-ring is
rem                  passed unless -noring is given.
rem   -budget N     resident KVMem window in tokens (default 32768; multiple of 128)
rem   -noring       drop --kv-ring. Legal only when -kvc >= -ctx, which makes the
rem                  pool cover the whole context and needs far more VRAM; that
rem                  defeats KVMem, so this is for A/B comparison only.
rem   -select M     recency (default) or retrieval
rem   -nothink      disable thinking. Costs no extra VRAM.
rem   -drafts N     MTP draft tokens. Default 0 = speculation OFF, which keeps the
rem                  window live. Any nonzero value disables KVMem selection.
rem   -hostkv MIB   pin the Host KV arena to MIB MiB (>= 3809 at 256K context).
rem   -prefill N    prefill chunk in tokens (default 1024; multiple of 128)
rem   -port N       listen port (default 8080)
rem   -model PATH   .ninfer artifact. Overrides the default below.
rem
rem MEMORY - this card is an RTX 4060 Laptop with 8188 MiB.
rem   VRAM: weights 5.52 GiB + KV pool 1.14 GiB at kvc 65536, plus ~195 MiB fixed.
rem        Scaling kvc up scales VRAM linearly at 17408 bytes/token.
rem   HOST: startup pins 1.15 GiB of host state regardless of -hostkv, and Host KV
rem        reaches ~3.9 GiB at 256K context. 32 GB of system RAM is comfortable;
rem        16 GB is tight (measured OK at 15.7 GB total / 8.9 GB free); 12 GB will
rem        not hold a 256K context. Lower -ctx to trade context for RAM.
rem
rem   A refusal is not a crash: the log names the exact shortfall, e.g.
rem     "requested Engine runtime reservation requires N bytes, but only M ..."
rem     or "--kvmem-budget needs a Host KV tier ...". Cut -ctx / -kvc to fit.
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
set "KVC="
set "BUDGET="
set "RING=1"
set "SELECT="
set "THINK=1"
set "DRAFTS="
set "PRE="
set "PORT="
set "HOSTKV="

:parseargs
if "%~1"=="" goto parsed
if /i "%~1"=="-ctx" (set "CTX=%~2" & shift & shift & goto parseargs)
if /i "%~1"=="-kvc" (set "KVC=%~2" & shift & shift & goto parseargs)
if /i "%~1"=="-budget" (set "BUDGET=%~2" & shift & shift & goto parseargs)
if /i "%~1"=="-noring" (set "RING=0" & shift & goto parseargs)
if /i "%~1"=="-select" (set "SELECT=%~2" & shift & shift & goto parseargs)
if /i "%~1"=="-think" (set "THINK=1" & shift & goto parseargs)
if /i "%~1"=="-nothink" (set "THINK=" & shift & goto parseargs)
if /i "%~1"=="-drafts" (set "DRAFTS=%~2" & shift & shift & goto parseargs)
if /i "%~1"=="-hostkv" (set "HOSTKV=%~2" & shift & shift & goto parseargs)
if /i "%~1"=="-prefill" (set "PRE=%~2" & shift & shift & goto parseargs)
if /i "%~1"=="-port" (set "PORT=%~2" & shift & shift & goto parseargs)
if /i "%~1"=="-model" (set "MODEL=%~2" & shift & shift & goto parseargs)
echo Unknown switch: %~1
echo Usage: start-ninfer-4060-kvmem.bat [-ctx N] [-kvc N] [-budget N] [-noring]
echo                                          [-select recency^|retrieval] [-nothink]
echo                                          [-drafts N] [-hostkv MIB] [-prefill N]
echo                                          [-port N] [-model PATH]
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

if "%CTX%"=="" set "CTX=262144"
if "%KVC%"=="" set "KVC=65536"
if "%BUDGET%"=="" set "BUDGET=32768"
if "%SELECT%"=="" set "SELECT=recency"
if "%DRAFTS%"=="" set "DRAFTS=0"
if "%PRE%"=="" set "PRE=1024"
if "%PORT%"=="" set "PORT=8080"

set /a GENRESERVE=KVC-BUDGET

rem --kv-ring is what permits kv_capacity < max_context; without it startup rejects
rem the pool with "kv_capacity must be at least max_context".
set "RINGFLAG="
if "%RING%"=="1" set "RINGFLAG=--kv-ring"

rem Omitted on purpose; see WHY -hostkv IS NOT PASSED BY DEFAULT above.
set "HOSTKVFLAG="
if not "%HOSTKV%"=="" set "HOSTKVFLAG=--host-kv-mib %HOSTKV%"

rem --- sampling and thinking switches -------------------------------
if defined THINK (
  set "SAMPLING=--temperature 1.0 --top-p 0.95 --top-k 20 --min-p 0.05 --presence-penalty 0.0"
  set "THINKFLAG="
) else (
  set "SAMPLING=--temperature 0.7 --top-p 0.80 --top-k 20 --min-p 0.0 --presence-penalty 1.5"
  set "THINKFLAG=--no-thinking"
)

rem --- speculation --------------------------------------------------
rem Drafts default to 0. Enabling them makes KVMem selection inert (target
rem verification still reads absolute positions), so this is not just a VRAM knob.
set "SPEC=--spec mtp --draft-tokens %DRAFTS%"
if "%DRAFTS%"=="0" set "SPEC="

rem --- stop any stale server ----------------------------------------
taskkill /f /im ninfer-serve.exe >nul 2>&1
ping -n 3 127.0.0.1 >nul

rem --- runtime DLLs: ffmpeg + curl come from vcpkg, cudart from CUDA --
set "PATH=%VCPKG%\installed\x64-windows\bin;%CUDA%\bin;%PATH%"

echo === ninfer-serve (KVMem bounded KV working set) ===
echo   exe      : %EXE%
echo   model    : %MODEL%
if defined THINK (echo   mode     : THINK) else (echo   mode     : no-thinking)
echo   context  : %CTX%   (logical ceiling)
echo   kv pool  : %KVC%   (device working set)
echo   window   : %BUDGET%   (kvmem resident budget)
echo   reserve  : %GENRESERVE%   (= kv_capacity - budget)
echo   select   : %SELECT%
echo   ring     : %RINGFLAG%
echo   spec     : %SPEC%
echo   kv       : rk4v4-e8   lanes 1   prefill %PRE%
echo   port     : %PORT%
echo.
echo   VRAM need : weights 5.52 GiB + kv pool ~1.14 GiB at kvc %KVC%
echo   HOST need : ~1.15 GiB startup + ~3.9 GiB KV at ctx %CTX%
echo.
echo   WebUI   http://localhost:%PORT%/
echo   API     http://localhost:%PORT%/v1
echo   Monitor http://localhost:%PORT%/monitor
echo.
echo Log follows. Ctrl+C stops the server.
echo ==========================================================
echo.

"%EXE%" "%MODEL%" ^
  --host 0.0.0.0 --port %PORT% ^
  --max-context %CTX% --kv-capacity %KVC% %RINGFLAG% ^
  --kvmem-budget %BUDGET% --kvmem-select %SELECT% ^
  --kv-dtype rk4v4-e8 ^
  --max-concurrency 1 --default-max-tokens 262144 ^
  --prefill-chunk %PRE% ^
  %HOSTKVFLAG% ^
  --device-state-slots 0 ^
  %THINKFLAG% %SAMPLING% %SPEC%

set "RC=%ERRORLEVEL%"
echo.
if not "%RC%"=="0" if not "%RC%"=="1" echo === server exited with code %RC% ===
echo Server stopped.
pause
exit /b %RC%