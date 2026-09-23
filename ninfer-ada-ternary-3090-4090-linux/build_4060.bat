@echo off
rem 一键编译 RTX 4060 (sm_89) -> build_4060\apps\ninfer.exe
rem 用法: 双击或回车执行；参数 clean = 从零编译
rem 低内存（7.6 GiB）档默认 -j1 防止 CUDA 单文件 OOM；NINFER_JOBS 可覆盖
setlocal
cd /d "%~dp0"
set "MODE=%~1"
if "%MODE%"=="" set "MODE=incremental"
if not defined NINFER_JOBS set "NINFER_JOBS=1"

set "ROOT=%CD%"
set "BUILD=%ROOT%\build_4060"
set "ARCH=89"

where cmake >nul 2>&1 || (echo 需要 cmake 在 PATH 中 & exit /b 1)
where ninja  >nul 2>&1 || (echo 需要 ninja 在 PATH 中 & exit /b 1)
where nvcc   >nul 2>&1 || (echo 需要 CUDA nvcc 在 PATH 中 & exit /b 1)

echo ==^> configure sm_%ARCH% build=%BUILD% mode=%MODE%
cmake -S "%ROOT%" -B "%BUILD%" -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_CUDA_ARCHITECTURES=%ARCH% ^
  -DNINFER_BUILD_APPS=ON
if errorlevel 1 exit /b 1

if /i "%MODE%"=="clean" (
  cmake --build "%BUILD%" --target clean
)

echo ==^> build (jobs=%NINFER_JOBS%)
cmake --build "%BUILD%" -j %NINFER_JOBS%
if errorlevel 1 exit /b 1

echo.
echo OK: %BUILD%\apps\ninfer.exe
if exist "%BUILD%\apps\ninfer.exe" "%BUILD%\apps\ninfer.exe" --help
exit /b 0
