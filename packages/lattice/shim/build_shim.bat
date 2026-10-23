@echo off
rem ============================================================================
rem  lattice shim build script: compile the Lattice LibTorch bridge DLL
rem  (pure C++, no Python). Requires a local LibTorch SDK (include/ + lib/).
rem
rem  Dev default: miniconda torch ships the official LibTorch C++ SDK layout,
rem  so no download is needed. Overrides:
rem    set TORCH_ROOT=...    SDK root containing include/ and lib/
rem    set LATTICE_PIN=x.y.z pinned torch version (must match TORCH_ROOT,
rem                          mismatch fails at lattice_init smoke / corrupts)
rem ============================================================================
setlocal
set "CLANG=%SCOURGE_CLANG%"
if "%CLANG%"=="" set "CLANG=D:\LLVM\bin\clang++.exe"
if not exist "%CLANG%" set "CLANG=clang++"

if "%TORCH_ROOT%"=="" set "TORCH_ROOT=C:\Users\38820\miniconda3\Lib\site-packages\torch"
if not exist "%TORCH_ROOT%\include\torch\script.h" (
    echo [lattice] LibTorch SDK not found: %TORCH_ROOT%
    echo [lattice] set TORCH_ROOT to a LibTorch root containing include/ and lib/
    exit /b 1
)

set "PIN=2.13.0"
if not "%LATTICE_PIN%"=="" set "PIN=%LATTICE_PIN%"

if not exist "..\bin" mkdir "..\bin"

rem pp-number trick: 2.13.0 is a single token, so stringizing needs no quote
rem escaping (see lattice_shim.hpp)
"%CLANG%" -shared -O2 -std=c++20 ^
  -D_MT -D_DLL -D_CRT_SECURE_NO_WARNINGS ^
  -Wno-ignored-attributes ^
  -Xclang --dependent-lib=msvcrt ^
  -Xclang --dependent-lib=vcruntime ^
  -Xclang --dependent-lib=ucrt ^
  -DTORCH_VERSION_EXPECTED=%PIN% ^
  -I"%TORCH_ROOT%\include" ^
  -I"%TORCH_ROOT%\include\torch\csrc\api\include" ^
  -L"%TORCH_ROOT%\lib" ^
  -ltorch_cpu -lc10 ^
  lattice_core.cpp lattice_tensor.cpp lattice_train.cpp ^
  -o "..\bin\lattice.dll"
if errorlevel 1 exit /b 1
echo lattice: built ..\bin\lattice.dll ^| torch pin %PIN% ^| SDK %TORCH_ROOT%
endlocal
