@echo off
setlocal enableextensions enabledelayedexpansion
title SplatKit one-click installer

REM ===========================================================================
REM  SplatKit one-click installer for ComfyUI (Windows)
REM
REM  What to do with this file:
REM    1. Download it from the SplatKit repo.
REM    2. Drop it into your ComfyUI folder -- the portable folder that holds
REM       "python_embeded" and "ComfyUI" (where run_nvidia_gpu.bat lives),
REM       OR inside your "ComfyUI" folder next to "custom_nodes".
REM    3. Double-click it.
REM
REM  It will: fetch the SplatKit node pack into custom_nodes, install the node
REM  dependencies into ComfyUI's Python, and (optionally) build the self-
REM  contained CUDA backend used by the 4D generator and the splat trainer.
REM
REM  Fork / branch note: if you fork the repo or publish to a different branch,
REM  change REPO_URL / BRANCH below. Once this branch is merged, set BRANCH=main.
REM ===========================================================================

set "REPO_URL=https://github.com/mickmumpitz/ComfyUI-SplatKit"
set "BRANCH=feature/one-click-install"
set "PACK_NAME=ComfyUI-SplatKit"

echo.
echo   ==================================================================
echo     SplatKit installer
echo     Gaussian-splat datasets + 4D / training nodes for ComfyUI
echo   ==================================================================
echo.

set "HERE=%~dp0"
if "%HERE:~-1%"=="\" set "HERE=%HERE:~0,-1%"

call :find_python
if not defined PYEXE (
  echo   [X] Could not find ComfyUI's Python.
  echo       Put this file in your ComfyUI folder -- the one that contains
  echo       "python_embeded" ^(portable^), or inside your ComfyUI folder next
  echo       to "custom_nodes" if you use a system / venv Python.
  goto :fail
)

call :find_custom_nodes
if not defined CUSTOM_NODES (
  echo   [X] Could not find a "custom_nodes" folder near this installer.
  echo       Put this file in your ComfyUI portable folder, or inside ComfyUI.
  goto :fail
)

set "PACK_DIR=!CUSTOM_NODES!\%PACK_NAME%"

echo   Python        : !PYEXE!
echo   custom_nodes  : !CUSTOM_NODES!
echo   Source        : %REPO_URL% ^(%BRANCH%^)
echo.

REM ---------------------------------------------------------------- 1. fetch
where git >nul 2>nul
if !errorlevel! equ 0 (
  call :get_with_git
) else (
  echo   [1/3] git not found -- downloading a zip instead ^(no git needed^).
  call :get_with_zip
)
if !errorlevel! neq 0 goto :fail

if not exist "!PACK_DIR!\requirements.txt" (
  echo   [X] The pack did not download correctly ^(requirements.txt missing^).
  goto :fail
)

REM ---------------------------------------------------- 2. host dependencies
echo.
echo   [2/3] Installing node dependencies into ComfyUI's Python ...
"!PYEXE!" -m pip install --upgrade pip >nul 2>nul
"!PYEXE!" -m pip install -r "!PACK_DIR!\requirements.txt"
if !errorlevel! neq 0 (
  echo   [X] Dependency install failed -- scroll up for the pip error.
  goto :fail
)

REM ----------------------------------------------------- 3. optional backend
echo.
echo   [3/3] The 4D generator and the splat trainer use a separate,
echo         self-contained CUDA backend ^(about 7.5 GB, Windows + NVIDIA only^).
echo         The panorama-to-dataset pipeline does NOT need it.
echo.
set "DOBACKEND=Y"
set /p "DOBACKEND=Install the CUDA backend now? [Y/n]: "
if /i "!DOBACKEND!"=="n" goto :skip_backend

echo.
echo   Building the backend -- this downloads several GB and can take a while.
echo   Leave the window open; it prints progress as it goes.
echo.
"!PYEXE!" "!PACK_DIR!\tools\install_splat_backend.py"
if !errorlevel! neq 0 (
  echo.
  echo   [!] Backend build did not finish. The panorama pipeline still works.
  echo       Re-run this installer later, or use the "Splat Backend Setup"
  echo       node inside ComfyUI to try again.
) else (
  echo   Backend ready.
)
goto :done

:skip_backend
echo.
echo   Skipped the CUDA backend. Run this installer again -- or the
echo   "Splat Backend Setup" node in ComfyUI -- whenever you want the
echo   4D / training nodes.

:done
echo.
echo   ==================================================================
echo     Done. Restart ComfyUI, then look for the "SplatKit" node group.
echo     The MoGe and SphereSfM models download automatically on first use.
echo   ==================================================================
echo.
pause
endlocal
exit /b 0

:fail
echo.
echo   Installation stopped. Fix the issue above and run this file again.
echo.
pause
endlocal
exit /b 1

REM =============================== helpers ===================================

:find_python
set "PYEXE="
if not defined PYEXE if exist "%HERE%\python_embeded\python.exe"          set "PYEXE=%HERE%\python_embeded\python.exe"
if not defined PYEXE if exist "%HERE%\..\python_embeded\python.exe"       set "PYEXE=%HERE%\..\python_embeded\python.exe"
if not defined PYEXE if exist "%HERE%\ComfyUI\python_embeded\python.exe"  set "PYEXE=%HERE%\ComfyUI\python_embeded\python.exe"
if not defined PYEXE for %%P in (python.exe) do if not defined PYEXE set "PYEXE=%%~$PATH:P"
goto :eof

:find_custom_nodes
set "CUSTOM_NODES="
if not defined CUSTOM_NODES if exist "%HERE%\ComfyUI\custom_nodes" set "CUSTOM_NODES=%HERE%\ComfyUI\custom_nodes"
if not defined CUSTOM_NODES if exist "%HERE%\custom_nodes"         set "CUSTOM_NODES=%HERE%\custom_nodes"
if not defined CUSTOM_NODES if exist "%HERE%\..\custom_nodes"      set "CUSTOM_NODES=%HERE%\..\custom_nodes"
goto :eof

:get_with_git
if exist "!PACK_DIR!\.git" (
  echo   [1/3] Updating the existing SplatKit checkout with git ...
  git -C "!PACK_DIR!" fetch --depth 1 origin "%BRANCH%"
  if !errorlevel! neq 0 exit /b 1
  git -C "!PACK_DIR!" checkout "%BRANCH%" 1>nul 2>nul
  git -C "!PACK_DIR!" reset --hard "origin/%BRANCH%"
  if !errorlevel! neq 0 exit /b 1
) else (
  echo   [1/3] Cloning the SplatKit node pack with git ...
  git clone --depth 1 --branch "%BRANCH%" "%REPO_URL%" "!PACK_DIR!"
  if !errorlevel! neq 0 (
    echo   [X] git clone failed.
    exit /b 1
  )
)
exit /b 0

:get_with_zip
set "SK_URL=%REPO_URL%/archive/refs/heads/%BRANCH%.zip"
set "SK_DEST=!PACK_DIR!"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; $u=$env:SK_URL; $d=$env:SK_DEST; $z=Join-Path $env:TEMP 'splatkit.zip'; $t=Join-Path $env:TEMP ('sk_'+[guid]::NewGuid().ToString()); Invoke-WebRequest -UseBasicParsing -Uri $u -OutFile $z; Expand-Archive -Path $z -DestinationPath $t -Force; $src=(Get-ChildItem -Path $t -Directory | Select-Object -First 1).FullName; if(Test-Path $d){Remove-Item -Recurse -Force $d}; Move-Item -Path $src -Destination $d; Remove-Item -Force $z; Remove-Item -Recurse -Force $t"
if !errorlevel! neq 0 (
  echo   [X] Download failed. Check your internet connection and try again,
  echo       or install git and re-run this file.
  exit /b 1
)
exit /b 0
