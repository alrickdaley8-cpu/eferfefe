@echo off
REM ============================================================================
REM Spidey Sense Mod — build script for Minecraft 1.21.1 Fabric (Windows)
REM ============================================================================
REM
REM This script does everything in one shot on Windows.
REM
REM   1. Detects an existing JDK 21 — uses it if found.
REM   2. If no JDK 21 exists, downloads a portable Eclipse Temurin JDK 21
REM      into .\build-tools\ and uses that. Tries multiple mirrors.
REM   3. Generates the Gradle wrapper.
REM   4. Builds the mod.
REM
REM Output: build\libs\spidey-sense-mod-1.1.0.jar
REM ============================================================================

setlocal enabledelayedexpansion

cd /d "%~dp0"

echo ==^> Looking for a JDK 21...

REM ---- 1. Find an existing JDK ------------------------------------------------
set "JDK_DIR="

REM Try JAVA_HOME first.
if defined JAVA_HOME (
    if exist "%JAVA_HOME%\bin\javac.exe" (
        "%JAVA_HOME%\bin\javac.exe" -version 2>nul | findstr /R "21\. 22\. 23\." >nul
        if not errorlevel 1 set "JDK_DIR=%JAVA_HOME%"
    )
)

REM Try PATH.
if "!JDK_DIR!"=="" (
    for /f "delims=" %%i in ('where javac 2^>nul') do (
        set "JAVAC_PATH=%%i"
        goto :check_javac
    )
)
:check_javac
if "!JDK_DIR!"=="" if defined JAVAC_PATH (
    "%JAVAC_PATH%" -version 2>nul | findstr /R "21\. 22\. 23\." >nul
    if not errorlevel 1 (
        for %%j in ("!JAVAC_PATH!") do set "JDK_DIR=%%~dpj.."
    )
    set "JAVAC_PATH="
)

REM Try common install locations.
if "!JDK_DIR!"=="" (
    for %%p in (
        "C:\Program Files\Eclipse Adoptium\jdk-21*"
        "C:\Program Files\Microsoft\jdk-21*"
        "C:\Program Files\Java\jdk-21*"
        "C:\Program Files\BellSoft\LibericaJDK-21*"
    ) do (
        if exist "%%~p\bin\javac.exe" (
            "%%~p\bin\javac.exe" -version 2>nul | findstr /R "21\. 22\. 23\." >nul
            if not errorlevel 1 set "JDK_DIR=%%~p"
        )
    )
)

REM ---- 2. Download JDK 21 if needed ------------------------------------------
if "!JDK_DIR!"=="" (
    echo     No system JDK 21 found. Downloading a portable JDK 21...
    if not exist "build-tools" mkdir "build-tools"
    cd "build-tools"

    REM Try multiple mirrors.
    set "URLS=https://github.com/adoptium/temurin21-binaries/releases/download/jdk-21.0.5%%2B11/OpenJDK21U-jdk_x64_windows_hotspot_21.0.5_11.zip https://aka.ms/download-jdk/microsoft-jdk-21.0.5-windows-x64.zip"
    set success=0

    for %%u in (!URLS!) do (
        echo     Trying: %%u
        powershell -NoProfile -ExecutionPolicy Bypass -Command "try { (Invoke-WebRequest -UseBasicParsing -Uri '%%u' -OutFile 'jdk.zip' -TimeoutSec 240); if ((Get-Item jdk.zip).Length -gt 1000000) { exit 0 } else { exit 1 } } catch { exit 1 }"
        if !errorlevel!==0 (
            echo     Downloaded. Extracting...
            powershell -NoProfile -ExecutionPolicy Bypass -Command "Expand-Archive -Path 'jdk.zip' -DestinationPath '.' -Force"
            if exist jdk-* (
                if exist jdk rmdir /s /q jdk
                ren jdk-* jdk
                del jdk.zip
                set success=1
                goto :jdk_ready
            )
        )
        if exist jdk.zip del jdk.zip
        if exist jdk rmdir /s /q jdk
    )

    :jdk_ready
    cd ..
    if !success!==0 (
        echo.
        echo ERROR: Could not download a JDK 21 automatically.
        echo Please install one manually from https://adoptium.net/
        exit /b 1
    )
    set "JDK_DIR=%CD%\build-tools\jdk"
)

echo     Using JDK at: !JDK_DIR!
set "JAVA_HOME=!JDK_DIR!"
set "PATH=!JAVA_HOME!\bin;%PATH%"

echo.
echo ==^> JDK version:
"!JAVA_HOME!\bin\javac.exe" -version
"!JAVA_HOME!\bin\java.exe" -version

REM ---- 3. Generate Gradle wrapper & build ------------------------------------
echo.
echo ==^> Generating Gradle wrapper...
gradle wrapper --gradle-version 8.7
if errorlevel 1 goto :err

echo. >> gradle.properties
echo org.gradle.java.home=!JAVA_HOME!>> gradle.properties

echo.
echo ==^> Building Spidey Sense Mod for Minecraft 1.21.1...
call .\gradlew.bat build --no-daemon
if errorlevel 1 goto :err

echo.
echo ============================================================================
echo   BUILD COMPLETE
echo ============================================================================
echo.
echo   Runnable mod JAR:  build\libs\spidey-sense-mod-1.1.0.jar
echo.
echo   Install:
echo     1. Install Fabric Loader 0.16+ and Fabric API for Minecraft 1.21.1
echo     2. Drop the JAR into your .minecraft\mods\ folder
echo     3. Launch Minecraft 1.21.1
echo     4. Hold V to charge, release to activate Spidey Sense!
echo.
exit /b 0

:err
echo Build failed.
exit /b 1
