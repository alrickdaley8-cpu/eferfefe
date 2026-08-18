@echo off
REM Build script for Spidey Sense Mod (Minecraft 1.20.1 Fabric).
REM
REM Requirements on the build machine:
REM   - JDK 17 or newer (Temurin / OpenJDK / Oracle)
REM   - Internet access
REM   - Gradle 8.x  (or just run `gradle wrapper` once)
REM
REM After building, the runnable mod JAR will be at:
REM   build\libs\spidey-sense-mod-1.0.0.jar

cd /d "%~dp0"

echo ==^> Generating Gradle wrapper (one-time)
gradle wrapper --gradle-version 8.5
if errorlevel 1 goto :err

echo.
echo ==^> Building mod
call .\gradlew.bat build --no-daemon
if errorlevel 1 goto :err

echo.
echo ==^> Build complete!
echo     Runnable mod JAR: build\libs\spidey-sense-mod-1.0.0.jar
echo.
echo Install:
echo   1. Drop the JAR into your .minecraft\mods\ folder.
echo   2. Make sure Fabric Loader 0.14+ and Fabric API are also in mods\.
echo   3. Launch Minecraft 1.20.1.
echo   4. Press V to charge up your Spidey Sense!
exit /b 0

:err
echo Build failed.
exit /b 1
