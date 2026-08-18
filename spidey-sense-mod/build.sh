#!/bin/bash
# Build script for Spidey Sense Mod (Minecraft 1.20.1 Fabric).
#
# Requirements on the build machine:
#   - JDK 17 or newer  (Temurin / OpenJDK / Oracle — any will do)
#   - Internet access (Gradle needs to download Minecraft, Yarn, Fabric API)
#   - Gradle 8.x     (or just run `gradle wrapper` once as shown below)
#
# After building, the runnable mod JAR will be at:
#   build/libs/spidey-sense-mod-1.0.0.jar
#
# Install:
#   1. Install Fabric Loader for Minecraft 1.20.1
#   2. Install Fabric API for 1.20.1
#   3. Drop the JAR into .minecraft/mods/

set -e

cd "$(dirname "$0")"

echo "==> Generating Gradle wrapper (one-time)"
gradle wrapper --gradle-version 8.5

echo ""
echo "==> Building mod"
./gradlew build --no-daemon

echo ""
echo "==> Build complete!"
echo "    Runnable mod JAR: build/libs/spidey-sense-mod-1.0.0.jar"
echo ""
echo "Install instructions:"
echo "  1. Drop the JAR into your .minecraft/mods/ folder."
echo "  2. Make sure Fabric Loader 0.14+ and Fabric API are also in mods/."
echo "  3. Launch Minecraft 1.20.1."
echo "  4. In-game, press V to charge up your Spidey Sense!"
