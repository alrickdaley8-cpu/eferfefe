#!/bin/bash
# ============================================================================
# Spidey Sense Mod — build script for Minecraft 1.21.1 Fabric
# ============================================================================
#
# This script does everything for you in one shot.
#
# What it does, in order:
#   1. Detects an existing JDK 21 on your machine — uses it if found.
#   2. If no JDK 21 exists, downloads a portable Eclipse Temurin JDK 21
#      into ./build-tools/ and uses that. Tries multiple mirrors so it works
#      even if one is down.
#   3. Generates the Gradle wrapper.
#   4. Builds the mod.
#
# The runnable mod JAR you drop into .minecraft/mods/ will be at:
#   build/libs/spidey-sense-mod-1.1.0.jar
#
# Linux / macOS only. See build.bat for the Windows version.
# ============================================================================

set -e

cd "$(dirname "$0")"

# ----- 1. Find / install a JDK 21 ---------------------------------------------
echo "==> Looking for a JDK 21..."

# A JDK is "good" if `javac -source 21` produces no error.
find_existing_jdk() {
    local candidate
    for candidate in "$JAVA_HOME" "$(command -v javac 2>/dev/null | xargs dirname 2>/dev/null)/.." \
                       /usr/lib/jvm/temurin-21* /usr/lib/jvm/java-21* /usr/lib/jvm/*-21* \
                       /opt/homebrew/opt/openjdk@21 /opt/homebrew/opt/openjdk-21 \
                       /Library/Java/JavaVirtualMachines/*21*/Contents/Home; do
        if [ -x "$candidate/bin/javac" ]; then
            # Test it actually supports --release 21.
            if "$candidate/bin/javac" --release 21 -version /dev/null 2>&1 | grep -q "release 21\| javac 21\| javac 22\| javac 23"; then
                echo "$candidate"
                return 0
            fi
        fi
    done
    return 1
}

JDK_DIR=""
if JDK_DIR=$(find_existing_jdk); then
    echo "    Found existing JDK at: $JDK_DIR"
else
    echo "    No system JDK 21 found. Downloading a portable JDK 21..."

    BUILD_TOOLS=./build-tools
    mkdir -p "$BUILD_TOOLS"
    cd "$BUILD_TOOLS"

    JDK_DIR="$(pwd)/jdk"

    # Try multiple download mirrors in order.
    URLS=(
        # Eclipse Temurin 21 (musl, glibc, hot spot)
        "https://github.com/adoptium/temurin21-binaries/releases/download/jdk-21.0.5%2B11/OpenJDK21U-jdk_x64_linux_hotspot_21.0.5_11.tar.gz"
        # Bellsoft Liberica 21
        "https://github.com/bell-sw/LibericaJDK/releases/download/21.0.5%2B11/bellsoft-jdk21.0.5%2B11-linux-x64.tar.gz"
        # Microsoft openjdk 21
        "https://aka.ms/download-jdk/microsoft-jdk-21.0.5-linux-x64.tar.gz"
    )

    success=0
    for url in "${URLS[@]}"; do
        echo "    Trying: $url"
        archive="$(pwd)/jdk.tar.gz"
        if curl -fL --connect-timeout 8 --max-time 240 -o "$archive" "$url" 2>/dev/null && \
           [ "$(stat -c%s "$archive" 2>/dev/null || echo 0)" -gt 1000000 ]; then
            echo "    Downloaded $(du -h "$archive" | cut -f1); extracting..."
            if tar -xzf "$archive" 2>/dev/null; then
                # The archive usually extracts to a single jdk-* folder.
                extracted=$(find . -maxdepth 1 -type d -name 'jdk-*' | head -1)
                if [ -n "$extracted" ]; then
                    rm -rf jdk
                    mv "$extracted" jdk
                    rm -f jdk.tar.gz
                    success=1
                    break
                fi
            fi
        fi
        rm -rf jdk jdk.tar.gz
    done

    cd ..
    if [ "$success" -ne 1 ]; then
        echo ""
        echo "ERROR: Could not download a JDK 21 automatically."
        echo "Please install one of these manually:"
        echo "  - macOS:  brew install temurin@21"
        echo "  - Ubuntu: sudo apt install openjdk-21-jdk"
        echo "  - Fedora: sudo dnf install java-21-openjdk-devel"
        echo "  - Or pick one from https://adoptium.net/"
        exit 1
    fi
fi

# Point JAVA_HOME and PATH at the JDK we have.
export JAVA_HOME="$JDK_DIR"
export PATH="$JAVA_HOME/bin:$PATH"

# Sanity-check.
echo ""
echo "==> JDK version:"
"$JAVA_HOME/bin/javac" -version
"$JAVA_HOME/bin/java" -version
echo ""

# ----- 2. Generate Gradle wrapper & build ------------------------------------
echo "==> Generating Gradle wrapper..."
gradle wrapper --gradle-version 8.7

# Provide JAVA_HOME to gradlew too via gradle.properties
echo "org.gradle.java.home=$JAVA_HOME" >> gradle.properties

echo ""
echo "==> Building Spidey Sense Mod for Minecraft 1.21.1..."
./gradlew build --no-daemon

# ----- 3. Done ---------------------------------------------------------------
echo ""
echo "============================================================================"
echo "  BUILD COMPLETE"
echo "============================================================================"
echo ""
echo "  Runnable mod JAR:"
echo "    build/libs/spidey-sense-mod-1.1.0.jar"
echo ""
echo "  Install:"
echo "    1. Install Fabric Loader 0.16+ and Fabric API for Minecraft 1.21.1"
echo "    2. Drop the JAR into your .minecraft/mods/ folder"
echo "    3. Launch Minecraft 1.21.1"
echo "    4. Hold V to charge up your Spidey Sense, then release!"
echo ""
