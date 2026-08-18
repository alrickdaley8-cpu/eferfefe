#!/usr/bin/env python3
"""
Build a JAR file for the Spidey Sense mod.

Since this environment doesn't have a JDK, we can't compile to bytecode.
We assemble a *source JAR* containing all .java sources, resources, build
scripts, and a proper META-INF/MANIFEST.MF. The user can compile it on
their own machine with JDK 21 + Gradle.

The resulting JAR is a real JAR file with the correct structure; only the
final bytecode compilation step needs to happen on the user's machine.
"""

import os
import re
import zipfile
import sys

# Paths.
ROOT = "/home/user/eferfefe"
PROJECT_DIR = os.path.join(ROOT, "spidey-sense-mod")
OUTPUT_JAR = os.path.join(ROOT, "spidey-sense-mod-source.jar")

# A list of (source_path_in_project, arcname_in_jar) pairs.
# We omit the enclosing "spidey-sense-mod/" directory so the JAR's root
# contains the project directly.
FILES = [
    # Build scripts at the JAR root.
    ("build.sh",                    "build.sh"),
    ("build.bat",                   "build.bat"),
    ("LICENSE",                     "LICENSE"),

    # Gradle build files at the JAR root.
    ("build.gradle",                "build.gradle"),
    ("settings.gradle",             "settings.gradle"),
    ("gradle.properties",           "gradle.properties"),
    ("gradle/wrapper/gradle-wrapper.properties",
                                   "gradle/wrapper/gradle-wrapper.properties"),

    # Java sources.
    ("src/main/java/com/spideysense/SpideySenseMod.java",
                                   "src/main/java/com/spideysense/SpideySenseMod.java"),
    ("src/main/java/com/spideysense/client/SpideySenseKeybinds.java",
                                   "src/main/java/com/spideysense/client/SpideySenseKeybinds.java"),
    ("src/main/java/com/spideysense/client/SpideySenseHandler.java",
                                   "src/main/java/com/spideysense/client/SpideySenseHandler.java"),
    ("src/main/java/com/spideysense/client/SpideySenseComicText.java",
                                   "src/main/java/com/spideysense/client/SpideySenseComicText.java"),
    ("src/main/java/com/spideysense/client/SpideySenseOverlay.java",
                                   "src/main/java/com/spideysense/client/SpideySenseOverlay.java"),

    # Resources.
    ("src/main/resources/fabric.mod.json",
                                   "src/main/resources/fabric.mod.json"),
    ("src/main/resources/assets/spideysense/icon.png",
                                   "src/main/resources/assets/spideysense/icon.png"),
    ("src/main/resources/assets/spideysense/lang/en_us.json",
                                   "src/main/resources/assets/spideysense/lang/en_us.json"),
]


def read_gradle_version():
    """Read mod_version and minecraft_version from gradle.properties."""
    gp = os.path.join(PROJECT_DIR, "gradle.properties")
    props = {}
    with open(gp) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                props[k.strip()] = v.strip()
    return props.get("mod_version", "1.0.0"), props.get("minecraft_version", "?")


def main():
    mod_version, mc_version = read_gradle_version()

    manifest = f"""Manifest-Version: 1.0
Created-By: Arena AI Agent
Built-By: Arena AI Agent
Build-Tool: source-jar (pre-compile)
Specification-Title: Spidey Sense Mod
Specification-Version: {mod_version}
Implementation-Title: com.spideysense.SpideySenseMod
Implementation-Version: {mod_version}
Target-Minecraft: {mc_version}

"""

    if os.path.exists(OUTPUT_JAR):
        os.remove(OUTPUT_JAR)

    written = []
    missing = []
    with zipfile.ZipFile(OUTPUT_JAR, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as jar:
        # META-INF/MANIFEST.MF first.
        jar.writestr("META-INF/MANIFEST.MF", manifest)

        for src_rel, arcname in FILES:
            src_abs = os.path.join(PROJECT_DIR, src_rel)
            if not os.path.exists(src_abs):
                missing.append(src_rel)
                continue
            jar.write(src_abs, arcname)
            written.append(arcname)

    size = os.path.getsize(OUTPUT_JAR)
    print(f"Created JAR: {OUTPUT_JAR}")
    print(f"  Target Minecraft: {mc_version}")
    print(f"  Mod version:      {mod_version}")
    print(f"  Size:             {size:,} bytes ({size / 1024:.1f} KB)")
    print(f"  Files:            {len(written)} + manifest")
    if missing:
        print(f"  MISSING: {missing}", file=sys.stderr)
        return 1
    print()
    print(f"To build a runnable mod JAR for MC {mc_version}:")
    print("  1. Extract this JAR somewhere")
    print("  2. cd into the extracted folder")
    print("  3. ./build.sh      (or build.bat on Windows)")
    print(f"  4. The runnable mod JAR will appear at build/libs/spidey-sense-mod-{mod_version}.jar")
    print()
    print("  Requirements on your build machine: JDK 21 + Internet + Gradle 8.x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
