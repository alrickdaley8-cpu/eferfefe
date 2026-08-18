#!/usr/bin/env python3
"""
Build a JAR file for the Spidey Sense mod.

Since this environment doesn't have a JDK, we can't compile to bytecode.
We assemble a *source JAR* containing all .java sources, resources, build
scripts, and a proper META-INF/MANIFEST.MF. The user can compile it on
their own machine with JDK 17 + Gradle.

The resulting JAR is a real JAR file with the correct structure; only the
final bytecode compilation step needs to happen on the user's machine.
"""

import os
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

# Manifest that Fabric Loader will read to identify this as a Fabric mod JAR.
# Note: fabric.mod.json is what Fabric actually reads; this MANIFEST just makes
# the JAR a well-formed Java archive.
MANIFEST = """Manifest-Version: 1.0
Created-By: Arena AI Agent
Built-By: Arena AI Agent
Build-Tool: source-jar (pre-compile)
Specification-Title: Spidey Sense Mod
Specification-Version: 1.0.0
Implementation-Title: com.spideysense.SpideySenseMod
Implementation-Version: 1.0.0

"""

def main():
    if os.path.exists(OUTPUT_JAR):
        os.remove(OUTPUT_JAR)

    written = []
    missing = []
    with zipfile.ZipFile(OUTPUT_JAR, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as jar:
        # META-INF/MANIFEST.MF first.
        jar.writestr("META-INF/MANIFEST.MF", MANIFEST)

        for src_rel, arcname in FILES:
            src_abs = os.path.join(PROJECT_DIR, src_rel)
            if not os.path.exists(src_abs):
                missing.append(src_rel)
                continue
            jar.write(src_abs, arcname)
            written.append(arcname)

    size = os.path.getsize(OUTPUT_JAR)
    print(f"Created JAR: {OUTPUT_JAR}")
    print(f"  Size: {size:,} bytes ({size / 1024:.1f} KB)")
    print(f"  Files: {len(written)} + manifest")
    if missing:
        print(f"  MISSING: {missing}", file=sys.stderr)
        return 1
    print()
    print("To build a runnable mod JAR on a machine with JDK 17 + Gradle:")
    print("  1. Extract this JAR somewhere")
    print("  2. cd into the extracted folder")
    print("  3. ./build.sh      (or build.bat on Windows)")
    print("  4. The runnable mod JAR will appear at build/libs/spidey-sense-mod-1.0.0.jar")
    return 0


if __name__ == "__main__":
    sys.exit(main())
