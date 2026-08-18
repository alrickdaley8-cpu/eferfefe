# 🕷️ Spidey Sense Mod — Minecraft 1.21.1 Fabric

A complete Minecraft Fabric mod project — 100% original code, ~67 KB of Java, ~73 KB of resources.

> ## ⚠️ **CRITICAL: this file is a SOURCE PROJECT, not a pre-built mod!**
>
> The `.jar` you downloaded from the GitHub repo is **NOT** a Fabric mod yet. It's a
> project bundle (sources + build files) packed in JAR format. Fabric Loader scans
> for compiled `.class` files at the JAR root — there's none here, only `.java`
> sources in `src/main/java/...`, so the loader correctly rejects it as invalid.
>
> **You build the real mod with one command, described below.** Takes ~2 minutes.

---

## 🚀 Build the runnable mod in one command

### Linux / macOS

```bash
unzip spidey-sense-mod-source.jar -d spidey-sense-mod
cd spidey-sense-mod
./build.sh
```

### Windows

```bat
unzip spidey-sense-mod-source.jar -d spidey-sense-mod
cd spidey-sense-mod
build.bat
```

The script:
1. **Finds** a JDK 21 on your machine — if found, uses it.
2. **Downloads** a portable Temurin JDK 21 into `./build-tools/jdk/` if not (tries 3 mirrors with fallbacks).
3. **Generates** the Gradle wrapper.
4. **Compiles** the mod against Minecraft 1.21.1 + Fabric API 0.110.5.
5. **Outputs** `build/libs/spidey-sense-mod-1.1.0.jar` — drop THAT into `.minecraft/mods/`.

You need: **internet access** (script downloads JDK + Minecraft + Yarn if missing).

---

## 📦 After the build

The runnable mod will be at:

```
build/libs/spidey-sense-mod-1.1.0.jar     ← this is the actual mod
```

Drop it into your **`.minecraft/mods/`** folder. Also install:
- [Fabric Loader 0.16+](https://fabricmc.net/use/) for Minecraft 1.21.1
- [Fabric API for 1.21.1](https://modrinth.com/mod/fabric-api)

Launch Minecraft 1.21.1 → press **V** to charge → release for full Spider-Verse effect.

---

## ✨ Features

| Category | What's in it |
|---|---|
| **Activation** | Hold V to charge a glowing ring converging on screen edges, release to trigger. Tick-tock sound accelerates with charge. Auto-fires at max charge. |
| **Time freeze** | World ticks stop for 3 seconds (full charge), you keep moving freely |
| **Hostile reveal** | Every hostile within 30 blocks glows red through walls |
| **Auto-dodge** | Resistance IV + Absorption IV during effect; velocity sidestep on inbound projectiles |
| **17 visual layers** | Danger tint, halftone dots, bioelectric veins, sky lightning, 48 speed lines, double burst ring, 4-layer vignette, white flash, glitch tears, panel splits, comic panel border, web corners, spider eyes, spider logo, brightness flickers, comic pop-ups, web crosshair |
| **Comic pop-ups** | 30 action words (POW!, WHAM!, BAM!, THWIP!, KAPOW!, SHAZAM!...) + 8 titles + INCOMING! warning + DODGED! finisher |
| **Suit aura** | Continuous red dust on chest, blue dust on legs, white web accents — looks like the Spider-Man suit is glowing around you |
| **Particles** | 130-particle web burst on activation, flame particles around hostiles, vibration particles around you every tick |
| **Misc** | Hunger cost scales with charge, screen shake, zoom pulse, charge-up ring |

---

## 📁 Project layout (the source JAR)

```
spidey-sense-mod-source.jar
├── README.md, LICENSE                            ← you're reading it
├── META-INF/MANIFEST.MF                          ← well-formed JAR manifest
├── build.sh, build.bat                           ← one-command build scripts
├── build.gradle, settings.gradle, gradle.properties
├── gradle/wrapper/gradle-wrapper.properties
├── src/main/java/com/spideysense/
│   ├── SpideySenseMod.java                       ← entrypoint
│   └── client/
│       ├── SpideySenseKeybinds.java              ← V key
│       ├── SpideySenseHandler.java               ← state machine + particles + auto-dodge
│       ├── SpideySenseComicText.java             ← pop-up system
│       └── SpideySenseOverlay.java               ← all 17 visual layers
└── src/main/resources/
    ├── fabric.mod.json
    └── assets/spideysense/
        ├── icon.png
        └── lang/en_us.json
```

---

## 🔧 Source code stats
- **5 Java files** totaling ~67 KB
- **3 resource files** totaling ~2.4 MB (mostly the icon)
- All handwritten for this project — no copy-pasted Minecraft code

---

## 📜 License

MIT.
