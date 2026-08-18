# 🕷️ Spidey Sense Mod — MAXED OUT (Minecraft 1.21.1 Fabric)

The most extreme version of Miles Morales' spider-sense as a Minecraft Fabric mod.
**Hold V** to charge up your bioelectric spidey sense, **release to trigger** —
the screen explodes into a 17-layer comic-book freeze frame with halftone dots,
lightning, glitch tears, panel splits, glowing eyes, a watermark logo, a custom
crosshair, a glowing red+blue Spider-Man suit aura, and 30 different comic words
flying around — while you freeze time and auto-dodge every incoming attack.

> ⚠️ This mod lives in the `spidey-sense-mod/` subfolder of this repo. The rest of
> the repo is unrelated (a web game called *Greedy Growers*) — you only need this folder.

---

## ✨ Every aspect of Miles Morales — maxed

### The tingle / charge-up
- 🎯 **Hold V** to charge — ring converges from screen edges toward centre
- ⏱️ Tick-tock sound accelerates as you near max charge
- ⚡ Vibration particles spawn around you as you build up
- 💥 Auto-fires at max charge (3 seconds) if you don't release
- ✨ White border pulses around screen when fully charged

### The freeze frame — 17 visual layers
| # | Layer |
|---|---|
| 1 | Red/orange danger tint over whole screen |
| 2 | Yellow **halftone dot field** (comic-book shading, 22px spacing, breathes) |
| 3 | Procedural **bioelectric vein lightning**, regenerated every 60ms |
| 4 | Vertical **sky lightning bolts** (regenerated every 130ms) |
| 5 | **48 radial speed lines** rotating every 2.8s |
| 6 | **Double expanding comic burst rings** (yellow + orange) |
| 7 | **Four-layered vignette** (red halo → red edge → orange → yellow) |
| 8 | Brief **white flash** on activation |
| 9 | **Glitch tear bars** — random horizontal screen tears |
| 10 | **Multi-panel split lines** — two black horizontal lines dividing screen |
| 11 | Thick **black comic panel border** with yellow inner accent |
| 12 | Procedural **spider-web corners** — 4 corner webs |
| 13 | **Glowing red spider eyes** at top-centre |
| 14 | **Spider-logo watermark** in bottom-right corner |
| 15 | **Random brightness flickers** (~20% chance every 70ms) |
| 16 | **Floating comic pop-ups** with colour cycling |
| 17 | **Yellow spider-web crosshair** drawn over centre |

### The comic words
- 🏷️ **Title splash** — SPIDER-SENSE!, TINGLING!, DANGER!, WEB OF AWARENESS!, BIO-ELECTRIC!, VENOM STING!, AWARENESS!
- 💥 **Action words** — POW, WHAM, BAM, ZAP, ZOWIE, CRACK, BOOM, THWIP, TINGLE, BZZZT, SNAP, CRACKLE, KAPOW, BIFF, DANGER, WHOOSH, ZING, FWOOSH, BLAM, SOCK, BONK, WHIP, CRACK-BOOM, SHAZAM, KAZAM, YEOW, OUCH, ZONK, KERPOW (30 words!)
- ⚠️ **INCOMING!** — red warning when a projectile is flying at you
- 🏁 **DODGED!** — big yellow text when the effect ends

### The Spider-Man suit
- 🦸 **Red dust** on upper body (chest), **blue dust** on lower body (legs), **white web accents** continuously rain around the player while spidey sense is active
- 🦸 On activation, 50 suit dust particles **converge inward** from outside — looks like the suit is powering up

### The auto-dodge
- 🛡️ **Resistance IV** + **Absorption IV** during effect → effectively untouchable
- 🏃 When a projectile is detected within 8 blocks and heading toward the player, a perpendicular velocity kick is applied so the player **sidesteps out of the way**
- Dodge direction is auto-chosen to move **away** from the projectile

### The mechanics
- 🎯 Hostiles within 30 blocks **glow red through walls**
- ⏱️ **Time freezes** for 3 seconds (full charge)
- 🔥 **Red flame particles** around every detected hostile
- 🕸️ **130 web-burst particles** on activation
- ⚡ **Continuous vibration particles** around the player every tick
- 🍗 **Hunger cost** scales with charge (free for very brief pulses)
- ⏲️ **Cooldown** scales with charge (5s quick → 30s full)
- 📳 Screen shake, zoom pulse, white flash, glitch tears

---

## 🛠️ Build it

```bash
cd spidey-sense-mod
./build.sh        # or build.bat on Windows
```

**Requirements on your build machine:**
- **JDK 21 or newer** (Minecraft 1.21+ requires Java 21)
- Internet access (Gradle downloads Minecraft, Yarn, Fabric API)
- Gradle 8.x (the build script will generate the wrapper for you)

The script runs `gradle wrapper` then `./gradlew build`. The compiled JAR lands in `build/libs/spidey-sense-mod-1.1.0.jar`.

---

## 📦 Install

1. Install [Fabric Loader](https://fabricmc.net/use/) **0.16+** for **Minecraft 1.21.1**
2. Install [Fabric API](https://modrinth.com/mod/fabric-api) for 1.21.1
3. Drop `spidey-sense-mod-1.1.0.jar` into `.minecraft/mods/`
4. Launch Minecraft 1.21.1, hold V for 1.5 seconds, release

---

## 🔧 Tuning

`SpideySenseHandler.java` constants:

```java
public static final int DURATION_TICKS = 60;          // 3s at full charge
public static final int COOLDOWN_TICKS = 20 * 30;     // 30s at full charge
public static final int DETECT_RADIUS = 30;
public static final int HUNGER_COST = 1;
public static final int CHARGE_DURATION = 30;        // 1.5s to full charge
public static final int RANDOM_POP_INTERVAL = 5;      // comic words frequency
```

`SpideySenseComicText.java` — edit vocabulary:

```java
private static final String[] ACTION_WORDS = { "POW!", "WHAM!", ... };
private static final String[] TITLE_WORDS = { "SPIDER-SENSE!", ... };
```

---

## 📁 Project layout

```
spidey-sense-mod/
├── build.gradle           # fabric-loom 1.7, Java 21
├── settings.gradle
├── gradle.properties      # minecraft 1.21.1, yarn 1.21.1+build.3, loader 0.16.5
├── gradle/wrapper/gradle-wrapper.properties
├── build.sh, build.bat    # one-command build scripts
└── src/main/
    ├── java/com/spideysense/
    │   ├── SpideySenseMod.java                  ← entrypoint
    │   └── client/
    │       ├── SpideySenseKeybinds.java         ← V key
    │       ├── SpideySenseHandler.java          ← state, particles, projectile check, suit dust, auto-dodge
    │       ├── SpideySenseComicText.java        ← 30 action words + colour cycling
    │       └── SpideySenseOverlay.java          ← all 17 visual layers
    └── resources/
        ├── fabric.mod.json
        └── assets/spideysense/{icon.png, lang/en_us.json}
```

---

## 📜 License

MIT.
