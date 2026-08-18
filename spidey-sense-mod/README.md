# 🕷️ Spidey Sense Mod (Minecraft Fabric)

A Minecraft Fabric mod inspired by **Miles Morales' spider-sense** from *Spider-Verse*.
**Hold V** to charge up your bioelectric spidey sense, **release to trigger**. Longer
hold = longer effect, longer cooldown, more intense visuals. The full effect drops
you into a comic-book freeze frame packed with halftone dots, bioelectric lightning,
speed lines, comic text, and a thick black panel border around the screen.

> ⚠️ This mod lives in the `spidey-sense-mod/` subfolder of this repo. The rest of the
> repo is unrelated (a web game called *Greedy Growers*) — you only need this folder.

---

## ✨ Every aspect of Miles Morales' spidey sense

### The tingle / charge-up
- 🎯 **Hold V** to charge — ring converges from screen edges toward centre
- ⏱️ Tick-tock sound accelerates as you near max charge
- ⚡ Vibration particles spawn around you as you build up
- 💥 Auto-fires at max charge (3 seconds) if you don't release

### The freeze frame (when you release)
- ⏱️ **Time freeze** — world ticks stop, you keep moving
- 🎯 **Glowing hostiles** — every hostile within 30 blocks glows red through walls
- 🟥 **Danger tint** — subtle red/orange wash over the whole screen
- ☁️ **Yellow halftone dots** — comic-book shading field that breathes
- ⚡ **Bioelectric veins** — procedural yellow lightning flickering across the view
- ⚡ **Speed lines** — 32 radial yellow lines rotating slowly
- 💥 **Burst ring** — expanding yellow ring shoots from screen centre
- 💥 **White flash** — brief brightness pop on the first frame
- 🔍 **Zoom pulse** — HUD punches in ~6%, snaps back
- 📳 **Screen shake** — subtle shake that fades over the duration
- 🟧 **Three-layered vignette** — wide red halo + sharp red edge + thin yellow highlight
- 🖼️ **Comic panel border** — thick black border with yellow inner accent
- 💬 **Comic pop-ups** — "POW!" "WHAM!" "BAM!" "ZAP!" "THWIP!" "BZZZT!" "KAPOW!"...
- 🏷️ **Title splash** — giant "SPIDER-SENSE!" / "DANGER!" / "WEB OF AWARENESS!"
- ⚠️ **INCOMING!** — red warning when a projectile is flying at you
- 🔥 **Hostile particles** — red flames around every detected mob
- 🕸️ **Vibration particles** — constant sparks around you the whole time
- 🏁 **DODGED!** finisher when the effect ends

### Mechanics
- 🍗 **Cost:** 1 hunger point at full charge (free below 30% charge)
- ⏲️ **Cooldown:** scales with charge (5s quick → 30s full)
- 🎮 **Controls:** `V` (configurable in Options → Controls → Spidey Sense)

---

## 🛠️ Build it

You need **JDK 17+** and **Gradle 8.x** installed.

```bash
cd spidey-sense-mod
gradle wrapper        # one-time — creates gradlew, gradlew.bat, gradle-wrapper.jar
./gradlew build
```

The compiled jar lands in `build/libs/spidey-sense-mod-1.0.0.jar`.

---

## 📦 Install it

1. Install [Fabric Loader](https://fabricmc.net/use/) for **Minecraft 1.20.1**
2. Install [Fabric API](https://modrinth.com/mod/fabric-api) for 1.20.1
3. Drop `spidey-sense-mod-1.0.0.jar` into your `.minecraft/mods/` folder
4. Launch Minecraft

**Tip:** the more you hold V before releasing, the better the effect. Try a 1.5
second hold for the full spider-sense experience.

---

## 🔧 Tuning

Open `SpideySenseHandler.java` and tweak the constants at the top:

```java
public static final int DURATION_TICKS = 60;          // 3s at full charge
public static final int COOLDOWN_TICKS = 20 * 30;     // 30s at full charge
public static final int DETECT_RADIUS = 30;
public static final int HUNGER_COST = 1;
public static final int CHARGE_DURATION = 30;        // 1.5s to full charge
public static final int QUICK_DURATION = 20;         // 1s quick-effect duration
public static final int QUICK_COOLDOWN = 20 * 5;     // 5s quick-effect cooldown
```

Edit the comic vocabulary in `SpideySenseComicText.java`:

```java
private static final String[] ACTION_WORDS = {
    "POW!", "WHAM!", "BAM!", "ZAP!", ...
};
private static final String[] TITLE_WORDS = {
    "SPIDER-SENSE!", "DANGER!", "WEB OF AWARENESS!"
};
```

Rebuild with `./gradlew build` after editing.

---

## 📁 Project layout

```
spidey-sense-mod/
├── build.gradle
├── settings.gradle
├── gradle.properties
├── gradle/wrapper/gradle-wrapper.properties
└── src/main/
    ├── java/com/spideysense/
    │   ├── SpideySenseMod.java                  ← entrypoint
    │   └── client/
    │       ├── SpideySenseKeybinds.java         ← V key
    │       ├── SpideySenseHandler.java          ← charge state, cooldown, particles, projectile check
    │       ├── SpideySenseComicText.java        ← floating pop-ups (POW/WHAM/INCOMING/DODGED)
    │       └── SpideySenseOverlay.java          ← all 10 visual layers
    └── resources/
        ├── fabric.mod.json
        └── assets/spideysense/
            ├── icon.png
            └── lang/en_us.json
```

---

## 📜 License

MIT — do whatever you want with it.
