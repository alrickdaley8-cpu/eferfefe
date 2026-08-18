package com.spideysense.client;

import net.minecraft.client.MinecraftClient;
import net.minecraft.client.font.TextRenderer;
import net.minecraft.client.gui.DrawContext;
import net.minecraft.util.math.RotationAxis;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

/**
 * Manages the floating comic-book "POW! / WHAM! / THWIP!" style pop-ups that
 * explode onto the player's screen whenever Spidey Sense is active. Each pop-up
 * has its own position, scale, rotation, lifetime, and colour. We render them
 * with a thick black outline so they look like a comic-book speech burst.
 */
public final class SpideySenseComicText {
    private SpideySenseComicText() {}

    private static final List<Pop> POPS = new ArrayList<>();
    private static final Random RANDOM = new Random();

    // Words used during the 3-second freeze (cycle through these randomly).
    private static final String[] ACTION_WORDS = {
            "POW!", "WHAM!", "BAM!", "ZAP!", "ZOWIE!", "CRACK!",
            "BOOM!", "THWIP!", "TINGLE!", "BZZZT!", "SNAP!",
            "CRACKLE!", "KAPOW!", "BIFF!", "DANGER!"
    };

    // Big title words that announce the ability when it activates.
    private static final String[] TITLE_WORDS = {
            "SPIDER-SENSE!", "TINGLING!", "DANGER!", "WEB OF AWARENESS!"
    };

    // Classic comic-book colours. Pops pick one at random.
    private static final int[] COMIC_COLOURS = {
            0xFFFFEB3B, // bright yellow
            0xFFFF1744, // bright red
            0xFF00E5FF, // cyan
            0xFFFF6F00, // deep orange
            0xFFE91E63, // hot pink
            0xFFFFFFFF  // white (with outline)
    };

    /** Spawn a giant title pop-up in the centre of the screen. */
    public static void spawnTitle(MinecraftClient client) {
        String text = TITLE_WORDS[RANDOM.nextInt(TITLE_WORDS.length)];
        float x = client.getWindow().getScaledWidth() / 2f;
        float y = client.getWindow().getScaledHeight() / 2f;
        int colour = COMIC_COLOURS[RANDOM.nextInt(COMIC_COLOURS.length)];
        float rotation = (RANDOM.nextFloat() - 0.5f) * 12f;   // ±6°
        POPS.add(new Pop(text, x, y, colour, 1.0f, rotation, 40, true));
    }

    /** Spawn a random small action word somewhere on screen. */
    public static void spawnRandom(MinecraftClient client) {
        String text = ACTION_WORDS[RANDOM.nextInt(ACTION_WORDS.length)];
        int sw = client.getWindow().getScaledWidth();
        int sh = client.getWindow().getScaledHeight();
        // Bias toward the upper-middle / left-right so it doesn't sit on the hotbar.
        float x = sw * (0.18f + RANDOM.nextFloat() * 0.64f);
        float y = sh * (0.18f + RANDOM.nextFloat() * 0.45f);
        float scale = 1.6f + RANDOM.nextFloat() * 1.2f;       // 1.6 – 2.8
        float rotation = (RANDOM.nextFloat() - 0.5f) * 28f;   // ±14°
        int colour = COMIC_COLOURS[RANDOM.nextInt(COMIC_COLOURS.length)];
        POPS.add(new Pop(text, x, y, colour, scale, rotation, 22, false));
    }

    /** Spawn a "DODGED!" type pop-up on deactivation. */
    public static void spawnFinisher(MinecraftClient client) {
        POPS.add(new Pop("DODGED!", client.getWindow().getScaledWidth() / 2f,
                client.getWindow().getScaledHeight() / 2f,
                0xFFFFEB3B, 2.5f, (RANDOM.nextFloat() - 0.5f) * 20f, 24, true));
    }

    /**
     * Spawn a big red "INCOMING!" warning when a projectile is flying at the
     * player. We don't stack these — at most one INCOMING! is alive at a time.
     */
    public static void spawnIncoming() {
        for (Pop p : POPS) {
            if ("INCOMING!".equals(p.text)) return;
        }
        MinecraftClient client = MinecraftClient.getInstance();
        int sw = client.getWindow().getScaledWidth();
        int sh = client.getWindow().getScaledHeight();
        float rotation = (RANDOM.nextFloat() - 0.5f) * 14f;
        POPS.add(new Pop("INCOMING!", sw / 2f, sh * 0.32f, 0xFFFF1744, 2.2f, rotation, 20, true));
    }

    /** Age every pop by one tick and remove dead ones. */
    public static void tick() {
        Iterator<Pop> it = POPS.iterator();
        while (it.hasNext()) {
            Pop p = it.next();
            p.age++;
            if (p.age >= p.maxAge) it.remove();
        }
    }

    /** Clear all pop-ups (e.g. when leaving the world). */
    public static void clear() {
        POPS.clear();
    }

    /** Cheap check used by the overlay to know whether to keep rendering. */
    public static boolean hasLivePops() {
        return !POPS.isEmpty();
    }

    /** Draw every live pop-up. Called from the HUD overlay. */
    public static void render(DrawContext ctx, TextRenderer tr) {
        for (Pop p : POPS) {
            float progress = (float) p.age / p.maxAge;
            float scale = p.isTitle ? titleScale(progress) : actionScale(progress);
            if (scale <= 0.001f) continue;
            drawComicText(ctx, tr, p.text, p.x, p.y, p.colour, scale, p.rotation);
        }
    }

    /**
     * Big title animation: 0 → 4× (overshoot), settle to 3.4×, hold, shrink out.
     * Looks like a comic-book impact word exploding onto screen.
     */
    private static float titleScale(float t) {
        if (t < 0.10f) return easeOutBack(t / 0.10f) * 4.0f;
        if (t < 0.25f) return 4.0f - (t - 0.10f) / 0.15f * 0.6f;       // settle
        if (t < 0.80f) return 3.4f;                                      // hold
        return 3.4f * (1f - (t - 0.80f) / 0.20f);                       // fade out
    }

    /**
     * Small action-word animation: 0.7× → 1.3× pop → 1.0× hold → 0× fade.
     */
    private static float actionScale(float t) {
        if (t < 0.18f) return 0.7f + easeOutBack(t / 0.18f) * 0.6f;
        if (t < 0.35f) return 1.3f - (t - 0.18f) / 0.17f * 0.3f;
        if (t < 0.70f) return 1.0f;
        return 1.0f * (1f - (t - 0.70f) / 0.30f);
    }

    /** Back-easing for that satisfying "snap" pop-in. */
    private static float easeOutBack(float x) {
        float c1 = 1.70158f;
        float c3 = c1 + 1f;
        return 1f + c3 * (float) Math.pow(x - 1f, 3) + c1 * (float) Math.pow(x - 1f, 2);
    }

    /**
     * Render a single piece of comic text with a thick black outline, scale, and rotation.
     */
    private static void drawComicText(DrawContext ctx, TextRenderer tr, String text,
                                      float centreX, float centreY, int fillColour,
                                      float scale, float rotationDeg) {
        int width = tr.getWidth(text);
        int height = tr.fontHeight;

        var m = ctx.getMatrices();
        m.push();
        // Move origin to the desired centre of the text.
        m.translate(centreX, centreY, 0);
        m.scale(scale, scale, 1f);
        m.multiply(RotationAxis.POSITIVE_Z.rotationDegrees(rotationDeg));
        // Now move origin so text is centred around it.
        m.translate(-width / 2f, -height / 2f, 0);

        // Draw a thick black outline (16 short offsets) before the fill on top.
        int outline = 0xFF000000;
        int[][] offsets = {
                {-2, 0}, {2, 0}, {0, -2}, {0, 2},
                {-1, -1}, {1, -1}, {-1, 1}, {1, 1},
                {-2, -1}, {2, -1}, {-2, 1}, {2, 1},
                {-1, -2}, {1, -2}, {-1, 2}, {1, 2}
        };
        for (int[] off : offsets) {
            ctx.drawText(tr, text, off[0], off[1], outline, false);
        }
        // The actual comic word on top.
        ctx.drawText(tr, text, 0, 0, fillColour, false);
        m.pop();
    }

    /** A single pop-up. */
    private static final class Pop {
        final String text;
        final float x, y;
        final int colour;
        final float baseScale;
        final float rotation;
        final int maxAge;
        final boolean isTitle;
        int age = 0;

        Pop(String text, float x, float y, int colour, float scale, float rotation,
            int maxAge, boolean isTitle) {
            this.text = text;
            this.x = x;
            this.y = y;
            this.colour = colour;
            this.baseScale = scale;
            this.rotation = rotation;
            this.maxAge = maxAge;
            this.isTitle = isTitle;
        }
    }
}
