package com.spideysense.client;

import com.mojang.blaze3d.systems.RenderSystem;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.font.TextRenderer;
import net.minecraft.client.gui.DrawContext;
import net.minecraft.client.render.BufferBuilder;
import net.minecraft.client.render.BufferRenderer;
import net.minecraft.client.render.GameRenderer;
import net.minecraft.client.render.RenderTickCounter;
import net.minecraft.client.render.Tessellator;
import net.minecraft.client.render.VertexFormat;
import net.minecraft.client.render.VertexFormats;
import org.joml.Matrix4f;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Renders the entire Spider-Verse freeze-frame visual stack — the most
 * maxed-out version of the spidey sense overlay.
 *
 *   Layer stack (back → front, 17 layers):
 *     1.  Red/orange full-screen tint
 *     2.  Yellow halftone dot grid (comic-book shading)
 *     3.  Procedural bioelectric vein lightning
 *     4.  Vertical sky lightning bolts (NEW)
 *     5.  48 rotating radial speed lines
 *     6.  Expanding comic burst ring
 *     7.  Four-layered red/orange/yellow vignette
 *     8.  Brief white flash
 *     9.  Glitch tear bars (NEW)
 *    10.  Multi-panel horizontal split lines (NEW)
 *    11.  Thick black comic panel border
 *    12.  Procedural spider-web corner overlay (NEW)
 *    13.  Glowing red spider-eyes at top of screen (NEW)
 *    14.  Procedural spider-logo watermark (NEW)
 *    15.  Random brightness flickers (NEW)
 *    16.  Floating comic pop-ups with colour cycling
 *    17.  Custom web crosshair during effect (NEW)
 *
 *   Plus a separate charge-up indicator that draws only while V is held.
 */
public final class SpideySenseOverlay {
    private SpideySenseOverlay() {}

    private static final Random RANDOM = new Random();

    // Cached lightning path buffer.
    private static final List<float[]> LIGHTNING_SEGMENTS = new ArrayList<>();
    private static long lightningLastSeed = 0;

    public static void render(DrawContext ctx, RenderTickCounter tickCounter) {
        MinecraftClient client = MinecraftClient.getInstance();
        if (client.player == null || client.world == null) return;

        boolean active = SpideySenseHandler.isActive();
        boolean charging = SpideySenseHandler.isCharging();

        if (!(active || charging || SpideySenseComicText.hasLivePops())) return;

        ctx.getMatrices().push();

        applyShake(ctx, active);
        applyZoom(ctx, active);

        float life = active ? SpideySenseHandler.getActivationProgress() : 0f;

        if (active) {
            // 1. Tint
            renderScreenTint(ctx, life);
            // 2. Halftone
            renderHalftone(ctx, life);
            // 3. Bioelectric veins
            renderBioElectricVeins(ctx, life);
            // 4. Sky lightning
            renderSkyLightning(ctx, life);
            // 5. Speed lines
            renderSpeedLines(ctx, life);
            // 6. Burst ring
            renderBurstRing(ctx, life);
            // 7. Vignette (four layers)
            renderVignette(ctx, life);
            // 8. Flash
            renderFlash(ctx, life);
            // 9. Glitch tears
            renderGlitchTears(ctx, life);
            // 10. Multi-panel split
            renderMultiPanelSplit(ctx, life);
            // 11. Comic panel border
            renderComicPanelBorder(ctx, life);
            // 12. Web corners
            renderWebCorners(ctx, life);
            // 13. Spider eyes
            renderSpiderEyes(ctx, life);
            // 14. Spider logo
            renderSpiderLogo(ctx, life);
            // 15. Flicker
            renderFlicker(ctx, life);
        }

        if (charging) {
            renderChargeIndicator(ctx, SpideySenseHandler.getChargeProgress());
        }

        // 16. Comic pop-ups (with colour cycling)
        SpideySenseComicText.render(ctx, client.textRenderer);

        if (active) {
            // 17. Web crosshair on top of everything
            renderWebCrosshair(ctx, life);
        }

        ctx.getMatrices().pop();
    }

    // ----------------------------------------------------------------- transforms

    private static void applyShake(DrawContext ctx, boolean active) {
        if (!active) return;
        float intensity = SpideySenseHandler.getShakeIntensity();
        if (intensity <= 0f) return;
        float dx = (RANDOM.nextFloat() - 0.5f) * 8f * intensity;
        float dy = (RANDOM.nextFloat() - 0.5f) * 8f * intensity;
        ctx.getMatrices().translate(dx, dy, 0);
    }

    private static void applyZoom(DrawContext ctx, boolean active) {
        if (!active) return;
        float z = SpideySenseHandler.getZoomPulse();
        if (z <= 0f) return;
        int w = ctx.getScaledWindowWidth();
        int h = ctx.getScaledWindowHeight();
        ctx.getMatrices().translate(w / 2f, h / 2f, 0);
        ctx.getMatrices().scale(z, z, 1f);
        ctx.getMatrices().translate(-w / 2f, -h / 2f, 0);
    }

    // ----------------------------------------------------------------- 1. screen tint

    private static void renderScreenTint(DrawContext ctx, float life) {
        int w = ctx.getScaledWindowWidth();
        int h = ctx.getScaledWindowHeight();
        float a = 0.22f * (0.5f + 0.5f * life);
        drawFullColouredQuad(ctx, w, h, 0.55f, 0.10f, 0.05f, a);
    }

    // ----------------------------------------------------------------- 2. halftone (maxed: denser, brighter, faster pulse)

    private static void renderHalftone(DrawContext ctx, float intensity) {
        int w = ctx.getScaledWindowWidth();
        int h = ctx.getScaledWindowHeight();
        int spacing = 22;       // denser
        long time = System.currentTimeMillis();

        Matrix4f matrix = ctx.getMatrices().peek().getPositionMatrix();
        enablePositionColourBlend();
        BufferBuilder buf = Tessellator.getInstance().begin(VertexFormat.DrawMode.QUADS, VertexFormats.POSITION_COLOR);

        for (int x = spacing / 2; x < w; x += spacing) {
            for (int y = spacing / 2; y < h; y += spacing) {
                float wave = (float) (0.5 + 0.5 * Math.sin((x + y + time / 30.0) / 22.0));
                float r = 2.6f + wave * 2.4f;
                float a = intensity * (0.16f + wave * 0.20f);
                buf.vertex(matrix, x - r, y - r, 0).color(1f, 0.85f, 0.30f, a);
                buf.vertex(matrix, x + r, y - r, 0).color(1f, 0.85f, 0.30f, a);
                buf.vertex(matrix, x + r, y + r, 0).color(1f, 0.85f, 0.30f, a);
                buf.vertex(matrix, x - r, y + r, 0).color(1f, 0.85f, 0.30f, a);
            }
        }

        BufferRenderer.drawWithGlobalProgram(buf.end());
        disableBlend();
    }

    // ----------------------------------------------------------------- 3. bioelectric veins (maxed: more paths, faster regen)

    private static void renderBioElectricVeins(DrawContext ctx, float intensity) {
        int w = ctx.getScaledWindowWidth();
        int h = ctx.getScaledWindowHeight();

        long seed = System.currentTimeMillis() / 60;   // faster regen
        if (seed != lightningLastSeed) {
            lightningLastSeed = seed;
            LIGHTNING_SEGMENTS.clear();
            Random rng = new Random(seed);
            int numPaths = 10;                           // more paths
            for (int p = 0; p < numPaths; p++) {
                float x, y;
                int edge = rng.nextInt(4);
                switch (edge) {
                    case 0:  x = rng.nextFloat() * w;            y = 0;             break;
                    case 1:  x = rng.nextFloat() * w;            y = h;             break;
                    case 2:  x = 0;                              y = rng.nextFloat() * h; break;
                    default: x = w;                              y = rng.nextFloat() * h;
                }
                float startX = x, startY = y;
                int segments = 9 + rng.nextInt(10);
                for (int s = 0; s < segments; s++) {
                    float dx = (rng.nextFloat() - 0.5f) * 120f;
                    float dy = (rng.nextFloat() - 0.5f) * 120f;
                    float endX = clamp(x + dx, 0, w);
                    float endY = clamp(y + dy, 0, h);
                    LIGHTNING_SEGMENTS.add(new float[]{startX, startY, endX, endY});
                    startX = endX;
                    startY = endY;
                    x = endX;
                    y = endY;
                }
            }
        }

        if (LIGHTNING_SEGMENTS.isEmpty()) return;

        Matrix4f matrix = ctx.getMatrices().peek().getPositionMatrix();
        enablePositionColourBlend();
        BufferBuilder buf = Tessellator.getInstance().begin(VertexFormat.DrawMode.QUADS, VertexFormats.POSITION_COLOR);

        float a = intensity * 0.90f;
        for (float[] seg : LIGHTNING_SEGMENTS) {
            addThickLine(buf, matrix, seg[0], seg[1], seg[2], seg[3], 2.4f, 1f, 0.95f, 0.30f, a);
        }
        a = intensity * 0.98f;
        for (float[] seg : LIGHTNING_SEGMENTS) {
            addThickLine(buf, matrix, seg[0], seg[1], seg[2], seg[3], 1.0f, 1f, 1f, 0.85f, a);
        }

        BufferRenderer.drawWithGlobalProgram(buf.end());
        disableBlend();
    }

    // ----------------------------------------------------------------- 4. sky lightning bolts (NEW)

    private static final List<float[]> SKY_LIGHTNING = new ArrayList<>();
    private static long skyLightningSeed = 0;

    private static void renderSkyLightning(DrawContext ctx, float intensity) {
        int w = ctx.getScaledWindowWidth();
        int h = ctx.getScaledWindowHeight();
        long seed = System.currentTimeMillis() / 130;
        if (seed != skyLightningSeed) {
            skyLightningSeed = seed;
            SKY_LIGHTNING.clear();
            Random rng = new Random(seed);
            int bolts = 1 + rng.nextInt(2);
            for (int i = 0; i < bolts; i++) {
                float x = rng.nextFloat() * w;
                float boltLength = h * (0.30f + rng.nextFloat() * 0.40f);
                float segments = 6 + rng.nextInt(5);
                float stepY = boltLength / segments;
                float curX = x, curY = 0;
                for (int j = 0; j < segments; j++) {
                    float nextX = curX + (rng.nextFloat() - 0.5f) * 90f;
                    float nextY = curY + stepY;
                    SKY_LIGHTNING.add(new float[]{curX, curY, nextX, nextY});
                    curX = nextX;
                    curY = nextY;
                }
            }
        }
        if (SKY_LIGHTNING.isEmpty()) return;

        Matrix4f matrix = ctx.getMatrices().peek().getPositionMatrix();
        enablePositionColourBlend();
        BufferBuilder buf = Tessellator.getInstance().begin(VertexFormat.DrawMode.QUADS, VertexFormats.POSITION_COLOR);

        for (float[] seg : SKY_LIGHTNING) {
            addThickLine(buf, matrix, seg[0], seg[1], seg[2], seg[3], 4.5f, 1f, 0.95f, 0.40f, intensity * 0.65f);
            addThickLine(buf, matrix, seg[0], seg[1], seg[2], seg[3], 1.8f, 1f, 1f, 0.85f, intensity * 0.95f);
        }
        BufferRenderer.drawWithGlobalProgram(buf.end());
        disableBlend();
    }

    // ----------------------------------------------------------------- 5. speed lines (maxed: more lines, faster rotation)

    private static void renderSpeedLines(DrawContext ctx, float life) {
        int w = ctx.getScaledWindowWidth();
        int h = ctx.getScaledWindowHeight();
        float cx = w / 2f, cy = h / 2f;
        float minDim = Math.min(w, h);
        float innerR = minDim * 0.08f;
        float outerR = minDim * 0.58f;

        int lines = 48;
        float pulse = (float) (0.65 + 0.35 * Math.sin(System.currentTimeMillis() / 220.0));
        float intensity = (0.40f + 0.60f * life) * pulse;
        float rotation = (System.currentTimeMillis() / 2800f) % 360f;   // faster

        Matrix4f matrix = ctx.getMatrices().peek().getPositionMatrix();
        enablePositionColourBlend();
        BufferBuilder buf = Tessellator.getInstance().begin(VertexFormat.DrawMode.QUADS, VertexFormats.POSITION_COLOR);

        for (int i = 0; i < lines; i++) {
            float angle = (float) Math.toRadians((i * 360f / lines) + rotation);
            float perp = angle + (float) Math.PI / 2f;
            float lineWidth = 2.8f + (float) Math.sin(System.currentTimeMillis() / 200.0 + i) * 1.4f;
            float cos = (float) Math.cos(angle);
            float sin = (float) Math.sin(angle);
            float cosP = (float) Math.cos(perp);
            float sinP = (float) Math.sin(perp);

            float x1 = cx + cos * innerR + cosP * lineWidth / 2f;
            float y1 = cy + sin * innerR + sinP * lineWidth / 2f;
            float x2 = cx + cos * innerR - cosP * lineWidth / 2f;
            float y2 = cy + sin * innerR - sinP * lineWidth / 2f;
            float x3 = cx + cos * outerR - cosP * lineWidth / 2f;
            float y3 = cy + sin * outerR - sinP * lineWidth / 2f;
            float x4 = cx + cos * outerR + cosP * lineWidth / 2f;
            float y4 = cy + sin * outerR + sinP * lineWidth / 2f;

            float a = intensity * 0.60f;
            buf.vertex(matrix, x1, y1, 0).color(1f, 0.95f, 0.65f, a);
            buf.vertex(matrix, x2, y2, 0).color(1f, 0.95f, 0.65f, a);
            buf.vertex(matrix, x3, y3, 0).color(1f, 0.95f, 0.65f, 0f);
            buf.vertex(matrix, x4, y4, 0).color(1f, 0.95f, 0.65f, 0f);
        }

        BufferRenderer.drawWithGlobalProgram(buf.end());
        disableBlend();
    }

    // ----------------------------------------------------------------- 6. burst ring (double ring for impact)

    private static void renderBurstRing(DrawContext ctx, float life) {
        float t = SpideySenseHandler.getBurstProgress();
        if (t <= 0f || t >= 1f) return;

        int w = ctx.getScaledWindowWidth();
        int h = ctx.getScaledWindowHeight();
        float cx = w / 2f, cy = h / 2f;
        float maxRadius = (float) Math.hypot(w, h) / 2f;

        // Render two concentric rings for max impact.
        drawRing(ctx, cx, cy, t * maxRadius * 0.95f, 6f + t * 90f, (1f - t) * 0.75f);
        drawRing(ctx, cx, cy, t * maxRadius * 0.70f, 3f + t * 50f, (1f - t) * 0.55f);
    }

    private static void drawRing(DrawContext ctx, float cx, float cy, float radius, float thickness, float alpha) {
        int segments = 80;
        Matrix4f matrix = ctx.getMatrices().peek().getPositionMatrix();
        enablePositionColourBlend();
        BufferBuilder buf = Tessellator.getInstance().begin(VertexFormat.DrawMode.QUADS, VertexFormats.POSITION_COLOR);

        for (int i = 0; i < segments; i++) {
            float a1 = (float) (i * 2 * Math.PI / segments);
            float a2 = (float) ((i + 1) * 2 * Math.PI / segments);
            float cos1 = (float) Math.cos(a1), sin1 = (float) Math.sin(a1);
            float cos2 = (float) Math.cos(a2), sin2 = (float) Math.sin(a2);
            buf.vertex(matrix, cx + cos1 * radius,                 cy + sin1 * radius,                 0).color(1f, 0.95f, 0.55f, alpha);
            buf.vertex(matrix, cx + cos2 * radius,                 cy + sin2 * radius,                 0).color(1f, 0.95f, 0.55f, alpha);
            buf.vertex(matrix, cx + cos2 * (radius + thickness),   cy + sin2 * (radius + thickness),   0).color(1f, 0.70f, 0.10f, 0f);
            buf.vertex(matrix, cx + cos1 * (radius + thickness),   cy + sin1 * (radius + thickness),   0).color(1f, 0.70f, 0.10f, 0f);
        }
        BufferRenderer.drawWithGlobalProgram(buf.end());
        disableBlend();
    }

    // ----------------------------------------------------------------- 7. vignette (4 layers, brighter)

    private static void renderVignette(DrawContext ctx, float life) {
        int w = ctx.getScaledWindowWidth();
        int h = ctx.getScaledWindowHeight();
        float pulse = (float) (0.55 + 0.45 * Math.sin(System.currentTimeMillis() / 180.0));
        float baseAlpha = 0.85f * (0.4f + 0.6f * life) * pulse;

        enablePositionColourBlend();
        Matrix4f matrix = ctx.getMatrices().peek().getPositionMatrix();
        Tessellator tess = Tessellator.getInstance();

        renderVignettePass(tess, matrix, w, h, baseAlpha * 0.50f, 0.28f, 0.90f, 0.20f, 0.05f);
        renderVignettePass(tess, matrix, w, h, baseAlpha * 0.95f, 0.18f, 1.00f, 0.10f, 0.05f);
        renderVignettePass(tess, matrix, w, h, baseAlpha * 0.75f, 0.10f, 1.00f, 0.30f, 0.05f);
        renderVignettePass(tess, matrix, w, h, baseAlpha * 0.55f, 0.06f, 1.00f, 0.85f, 0.20f);

        disableBlend();
    }

    private static void renderVignettePass(Tessellator tess, Matrix4f m, int w, int h,
                                           float alpha, float fade, float r, float g, float b) {
        int edge = (int) (Math.min(w, h) * fade);
        BufferBuilder buf = tess.begin(VertexFormat.DrawMode.QUADS, VertexFormats.POSITION_COLOR);

        addEdgeQuad(buf, m, 0, 0, w, 0, w, edge, 0, edge, alpha, r, g, b);
        addEdgeQuad(buf, m, 0, h - edge, 0, h, w, h, w, h - edge, alpha, r, g, b);
        addEdgeQuad(buf, m, 0, 0, 0, h, edge, h, edge, 0, alpha, r, g, b);
        addEdgeQuad(buf, m, w - edge, 0, w - edge, h, w, h, w, 0, alpha, r, g, b);

        BufferRenderer.drawWithGlobalProgram(buf.end());
    }

    private static void addEdgeQuad(BufferBuilder buf, Matrix4f m,
                                    float x1, float y1, float x2, float y2,
                                    float x3, float y3, float x4, float y4,
                                    float alpha, float r, float g, float b) {
        buf.vertex(m, x1, y1, 0).color(r, g, b, alpha);
        buf.vertex(m, x2, y2, 0).color(r, g, b, alpha);
        buf.vertex(m, x3, y3, 0).color(r, g, b, 0f);
        buf.vertex(m, x4, y4, 0).color(r, g, b, 0f);
    }

    // ----------------------------------------------------------------- 8. flash (maxed: brighter)

    private static void renderFlash(DrawContext ctx, float life) {
        float t = SpideySenseHandler.getFlashProgress();
        if (t <= 0f) return;
        int w = ctx.getScaledWindowWidth();
        int h = ctx.getScaledWindowHeight();
        float a = (float) Math.pow(t, 1.4) * 0.75f;
        drawFullColouredQuad(ctx, w, h, 1f, 1f, 1f, a);
    }

    // ----------------------------------------------------------------- 9. glitch tears (NEW)

    /**
     * Random horizontal "tear" bars across the screen — mimics the
     * screen-tear glitch effect from Spider-Verse frame transitions.
     * Regenerated every ~60ms.
     */
    private static void renderGlitchTears(DrawContext ctx, float intensity) {
        if (intensity <= 0f) return;
        int w = ctx.getScaledWindowWidth();
        int h = ctx.getScaledWindowHeight();

        long seed = System.currentTimeMillis() / 60;
        Random rng = new Random(seed);
        int tears = 2 + rng.nextInt(3);

        for (int i = 0; i < tears; i++) {
            int y = rng.nextInt(h);
            int tearH = 3 + rng.nextInt(10);
            ctx.fill(0, y, w, y + tearH, 0xDD000000);
            // Occasional colored ghost
            if (rng.nextFloat() < 0.35f) {
                int offset = (rng.nextInt(60) - 30);
                int ghost = rng.nextFloat() < 0.5f ? 0x66FF0000 : 0x66FFFF00;
                int left = Math.max(0, offset);
                int right = Math.min(w, w + offset);
                if (right > left) ctx.fill(left, y + 1, right, y + tearH - 1, ghost);
            }
        }
    }

    // ----------------------------------------------------------------- 10. multi-panel split (NEW)

    /**
     * Two horizontal black lines that drift slowly up and down, dividing the
     * screen into 3 comic-book panels. Pure Spider-Verse.
     */
    private static void renderMultiPanelSplit(DrawContext ctx, float intensity) {
        if (intensity <= 0f) return;
        int w = ctx.getScaledWindowWidth();
        int h = ctx.getScaledWindowHeight();
        int t = 5;
        long time = System.currentTimeMillis();

        float line1Y = h * 0.33f + 30f * (float) Math.sin(time / 400.0);
        float line2Y = h * 0.66f + 30f * (float) Math.sin(time / 350.0 + 1);

        ctx.fill(0, (int) (line1Y - t / 2), w, (int) (line1Y + t / 2), 0xFF000000);
        ctx.fill(0, (int) (line2Y - t / 2), w, (int) (line2Y + t / 2), 0xFF000000);

        // Thin yellow accent just under each line.
        int accent = 0x88FFFF00;
        ctx.fill(0, (int) (line1Y + t / 2), w, (int) (line1Y + t / 2 + 2), accent);
        ctx.fill(0, (int) (line2Y + t / 2), w, (int) (line2Y + t / 2 + 2), accent);
    }

    // ----------------------------------------------------------------- 11. comic panel border

    private static void renderComicPanelBorder(DrawContext ctx, float intensity) {
        if (intensity <= 0f) return;
        int w = ctx.getScaledWindowWidth();
        int h = ctx.getScaledWindowHeight();
        int t = 12;
        int aa = 0xFF000000;
        ctx.fill(0, 0, w, t, aa);
        ctx.fill(0, h - t, w, h, aa);
        ctx.fill(0, 0, t, h, aa);
        ctx.fill(w - t, 0, w, h, aa);

        int accent = 0xFFFFAA00;
        ctx.fill(t, t, w - t, t + 2, accent);
        ctx.fill(t, h - t - 2, w - t, h - t, accent);
        ctx.fill(t, t, t + 2, h - t, accent);
        ctx.fill(w - t - 2, t, w - t, h - t, accent);
    }

    // ----------------------------------------------------------------- 12. web corners (NEW)

    /**
     * Procedural spider-web arcs in each corner of the screen. Pure cosmetic
     * frame, doesn't interfere with gameplay.
     */
    private static void renderWebCorners(DrawContext ctx, float intensity) {
        if (intensity <= 0f) return;
        int w = ctx.getScaledWindowWidth();
        int h = ctx.getScaledWindowHeight();
        float size = 70f;
        float alpha = intensity * 0.55f;

        Matrix4f matrix = ctx.getMatrices().peek().getPositionMatrix();
        enablePositionColourBlend();
        BufferBuilder buf = Tessellator.getInstance().begin(VertexFormat.DrawMode.QUADS, VertexFormats.POSITION_COLOR);

        drawWebCorner(buf, matrix, 0, 0, size, 1, 1, alpha);
        drawWebCorner(buf, matrix, w, 0, size, -1, 1, alpha);
        drawWebCorner(buf, matrix, 0, h, size, 1, -1, alpha);
        drawWebCorner(buf, matrix, w, h, size, -1, -1, alpha);

        BufferRenderer.drawWithGlobalProgram(buf.end());
        disableBlend();
    }

    private static void drawWebCorner(BufferBuilder buf, Matrix4f m, float cornerX, float cornerY,
                                       float size, float dirX, float dirY, float alpha) {
        // 4 concentric arcs.
        int numArcs = 4;
        float arcSpan = (float) Math.PI / 2f;
        for (int i = 0; i < numArcs; i++) {
            float r = size * (0.30f + i * 0.18f);
            int segments = 20;
            for (int j = 0; j < segments; j++) {
                float a1 = j * arcSpan / segments;
                float a2 = (j + 1) * arcSpan / segments;
                float x1 = cornerX + dirX * (float) Math.cos(a1) * r;
                float y1 = cornerY + dirY * (float) Math.sin(a1) * r;
                float x2 = cornerX + dirX * (float) Math.cos(a2) * r;
                float y2 = cornerY + dirY * (float) Math.sin(a2) * r;
                addThickLine(buf, m, x1, y1, x2, y2, 1.6f, 1f, 0.92f, 0.30f, alpha);
            }
        }
        // Radial spokes.
        int numSpokes = 6;
        for (int i = 0; i < numSpokes; i++) {
            float a = i * arcSpan / (numSpokes - 1);
            float x1 = cornerX + dirX * (float) Math.cos(a) * size * 0.18f;
            float y1 = cornerY + dirY * (float) Math.sin(a) * size * 0.18f;
            float x2 = cornerX + dirX * (float) Math.cos(a) * size;
            float y2 = cornerY + dirY * (float) Math.sin(a) * size;
            addThickLine(buf, m, x1, y1, x2, y2, 1.6f, 1f, 0.92f, 0.30f, alpha);
        }
    }

    // ----------------------------------------------------------------- 13. spider eyes (NEW)

    /**
     * Two glowing red spider-eyes at the top-centre of the screen, pulsing
     * rapidly. Pulses brighter as the effect wears on.
     */
    private static void renderSpiderEyes(DrawContext ctx, float intensity) {
        if (intensity <= 0.4f) return;
        int w = ctx.getScaledWindowWidth();
        int cx = w / 2;
        int cy = 56;
        int spacing = 26;
        int eyeR = 14;

        float pulse = 0.55f + 0.45f * (float) Math.sin(System.currentTimeMillis() / 90.0);
        float alpha = intensity * 0.80f * pulse;

        Matrix4f matrix = ctx.getMatrices().peek().getPositionMatrix();
        enablePositionColourBlend();
        BufferBuilder buf = Tessellator.getInstance().begin(VertexFormat.DrawMode.QUADS, VertexFormats.POSITION_COLOR);

        addFilledDisc(buf, matrix, cx - spacing, cy, eyeR, 1f, 0.10f, 0.05f, alpha);
        addFilledDisc(buf, matrix, cx + spacing, cy, eyeR, 1f, 0.10f, 0.05f, alpha);

        BufferRenderer.drawWithGlobalProgram(buf.end());
        disableBlend();
    }

    // ----------------------------------------------------------------- 14. spider logo (NEW)

    /**
     * Procedural yellow spider-logo watermark in the bottom-right corner,
     * pulsing gently. Body + 8 radiating legs.
     */
    private static void renderSpiderLogo(DrawContext ctx, float intensity) {
        if (intensity <= 0.5f) return;
        int w = ctx.getScaledWindowWidth();
        int h = ctx.getScaledWindowHeight();
        int size = 52;
        int cx = w - size - 18;
        int cy = h - size - 18;

        float pulse = 0.7f + 0.3f * (float) Math.sin(System.currentTimeMillis() / 220.0);
        float alpha = intensity * 0.70f * pulse;

        Matrix4f matrix = ctx.getMatrices().peek().getPositionMatrix();
        enablePositionColourBlend();
        BufferBuilder buf = Tessellator.getInstance().begin(VertexFormat.DrawMode.QUADS, VertexFormats.POSITION_COLOR);

        // Body — filled small disc.
        addFilledDisc(buf, matrix, cx, cy, 8f, 1f, 0.92f, 0.23f, alpha);

        // 8 radiating legs.
        for (int i = 0; i < 8; i++) {
            float angle = (float) (i * Math.PI / 4f);
            float x1 = cx + (float) Math.cos(angle) * 7f;
            float y1 = cy + (float) Math.sin(angle) * 7f;
            float x2 = cx + (float) Math.cos(angle) * 22f;
            float y2 = cy + (float) Math.sin(angle) * 22f;
            addThickLine(buf, matrix, x1, y1, x2, y2, 3.0f, 1f, 0.92f, 0.23f, alpha);
            // little bend at the end of each leg
            float bendA = angle + 0.5f * (i % 2 == 0 ? 1 : -1);
            float bx = x2 + (float) Math.cos(bendA) * 4f;
            float by = y2 + (float) Math.sin(bendA) * 4f;
            addThickLine(buf, matrix, x2, y2, bx, by, 2.5f, 1f, 0.92f, 0.23f, alpha);
        }

        BufferRenderer.drawWithGlobalProgram(buf.end());
        disableBlend();
    }

    // ----------------------------------------------------------------- 15. flicker (NEW)

    private static void renderFlicker(DrawContext ctx, float intensity) {
        long seed = System.currentTimeMillis() / 70;
        Random rng = new Random(seed);
        if (rng.nextFloat() < 0.20f) {
            int w = ctx.getScaledWindowWidth();
            int h = ctx.getScaledWindowHeight();
            float a = intensity * (0.06f + rng.nextFloat() * 0.10f);
            drawFullColouredQuad(ctx, w, h, 1f, 0.95f, 0.85f, a);
        }
    }

    // ----------------------------------------------------------------- charge indicator

    private static void renderChargeIndicator(DrawContext ctx, float charge) {
        if (charge <= 0f) return;
        int w = ctx.getScaledWindowWidth();
        int h = ctx.getScaledWindowHeight();
        float cx = w / 2f, cy = h / 2f;
        float minDim = Math.min(w, h);
        float radius = minDim * 0.48f * (1f - charge);
        float thickness = 4f + charge * 8f;
        float alpha = 0.55f + 0.4f * charge;
        boolean maxed = charge >= 1f;

        int segments = 96;
        Matrix4f matrix = ctx.getMatrices().peek().getPositionMatrix();
        enablePositionColourBlend();
        BufferBuilder buf = Tessellator.getInstance().begin(VertexFormat.DrawMode.QUADS, VertexFormats.POSITION_COLOR);

        for (int i = 0; i < segments; i++) {
            float a1 = (float) (i * 2 * Math.PI / segments);
            float a2 = (float) ((i + 1) * 2 * Math.PI / segments);
            float cos1 = (float) Math.cos(a1), sin1 = (float) Math.sin(a1);
            float cos2 = (float) Math.cos(a2), sin2 = (float) Math.sin(a2);
            float pulse = maxed ? 1.0f : 0.6f + 0.4f * Math.sin(System.currentTimeMillis() / 80.0 + i * 0.2);
            float aOuter = alpha * pulse;
            buf.vertex(matrix, cx + cos1 * radius,                       cy + sin1 * radius,                       0).color(1f, 0.95f, 0.30f, aOuter);
            buf.vertex(matrix, cx + cos2 * radius,                       cy + sin2 * radius,                       0).color(1f, 0.95f, 0.30f, aOuter);
            buf.vertex(matrix, cx + cos2 * (radius + thickness),         cy + sin2 * (radius + thickness),         0).color(1f, 0.70f, 0.10f, 0f);
            buf.vertex(matrix, cx + cos1 * (radius + thickness),         cy + sin1 * (radius + thickness),         0).color(1f, 0.70f, 0.10f, 0f);
        }
        BufferRenderer.drawWithGlobalProgram(buf.end());
        disableBlend();

        if (maxed) {
            ctx.fill(0, 0, w, 4, 0xFFFFFFFF);
            ctx.fill(0, h - 4, w, h, 0xFFFFFFFF);
            ctx.fill(0, 0, 4, h, 0xFFFFFFFF);
            ctx.fill(w - 4, 0, w, h, 0xFFFFFFFF);
        }
    }

    // ----------------------------------------------------------------- 17. web crosshair (NEW)

    /**
     * A bright yellow spider-web crosshair drawn over the centre of the screen
     * while the effect is active. 4 cardinal + 4 diagonal spokes + a glow ring.
     */
    private static void renderWebCrosshair(DrawContext ctx, float intensity) {
        if (intensity <= 0.3f) return;
        int w = ctx.getScaledWindowWidth();
        int h = ctx.getScaledWindowHeight();
        float cx = w / 2f, cy = h / 2f;
        float size = 14f;
        float thickness = 2.2f;
        float pulse = 0.7f + 0.3f * (float) Math.sin(System.currentTimeMillis() / 120.0);
        float alpha = intensity * 0.85f * pulse;

        Matrix4f matrix = ctx.getMatrices().peek().getPositionMatrix();
        enablePositionColourBlend();
        BufferBuilder buf = Tessellator.getInstance().begin(VertexFormat.DrawMode.QUADS, VertexFormats.POSITION_COLOR);

        // 8 spokes.
        for (int i = 0; i < 8; i++) {
            float angle = (float) (i * Math.PI / 4f);
            float x1 = cx + (float) Math.cos(angle) * 3f;
            float y1 = cy + (float) Math.sin(angle) * 3f;
            float x2 = cx + (float) Math.cos(angle) * size;
            float y2 = cy + (float) Math.sin(angle) * size;
            addThickLine(buf, matrix, x1, y1, x2, y2, thickness, 1f, 0.92f, 0.30f, alpha);
        }
        // Glow ring at centre.
        addFilledDisc(buf, matrix, cx, cy, 3.5f, 1f, 0.95f, 0.30f, alpha * 0.85f);

        BufferRenderer.drawWithGlobalProgram(buf.end());
        disableBlend();
    }

    // ----------------------------------------------------------------- helpers

    private static void enablePositionColourBlend() {
        RenderSystem.enableBlend();
        RenderSystem.defaultBlendFunc();
        RenderSystem.setShaderColor(1f, 1f, 1f, 1f);
        RenderSystem.setShader(GameRenderer::getPositionColorProgram);
    }

    private static void disableBlend() {
        RenderSystem.disableBlend();
    }

    private static void drawFullColouredQuad(DrawContext ctx, int w, int h,
                                             float r, float g, float b, float a) {
        enablePositionColourBlend();
        Matrix4f matrix = ctx.getMatrices().peek().getPositionMatrix();
        BufferBuilder buf = Tessellator.getInstance().begin(VertexFormat.DrawMode.QUADS, VertexFormats.POSITION_COLOR);
        buf.vertex(matrix, 0, 0, 0).color(r, g, b, a);
        buf.vertex(matrix, w, 0, 0).color(r, g, b, a);
        buf.vertex(matrix, w, h, 0).color(r, g, b, a);
        buf.vertex(matrix, 0, h, 0).color(r, g, b, a);
        BufferRenderer.drawWithGlobalProgram(buf.end());
        disableBlend();
    }

    /** Render a thick line as a quad with the given thickness. */
    private static void addThickLine(BufferBuilder buf, Matrix4f m,
                                     float x1, float y1, float x2, float y2,
                                     float thickness, float r, float g, float b, float a) {
        float dx = x2 - x1;
        float dy = y2 - y1;
        float length = (float) Math.sqrt(dx * dx + dy * dy);
        if (length < 0.001f) return;
        float nx = -dy / length * thickness * 0.5f;
        float ny =  dx / length * thickness * 0.5f;
        buf.vertex(m, x1 + nx, y1 + ny, 0).color(r, g, b, a);
        buf.vertex(m, x1 - nx, y1 - ny, 0).color(r, g, b, a);
        buf.vertex(m, x2 - nx, y2 - ny, 0).color(r, g, b, a);
        buf.vertex(m, x2 + nx, y2 + ny, 0).color(r, g, b, a);
    }

    /** Render a filled disc (triangle-fan as quads). */
    private static void addFilledDisc(BufferBuilder buf, Matrix4f m,
                                      float cx, float cy, float radius,
                                      float r, float g, float b, float a) {
        int segments = 20;
        for (int i = 0; i < segments; i++) {
            float a1 = (float) (i * 2 * Math.PI / segments);
            float a2 = (float) ((i + 1) * 2 * Math.PI / segments);
            buf.vertex(m, cx, cy, 0).color(r, g, b, a);
            buf.vertex(m, cx + (float) Math.cos(a1) * radius, cy + (float) Math.sin(a1) * radius, 0).color(r, g, b, a);
            buf.vertex(m, cx + (float) Math.cos(a2) * radius, cy + (float) Math.sin(a2) * radius, 0).color(r, g, b, a);
            buf.vertex(m, cx + (float) Math.cos(a2) * radius * 0.5f, cy + (float) Math.sin(a2) * radius * 0.5f, 0).color(r, g, b, a);
        }
    }

    private static float clamp(float v, float min, float max) {
        return Math.max(min, Math.min(max, v));
    }
}
