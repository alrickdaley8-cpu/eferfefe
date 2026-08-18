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
 * Renders the entire Spider-Verse freeze-frame look while Spidey Sense is active
 * — and the charge-up ring while the player is holding V.
 *
 *   Layer stack (back → front):
 *     1. Subtle red/orange full-screen tint
 *     2. Yellow halftone dot grid (comic-book shading)
 *     3. Procedural bioelectric vein lightning
 *     4. 32 rotating radial speed lines
 *     5. Expanding comic burst ring (on activation)
 *     6. Three-layered red/orange/yellow vignette
 *     7. Brief white flash (on activation)
 *     8. Thick black comic panel border with yellow inner accent
 *     9. Charge-up indicator ring (when holding V to charge)
 *    10. Floating comic pop-ups ("POW!", "WHAM!", "INCOMING!"...)
 *
 *   All transforms (shake + zoom) are applied to this layer only — vanilla
 *   HUD renders normally above us.
 */
public final class SpideySenseOverlay {
    private SpideySenseOverlay() {}

    private static final Random RANDOM = new Random();

    // Cached lightning path buffer so we don't allocate every frame.
    private static final List<float[]> LIGHTNING_SEGMENTS = new ArrayList<>();
    private static long lightningLastSeed = 0;

    public static void render(DrawContext ctx, RenderTickCounter tickCounter) {
        MinecraftClient client = MinecraftClient.getInstance();
        if (client.player == null || client.world == null) return;

        boolean active = SpideySenseHandler.isActive();
        boolean charging = SpideySenseHandler.isCharging();

        // Keep drawing while active, while charging, or while a pop-up is fading out.
        if (!(active || charging || SpideySenseComicText.hasLivePops())) return;

        ctx.getMatrices().push();

        // Shake (subtle, only while active). Zoom pulse (only while active).
        applyShake(ctx, active);
        applyZoom(ctx, active);

        float life = active ? SpideySenseHandler.getActivationProgress() : 0f;

        if (active) {
            renderScreenTint(ctx, life);
            renderHalftone(ctx, life);
            renderBioElectricVeins(ctx, life);
            renderSpeedLines(ctx, life);
            renderBurstRing(ctx, life);
            renderVignette(ctx, life);
            renderFlash(ctx, life);
            renderComicPanelBorder(ctx, life);
        }

        if (charging) {
            renderChargeIndicator(ctx, SpideySenseHandler.getChargeProgress());
        }

        // Comic pop-ups render on top of every other layer.
        SpideySenseComicText.render(ctx, client.textRenderer);

        ctx.getMatrices().pop();
    }

    // ----------------------------------------------------------------- transforms

    private static void applyShake(DrawContext ctx, boolean active) {
        if (!active) return;
        float intensity = SpideySenseHandler.getShakeIntensity();
        if (intensity <= 0f) return;
        float dx = (RANDOM.nextFloat() - 0.5f) * 7f * intensity;
        float dy = (RANDOM.nextFloat() - 0.5f) * 7f * intensity;
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
        float a = 0.20f * (0.5f + 0.5f * life);
        drawFullColouredQuad(ctx, w, h, 0.55f, 0.10f, 0.05f, a);
    }

    // ----------------------------------------------------------------- 2. halftone

    /**
     * A pulsing grid of yellow dots — the classic Spider-Verse "comic-book
     * shading" effect. Each dot's size and alpha varies with a sine wave so
     * the field appears to breathe.
     */
    private static void renderHalftone(DrawContext ctx, float intensity) {
        int w = ctx.getScaledWindowWidth();
        int h = ctx.getScaledWindowHeight();
        int spacing = 26;
        long time = System.currentTimeMillis();

        Matrix4f matrix = ctx.getMatrices().peek().getPositionMatrix();
        enablePositionColourBlend();
        BufferBuilder buf = Tessellator.getInstance().begin(VertexFormat.DrawMode.QUADS, VertexFormats.POSITION_COLOR);

        for (int x = spacing / 2; x < w; x += spacing) {
            for (int y = spacing / 2; y < h; y += spacing) {
                float wave = (float) (0.5 + 0.5 * Math.sin((x + y + time / 40.0) / 26.0));
                float r = 2.2f + wave * 1.8f;
                float a = intensity * (0.10f + wave * 0.16f);
                buf.vertex(matrix, x - r, y - r, 0).color(1f, 0.85f, 0.30f, a);
                buf.vertex(matrix, x + r, y - r, 0).color(1f, 0.85f, 0.30f, a);
                buf.vertex(matrix, x + r, y + r, 0).color(1f, 0.85f, 0.30f, a);
                buf.vertex(matrix, x - r, y + r, 0).color(1f, 0.85f, 0.30f, a);
            }
        }

        BufferRenderer.drawWithGlobalProgram(buf.end());
        disableBlend();
    }

    // ----------------------------------------------------------------- 3. bioelectric veins

    /**
     * Regenerate procedural lightning paths every ~80ms and render them as
     * thick yellow quads arcing across the screen. The seed changes every
     * frame-tick so the bolts flicker like real bioelectricity.
     */
    private static void renderBioElectricVeins(DrawContext ctx, float intensity) {
        int w = ctx.getScaledWindowWidth();
        int h = ctx.getScaledWindowHeight();

        // Regenerate lightning paths when the seed changes.
        long seed = System.currentTimeMillis() / 80;
        if (seed != lightningLastSeed) {
            lightningLastSeed = seed;
            LIGHTNING_SEGMENTS.clear();
            Random rng = new Random(seed);
            int numPaths = 6;
            for (int p = 0; p < numPaths; p++) {
                // Start from a random edge.
                float x, y;
                int edge = rng.nextInt(4);
                switch (edge) {
                    case 0:  x = rng.nextFloat() * w;            y = 0;             break;
                    case 1:  x = rng.nextFloat() * w;            y = h;             break;
                    case 2:  x = 0;                              y = rng.nextFloat() * h; break;
                    default: x = w;                              y = rng.nextFloat() * h;
                }
                float startX = x, startY = y;
                int segments = 8 + rng.nextInt(10);
                for (int s = 0; s < segments; s++) {
                    float dx = (rng.nextFloat() - 0.5f) * 110f;
                    float dy = (rng.nextFloat() - 0.5f) * 110f;
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

        float thickness = 2.2f;
        float a = intensity * 0.85f;
        for (float[] seg : LIGHTNING_SEGMENTS) {
            addThickLine(buf, matrix, seg[0], seg[1], seg[2], seg[3], thickness, 1f, 0.95f, 0.30f, a);
        }
        // Second pass: brighter, thinner core for that "glowing lightning" feel.
        thickness = 0.9f;
        a = intensity * 0.95f;
        for (float[] seg : LIGHTNING_SEGMENTS) {
            addThickLine(buf, matrix, seg[0], seg[1], seg[2], seg[3], thickness, 1f, 1f, 0.85f, a);
        }

        BufferRenderer.drawWithGlobalProgram(buf.end());
        disableBlend();
    }

    private static float clamp(float v, float min, float max) {
        return Math.max(min, Math.min(max, v));
    }

    // ----------------------------------------------------------------- 4. speed lines

    private static void renderSpeedLines(DrawContext ctx, float life) {
        int w = ctx.getScaledWindowWidth();
        int h = ctx.getScaledWindowHeight();
        float cx = w / 2f, cy = h / 2f;
        float minDim = Math.min(w, h);
        float innerR = minDim * 0.08f;
        float outerR = minDim * 0.55f;

        int lines = 32;
        float pulse = (float) (0.65 + 0.35 * Math.sin(System.currentTimeMillis() / 220.0));
        float intensity = (0.35f + 0.65f * life) * pulse;
        float rotation = (System.currentTimeMillis() / 4000f) % 360f;

        Matrix4f matrix = ctx.getMatrices().peek().getPositionMatrix();
        enablePositionColourBlend();
        BufferBuilder buf = Tessellator.getInstance().begin(VertexFormat.DrawMode.QUADS, VertexFormats.POSITION_COLOR);

        for (int i = 0; i < lines; i++) {
            float angle = (float) Math.toRadians((i * 360f / lines) + rotation);
            float perp = angle + (float) Math.PI / 2f;
            float lineWidth = 2.5f + (float) Math.sin(System.currentTimeMillis() / 200.0 + i) * 1.2f;
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

            float a = intensity * 0.55f;
            buf.vertex(matrix, x1, y1, 0).color(1f, 0.95f, 0.65f, a);
            buf.vertex(matrix, x2, y2, 0).color(1f, 0.95f, 0.65f, a);
            buf.vertex(matrix, x3, y3, 0).color(1f, 0.95f, 0.65f, 0f);
            buf.vertex(matrix, x4, y4, 0).color(1f, 0.95f, 0.65f, 0f);
        }

        BufferRenderer.drawWithGlobalProgram(buf.end());
        disableBlend();
    }

    // ----------------------------------------------------------------- 5. burst ring

    private static void renderBurstRing(DrawContext ctx, float life) {
        float t = SpideySenseHandler.getBurstProgress();
        if (t <= 0f || t >= 1f) return;

        int w = ctx.getScaledWindowWidth();
        int h = ctx.getScaledWindowHeight();
        float cx = w / 2f, cy = h / 2f;
        float maxRadius = (float) Math.hypot(w, h) / 2f;
        float radius = t * maxRadius * 0.95f;
        float thickness = 6f + t * 90f;
        float alpha = (1f - t) * 0.75f;

        int segments = 80;
        Matrix4f matrix = ctx.getMatrices().peek().getPositionMatrix();
        enablePositionColourBlend();
        BufferBuilder buf = Tessellator.getInstance().begin(VertexFormat.DrawMode.QUADS, VertexFormats.POSITION_COLOR);

        for (int i = 0; i < segments; i++) {
            float a1 = (float) (i * 2 * Math.PI / segments);
            float a2 = (float) ((i + 1) * 2 * Math.PI / segments);
            float cos1 = (float) Math.cos(a1), sin1 = (float) Math.sin(a1);
            float cos2 = (float) Math.cos(a2), sin2 = (float) Math.sin(a2);
            buf.vertex(matrix, cx + cos1 * radius,                   cy + sin1 * radius,                   0).color(1f, 0.95f, 0.55f, alpha);
            buf.vertex(matrix, cx + cos2 * radius,                   cy + sin2 * radius,                   0).color(1f, 0.95f, 0.55f, alpha);
            buf.vertex(matrix, cx + cos2 * (radius + thickness),     cy + sin2 * (radius + thickness),     0).color(1f, 0.70f, 0.10f, 0f);
            buf.vertex(matrix, cx + cos1 * (radius + thickness),     cy + sin1 * (radius + thickness),     0).color(1f, 0.70f, 0.10f, 0f);
        }

        BufferRenderer.drawWithGlobalProgram(buf.end());
        disableBlend();
    }

    // ----------------------------------------------------------------- 6. vignette

    private static void renderVignette(DrawContext ctx, float life) {
        int w = ctx.getScaledWindowWidth();
        int h = ctx.getScaledWindowHeight();
        float pulse = (float) (0.55 + 0.45 * Math.sin(System.currentTimeMillis() / 180.0));
        float baseAlpha = 0.75f * (0.4f + 0.6f * life) * pulse;

        enablePositionColourBlend();
        Matrix4f matrix = ctx.getMatrices().peek().getPositionMatrix();
        Tessellator tess = Tessellator.getInstance();

        renderVignettePass(tess, matrix, w, h, baseAlpha * 0.55f, 0.22f, 0.95f, 0.18f, 0.05f);
        renderVignettePass(tess, matrix, w, h, baseAlpha * 0.95f, 0.12f, 1.00f, 0.10f, 0.05f);
        renderVignettePass(tess, matrix, w, h, baseAlpha * 0.50f, 0.06f, 1.00f, 0.85f, 0.20f);

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

    // ----------------------------------------------------------------- 7. flash

    private static void renderFlash(DrawContext ctx, float life) {
        float t = SpideySenseHandler.getFlashProgress();
        if (t <= 0f) return;
        int w = ctx.getScaledWindowWidth();
        int h = ctx.getScaledWindowHeight();
        float a = (float) Math.pow(t, 1.5) * 0.55f;
        drawFullColouredQuad(ctx, w, h, 1f, 1f, 1f, a);
    }

    // ----------------------------------------------------------------- 8. comic panel border

    /**
     * A thick black border around the screen with a thin yellow inner accent.
     * This frames the view like a comic-book panel, reinforcing the visual
     * style during the effect.
     */
    private static void renderComicPanelBorder(DrawContext ctx, float intensity) {
        if (intensity <= 0f) return;
        int w = ctx.getScaledWindowWidth();
        int h = ctx.getScaledWindowHeight();
        int t = 10;
        int aa = 0xFF000000;
        ctx.fill(0, 0, w, t, aa);
        ctx.fill(0, h - t, w, h, aa);
        ctx.fill(0, 0, t, h, aa);
        ctx.fill(w - t, 0, w, h, aa);

        // Yellow accent line just inside the border.
        int accent = 0xFFFFAA00;
        ctx.fill(t, t, w - t, t + 2, accent);
        ctx.fill(t, h - t - 2, w - t, h - t, accent);
        ctx.fill(t, t, t + 2, h - t, accent);
        ctx.fill(w - t - 2, t, w - t, h - t, accent);
    }

    // ----------------------------------------------------------------- 9. charge indicator

    /**
     * A glowing yellow ring that converges from the screen edges toward the
     * centre as the player holds V. At max charge it reaches the centre and
     * flashes brightly.
     */
    private static void renderChargeIndicator(DrawContext ctx, float charge) {
        if (charge <= 0f) return;
        int w = ctx.getScaledWindowWidth();
        int h = ctx.getScaledWindowHeight();
        float cx = w / 2f, cy = h / 2f;
        float minDim = Math.min(w, h);
        // Ring radius shrinks from outer screen inward as charge grows.
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

        // When fully charged, pulse a thin white core ring.
        if (maxed) {
            float corePulse = 0.5f + 0.5f * (float) Math.sin(System.currentTimeMillis() / 60.0);
            ctx.fill(0, 0, w, 4, 0xFFFFFFFF);                                  // top
            ctx.fill(0, h - 4, w, h, 0xFFFFFFFF);                              // bottom
            ctx.fill(0, 0, 4, h, 0xFFFFFFFF);                                  // left
            ctx.fill(w - 4, 0, w, h, 0xFFFFFFFF);                              // right
        }
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
}
