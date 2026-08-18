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

import java.util.Random;

/**
 * Renders the entire Spider-Verse freeze-frame look while Spidey Sense is active:
 *
 *   - white flash on activation
 *   - zoom pulse (brief HUD punch-in)
 *   - subtle screen shake
 *   - subtle red/orange full-screen tint ("danger mood")
 *   - expanding comic-book burst ring
 *   - radial speed lines that rotate slowly
 *   - dramatic red/orange vignette at the edges
 *   - floating comic-book words ("POW!", "WHAM!", "BAM!", "THWIP!"...)
 *   - a giant centred "SPIDER-SENSE!" title
 *
 * No HUD bar, no cooldown indicator — the comic pop-ups ARE the UI.
 */
public final class SpideySenseOverlay {
    private SpideySenseOverlay() {}

    private static final Random RANDOM = new Random();

    public static void render(DrawContext ctx, RenderTickCounter tickCounter) {
        MinecraftClient client = MinecraftClient.getInstance();
        if (client.player == null || client.world == null) return;

        boolean active = SpideySenseHandler.isActive();
        // Comic pop-ups persist briefly after the ability ends so they get to
        // finish their animation (last ~0.4s after deactivation).
        if (active || SpideySenseComicText.hasLivePops()) {
            // Push the matrix once so the shake and zoom only affect our layer.
            ctx.getMatrices().push();

            applyShake(ctx);
            applyZoom(ctx, active);

            if (active) {
                float life = SpideySenseHandler.getActivationProgress();

                renderScreenTint(ctx, life);
                renderVignette(ctx, life);
                renderSpeedLines(ctx, life);
                renderBurstRing(ctx, life);
                renderFlash(ctx, life);
            }

            // Comic pop-ups render on top of everything else (and they keep
            // drawing for a frame or two after the ability ends).
            TextRenderer tr = client.textRenderer;
            SpideySenseComicText.render(ctx, tr);

            ctx.getMatrices().pop();
        }
    }

    // ----------------------------------------------------------------- shake + zoom

    private static void applyShake(DrawContext ctx) {
        // Subtle screen shake that decreases as the effect wears on.
        float intensity = SpideySenseHandler.getShakeIntensity();
        if (intensity <= 0f) return;
        float dx = (RANDOM.nextFloat() - 0.5f) * 6f * intensity;
        float dy = (RANDOM.nextFloat() - 0.5f) * 6f * intensity;
        ctx.getMatrices().translate(dx, dy, 0);
    }

    private static void applyZoom(DrawContext ctx, boolean active) {
        if (!active) return;
        // Zoom pulse: 1.0 → 1.06 → 1.0 over the first ~10 ticks.
        float z = SpideySenseHandler.getZoomPulse();
        if (z <= 0f) return;
        int w = ctx.getScaledWindowWidth();
        int h = ctx.getScaledWindowHeight();
        ctx.getMatrices().translate(w / 2f, h / 2f, 0);
        ctx.getMatrices().scale(z, z, 1f);
        ctx.getMatrices().translate(-w / 2f, -h / 2f, 0);
    }

    // ----------------------------------------------------------------- full-screen tint

    /** Subtle red/orange tint over the whole screen — sets the "danger mood". */
    private static void renderScreenTint(DrawContext ctx, float life) {
        int w = ctx.getScaledWindowWidth();
        int h = ctx.getScaledWindowHeight();
        float a = 0.18f * (0.5f + 0.5f * life);
        drawFullColouredQuad(ctx, w, h, 0.55f, 0.10f, 0.05f, a);
    }

    // ----------------------------------------------------------------- flash

    /** Brief white flash on activation (first ~6 ticks). */
    private static void renderFlash(DrawContext ctx, float life) {
        float t = SpideySenseHandler.getFlashProgress();
        if (t <= 0f) return;
        int w = ctx.getScaledWindowWidth();
        int h = ctx.getScaledWindowHeight();
        // Quick pop then fade.
        float a = (float) Math.pow(t, 1.5) * 0.55f;
        drawFullColouredQuad(ctx, w, h, 1f, 1f, 1f, a);
    }

    // ----------------------------------------------------------------- burst ring

    /**
     * A bright ring that expands outward from the centre of the screen for the
     * first ~15 ticks of the effect, then fades away. Classic comic-book impact.
     */
    private static void renderBurstRing(DrawContext ctx, float life) {
        float t = SpideySenseHandler.getBurstProgress();
        if (t <= 0f || t >= 1f) return;

        int w = ctx.getScaledWindowWidth();
        int h = ctx.getScaledWindowHeight();
        float cx = w / 2f;
        float cy = h / 2f;
        float maxRadius = (float) Math.hypot(w, h) / 2f;
        float radius = t * maxRadius * 0.95f;
        float thickness = 6f + t * 90f;
        float alpha = (1f - t) * 0.75f;

        int segments = 80;
        Matrix4f matrix = ctx.getMatrices().peek().getPositionMatrix();

        enablePositionColourBlend();
        Tessellator tess = Tessellator.getInstance();
        BufferBuilder buf = tess.begin(VertexFormat.DrawMode.QUADS, VertexFormats.POSITION_COLOR);

        // Draw a ring as a connected strip of trapezoids.
        for (int i = 0; i < segments; i++) {
            float a1 = (float) (i * 2 * Math.PI / segments);
            float a2 = (float) ((i + 1) * 2 * Math.PI / segments);
            float cos1 = (float) Math.cos(a1), sin1 = (float) Math.sin(a1);
            float cos2 = (float) Math.cos(a2), sin2 = (float) Math.sin(a2);

            // Bright inner edge, transparent outer edge.
            buf.vertex(matrix, cx + cos1 * radius,             cy + sin1 * radius,             0).color(1.0f, 0.95f, 0.55f, alpha);
            buf.vertex(matrix, cx + cos2 * radius,             cy + sin2 * radius,             0).color(1.0f, 0.95f, 0.55f, alpha);
            buf.vertex(matrix, cx + cos2 * (radius + thickness),cy + sin2 * (radius + thickness),0).color(1.0f, 0.70f, 0.10f, 0f);
            buf.vertex(matrix, cx + cos1 * (radius + thickness),cy + sin1 * (radius + thickness),0).color(1.0f, 0.70f, 0.10f, 0f);
        }
        BufferRenderer.drawWithGlobalProgram(buf.end());
        disableBlend();
    }

    // ----------------------------------------------------------------- speed lines

    /**
     * 32 radial "speed lines" emanating from the centre of the screen. They
     * rotate slowly and pulse, creating that freeze-frame action-panel feel.
     */
    private static void renderSpeedLines(DrawContext ctx, float life) {
        int w = ctx.getScaledWindowWidth();
        int h = ctx.getScaledWindowHeight();
        float cx = w / 2f;
        float cy = h / 2f;
        float minDim = Math.min(w, h);
        float innerR = minDim * 0.08f;
        float outerR = minDim * 0.55f;

        int lines = 32;
        float pulse = (float) (0.65 + 0.35 * Math.sin(System.currentTimeMillis() / 220.0));
        float intensity = (0.35f + 0.65f * life) * pulse;
        float rotation = (System.currentTimeMillis() / 4000f) % 360f;   // ~4s per full rotation

        Matrix4f matrix = ctx.getMatrices().peek().getPositionMatrix();
        enablePositionColourBlend();
        Tessellator tess = Tessellator.getInstance();
        BufferBuilder buf = tess.begin(VertexFormat.DrawMode.QUADS, VertexFormats.POSITION_COLOR);

        for (int i = 0; i < lines; i++) {
            float angle = (float) Math.toRadians((i * 360f / lines) + rotation);
            float perp = angle + (float) Math.PI / 2f;
            float lineWidth = 2.5f + (float) Math.sin(System.currentTimeMillis() / 200.0 + i) * 1.2f;
            float cos = (float) Math.cos(angle);
            float sin = (float) Math.sin(angle);
            float cosP = (float) Math.cos(perp);
            float sinP = (float) Math.sin(perp);

            // Quad corners. Inner end → outer end. Fade from solid → transparent
            // at the inner end so the centre stays clear.
            float x1 = cx + cos * innerR + cosP * lineWidth / 2f;
            float y1 = cy + sin * innerR + sinP * lineWidth / 2f;
            float x2 = cx + cos * innerR - cosP * lineWidth / 2f;
            float y2 = cy + sin * innerR - sinP * lineWidth / 2f;
            float x3 = cx + cos * outerR - cosP * lineWidth / 2f;
            float y3 = cy + sin * outerR - sinP * lineWidth / 2f;
            float x4 = cx + cos * outerR + cosP * lineWidth / 2f;
            float y4 = cy + sin * outerR + sinP * lineWidth / 2f;

            float a = intensity * 0.55f;
            buf.vertex(matrix, x1, y1, 0).color(1.0f, 0.95f, 0.65f, a);
            buf.vertex(matrix, x2, y2, 0).color(1.0f, 0.95f, 0.65f, a);
            buf.vertex(matrix, x3, y3, 0).color(1.0f, 0.95f, 0.65f, 0f);
            buf.vertex(matrix, x4, y4, 0).color(1.0f, 0.95f, 0.65f, 0f);
        }
        BufferRenderer.drawWithGlobalProgram(buf.end());
        disableBlend();
    }

    // ----------------------------------------------------------------- vignette

    /**
     * A bigger, more dramatic version of the Spider-Verse red/orange vignette.
     * Three layered passes give it depth: a wide outer halo, a sharper inner
     * edge, and a thin yellow inner highlight that "breathes".
     */
    private static void renderVignette(DrawContext ctx, float life) {
        int w = ctx.getScaledWindowWidth();
        int h = ctx.getScaledWindowHeight();
        float pulse = (float) (0.55 + 0.45 * Math.sin(System.currentTimeMillis() / 180.0));
        float baseAlpha = 0.75f * (0.4f + 0.6f * life) * pulse;

        enablePositionColourBlend();
        Matrix4f matrix = ctx.getMatrices().peek().getPositionMatrix();
        Tessellator tess = Tessellator.getInstance();

        // Wide soft outer halo.
        renderVignettePass(tess, matrix, w, h, baseAlpha * 0.55f, 0.22f, 0.95f, 0.18f, 0.05f);
        // Sharper red inner edge.
        renderVignettePass(tess, matrix, w, h, baseAlpha * 0.95f, 0.12f, 1.00f, 0.10f, 0.05f);
        // Thin yellow inner highlight.
        renderVignettePass(tess, matrix, w, h, baseAlpha * 0.50f, 0.06f, 1.00f, 0.85f, 0.20f);

        disableBlend();
    }

    private static void renderVignettePass(Tessellator tess, Matrix4f m, int w, int h,
                                           float alpha, float fade, float r, float g, float b) {
        int edge = (int) (Math.min(w, h) * fade);
        BufferBuilder buf = tess.begin(VertexFormat.DrawMode.QUADS, VertexFormats.POSITION_COLOR);

        addEdgeQuad(buf, m, 0, 0, w, 0, w, edge, 0, edge, alpha, r, g, b);                       // top
        addEdgeQuad(buf, m, 0, h - edge, 0, h, w, h, w, h - edge, alpha, r, g, b);               // bottom
        addEdgeQuad(buf, m, 0, 0, 0, h, edge, h, edge, 0, alpha, r, g, b);                       // left
        addEdgeQuad(buf, m, w - edge, 0, w - edge, h, w, h, w, 0, alpha, r, g, b);                // right

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
}
