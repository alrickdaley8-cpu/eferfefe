package com.spideysense.client;

import com.mojang.blaze3d.systems.RenderSystem;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.gui.DrawContext;
import net.minecraft.client.render.*;
import org.joml.Matrix4f;

/**
 * Draws the signature Miles-Morales red/orange vignette around the screen
 * edges while Spidey Sense is active, plus a status line and a cooldown bar.
 */
public final class SpideySenseOverlay {
    private SpideySenseOverlay() {}

    public static void render(DrawContext ctx, RenderTickCounter tickCounter) {
        MinecraftClient client = MinecraftClient.getInstance();
        if (client.player == null || client.world == null) return;

        boolean active = SpideySenseHandler.isActive();
        int cooldown = SpideySenseHandler.getCooldownTicksRemaining();

        // Cooldown bar shows even when not active, so the player can see it counting down.
        if (active || cooldown > 0) {
            renderCooldownBar(ctx, client, active, cooldown);
        }
        if (!active) return;

        renderVignette(ctx, client);
        renderStatusText(ctx, client);
    }

    /**
     * Render a red/orange vignette around the edges of the screen. The intensity
     * is strongest at the corners and weakest in the centre, and pulses slowly
     * to mimic a heartbeat.
     */
    private static void renderVignette(DrawContext ctx, MinecraftClient client) {
        int w = ctx.getScaledWindowWidth();
        int h = ctx.getScaledWindowHeight();

        // Soft pulsing brightness — like a heartbeat
        float pulse = (float)(0.65 + 0.35 * Math.sin(System.currentTimeMillis() / 220.0));

        // Alpha ramps with how much time is left in the effect
        float life = SpideySenseHandler.getActivationProgress();
        float baseAlpha = 0.55f * (0.4f + 0.6f * life) * pulse;

        // Render two passes — a wide soft outer halo and a sharper inner edge —
        // each as four thin edge-rectangles whose alpha fades toward the centre.
        // Because the rectangles overlap near the corners, we get a natural
        // "darker at the corners" gradient in a single draw call per pass.
        RenderSystem.enableBlend();
        RenderSystem.defaultBlendFunc();
        RenderSystem.setShaderColor(1f, 1f, 1f, 1f);
        RenderSystem.setShader(GameRenderer::getPositionColorProgram);

        Matrix4f matrix = ctx.getMatrices().peek().getPositionMatrix();
        Tessellator tessellator = Tessellator.getInstance();

        // Outer halo — wider, softer
        renderVignettePass(tessellator, matrix, w, h, baseAlpha * 0.55f, 0.20f, 0.95f, 0.20f, 0.05f);
        // Inner edge — narrower, brighter, hugging the corners
        renderVignettePass(tessellator, matrix, w, h, baseAlpha * 0.85f, 0.10f, 1.00f, 0.30f, 0.05f);

        RenderSystem.disableBlend();
    }

    /**
     * Renders one vignette pass. We draw four thin rectangles along each edge
     * of the screen, each fading out toward the centre.
     *
     * @param fade  how far the gradient extends inward, as a fraction of the shorter dimension
     * @param r,g,b the vignette tint (red/orange)
     */
    private static void renderVignettePass(Tessellator tessellator, Matrix4f matrix,
                                           int w, int h, float alpha, float fade,
                                           float r, float g, float b) {
        int edge = (int)(Math.min(w, h) * fade);

        BufferBuilder buf = tessellator.begin(VertexFormat.DrawMode.QUADS, VertexFormats.POSITION_COLOR);

        // top edge    — y goes from 0 (edge) to edge (centre)
        addEdgeQuad(buf, matrix, 0, 0, w, 0, w, edge, 0, edge, alpha, r, g, b);
        // bottom edge — y goes from h-edge (centre) to h (edge)
        addEdgeQuad(buf, matrix, 0, h - edge, 0, h, w, h, w, h - edge, alpha, r, g, b);
        // left edge
        addEdgeQuad(buf, matrix, 0, 0, 0, h, edge, h, edge, 0, alpha, r, g, b);
        // right edge
        addEdgeQuad(buf, matrix, w - edge, 0, w - edge, h, w, h, w, 0, alpha, r, g, b);

        BufferRenderer.drawWithGlobalProgram(buf.end());
    }

    /** Adds one edge quad with alpha fading toward the centre. */
    private static void addEdgeQuad(BufferBuilder buf, Matrix4f m,
                                    float x1, float y1, float x2, float y2,
                                    float x3, float y3, float x4, float y4,
                                    float alpha, float r, float g, float b) {
        // (x1,y1) and (x2,y2) are the screen-edge corners → full alpha
        // (x3,y3) and (x4,y4) are the inner corners        → zero alpha
        buf.vertex(m, x1, y1, 0).color(r, g, b, alpha);
        buf.vertex(m, x2, y2, 0).color(r, g, b, alpha);
        buf.vertex(m, x3, y3, 0).color(r, g, b, 0f);
        buf.vertex(m, x4, y4, 0).color(r, g, b, 0f);
    }

    /** Render the "SPIDEY SENSE" status line at the bottom of the HUD. */
    private static void renderStatusText(DrawContext ctx, MinecraftClient client) {
        var tr = client.textRenderer;
        String text = "§c§lSPIDEY SENSE";
        int x = (ctx.getScaledWindowWidth() - tr.getWidth(text)) / 2;
        int y = ctx.getScaledWindowHeight() - 40;
        ctx.drawText(tr, text, x, y, 0xFFFFFF, true);
    }

    /**
     * Render a small red cooldown bar at the bottom-centre of the screen.
     * Shows the remaining cooldown even when the ability is not active.
     */
    private static void renderCooldownBar(DrawContext ctx, MinecraftClient client,
                                          boolean active, int cooldownTicks) {
        int w = ctx.getScaledWindowWidth();
        int h = ctx.getScaledWindowHeight();
        int barW = 80;
        int barH = 4;
        int cx = (w - barW) / 2;
        int cy = h - 25;

        float progress;
        if (active) {
            progress = 1f;
        } else {
            progress = 1f - (float) cooldownTicks / SpideySenseHandler.getCooldownTicksTotal();
        }

        // Background
        ctx.fill(cx - 1, cy - 1, cx + barW + 1, cy + barH + 1, 0xFF000000);
        // Filled portion (orange when active, red while recharging)
        int fillColor = active ? 0xFFFFAA00 : 0xFFCC2222;
        ctx.fill(cx, cy, cx + (int)(barW * progress), cy + barH, fillColor);

        // Small label above the bar
        var tr = client.textRenderer;
        String label = active ? "ACTIVE" : "COOLDOWN";
        int labelColor = active ? 0xFFFFAA00 : 0xFFAA2222;
        int labelX = (w - tr.getWidth(label)) / 2;
        ctx.drawText(tr, label, labelX, cy - 10, labelColor, true);
    }
}
