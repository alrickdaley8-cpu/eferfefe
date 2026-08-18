package com.spideysense;

import com.spideysense.client.SpideySenseKeybinds;
import com.spideysense.client.SpideySenseOverlay;
import com.spideysense.client.SpideySenseHandler;
import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.fabric.api.client.rendering.v1.HudRenderCallback;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Spidey Sense Mod
 *
 * Inspired by Miles Morales' spider-sense from the Spider-Verse films.
 *
 *  HOLD-TO-CHARGE: hold V to charge up your bioelectric spidey sense.
 *  Release to trigger. Longer hold = longer effect, longer cooldown.
 *
 *  While active:
 *   - Time freezes for everyone except you
 *   - Hostile mobs within 30 blocks glow red through walls
 *   - Comic-book pop-ups ("POW!", "WHAM!", "THWIP!", "BAM!", "INCOMING!"...) explode
 *   - A giant "SPIDER-SENSE!" title splash appears
 *   - Yellow halftone dot pattern (comic-book shading)
 *   - Bioelectric vein lightning flickering across the screen
 *   - Rotating radial speed lines
 *   - Expanding comic burst ring + brief white flash
 *   - Three-layered red/orange/yellow vignette
 *   - Subtle red/orange screen tint, screen shake, zoom pulse
 *   - Thick black comic panel border with yellow inner accent
 *   - Red flames around detected hostiles
 *   - Continuous vibration particles around the player
 *   - "INCOMING!" warning when a projectile is flying at you
 *
 * Cost: 1 hunger point (full charge) in survival. Cooldown: 30s (full charge).
 */
public class SpideySenseMod implements ClientModInitializer {
    public static final String MOD_ID = "spideysense";
    public static final Logger LOGGER = LoggerFactory.getLogger(MOD_ID);

    @Override
    public void onInitializeClient() {
        LOGGER.info("[{}] Spidey Sense awakening...", MOD_ID);

        SpideySenseKeybinds.register();
        ClientTickEvents.END_CLIENT_TICK.register(SpideySenseHandler::onEndTick);
        HudRenderCallback.EVENT.register(SpideySenseOverlay::render);

        LOGGER.info("[{}] Spidey Sense ready. Press V to activate.", MOD_ID);
    }
}
