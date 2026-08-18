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
 * Spidey Sense Mod — MAXED OUT
 *
 * Inspired by Miles Morales' spider-sense from the Spider-Verse films.
 *
 *  HOLD-TO-CHARGE: hold V to charge up your bioelectric spidey sense.
 *  Release to trigger. Longer hold = longer effect, longer cooldown.
 *
 *  While active, 17 visual layers stack back-to-front:
 *   1. Red/orange danger tint
 *   2. Yellow halftone dot field (comic-book shading)
 *   3. Bioelectric vein lightning
 *   4. Vertical sky lightning bolts
 *   5. 48 rotating radial speed lines
 *   6. Double expanding comic burst rings
 *   7. Four-layered red/orange/yellow vignette
 *   8. White activation flash
 *   9. Glitch-tear screen-tear bars
 *  10. Two horizontal multi-panel split lines
 *  11. Thick black comic panel border + yellow accent
 *  12. Procedural spider-web corner overlay
 *  13. Glowing red spider eyes at top of screen
 *  14. Spider-logo watermark in corner
 *  15. Random brightness flickers
 *  16. Floating comic pop-ups with colour cycling
 *  17. Custom yellow spider-web crosshair
 *
 *  Plus: time freeze, glowing hostiles, "INCOMING!" projectile warning,
 *  web-burst particles, hostile flame particles, bioelectric vibration
 *  particles, charge-up sound tick, screen shake, zoom pulse.
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
