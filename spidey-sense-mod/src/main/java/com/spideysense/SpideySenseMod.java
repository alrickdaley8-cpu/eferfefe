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
 * Press V (default) to activate. While active:
 *   - Time freezes for everyone except you (3 seconds)
 *   - Hostile mobs within 30 blocks get a red glow visible through walls
 *   - Screen gets the signature red/orange vignette
 *   - Heartbeat + whoosh sound plays
 *
 * Cost: 1 hunger point in survival. Cooldown: 30 seconds.
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
