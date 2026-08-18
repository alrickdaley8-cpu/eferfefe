package com.spideysense.client;

import net.minecraft.client.MinecraftClient;
import net.minecraft.client.network.ClientPlayerEntity;
import net.minecraft.client.world.ClientWorld;
import net.minecraft.entity.Entity;
import net.minecraft.entity.LivingEntity;
import net.minecraft.entity.mob.HostileEntity;
import net.minecraft.entity.player.PlayerEntity;
import net.minecraft.potion.StatusEffectInstance;
import net.minecraft.potion.StatusEffects;
import net.minecraft.sound.SoundEvents;
import net.minecraft.util.math.Box;

/**
 * Core game logic for the Spidey Sense ability.
 *
 * - Listens for key presses and triggers the ability.
 * - Manages the active duration, cooldown, and time-freeze effect.
 * - Applies the glowing effect to nearby hostile mobs so they shine
 *   through walls (just like in the films).
 */
public final class SpideySenseHandler {
    private SpideySenseHandler() {}

    // ----- TUNABLE CONSTANTS ----------------------------------------------------
    /** How long the effect lasts, in ticks (20 ticks = 1 second). */
    public static final int DURATION_TICKS = 60;          // 3 seconds
    /** Cooldown after the effect ends, in ticks. */
    public static final int COOLDOWN_TICKS = 20 * 30;     // 30 seconds
    /** Radius (in blocks) within which hostiles are revealed. */
    public static final int DETECT_RADIUS = 30;
    /** Hunger cost in survival/adventure modes. */
    public static final int HUNGER_COST = 1;             // = 1/2 drumstick
    /** How long the glow effect stays on revealed mobs, in ticks. */
    public static final int GLOW_TICKS = DURATION_TICKS + 40; // a hair longer than duration
    // ---------------------------------------------------------------------------

    private static int activeTicksRemaining = 0;
    private static int cooldownTicksRemaining = 0;
    private static boolean timeFrozenByUs = false;

    /**
     * Called every client tick from {@link ClientTickEvents.END_CLIENT_TICK}.
     * Handles key input, the active timer, the cooldown timer, and the
     * "freeze time" toggle.
     */
    public static void onEndTick(MinecraftClient client) {
        ClientPlayerEntity player = client.player;
        ClientWorld world = client.world;
        if (player == null || world == null) return;

        // Cooldown always counts down, even on the title screen
        if (cooldownTicksRemaining > 0) cooldownTicksRemaining--;

        // ---- Press detection (consume one press per tick) ---------------------
        while (SpideySenseKeybinds.activate.wasPressed()) {
            if (activeTicksRemaining <= 0 && cooldownTicksRemaining <= 0) {
                activate(player, world);
            }
        }

        // ---- Active timer -----------------------------------------------------
        if (activeTicksRemaining > 0) {
            // First tick of activation -> freeze the world
            if (!timeFrozenByUs) {
                world.getTickManager().setFrozen(true);
                timeFrozenByUs = true;
            }
            // Keep revealing hostiles every tick
            applyGlowToThreats(world, player);

            activeTicksRemaining--;

            // Last tick of activation -> unfreeze and start cooldown
            if (activeTicksRemaining == 0) {
                if (timeFrozenByUs) {
                    world.getTickManager().setFrozen(false);
                    timeFrozenByUs = false;
                }
                cooldownTicksRemaining = COOLDOWN_TICKS;
                // Always unfreeze on the final tick, even if something went wrong
            }
        }
    }

    private static void activate(ClientPlayerEntity player, ClientWorld world) {
        activeTicksRemaining = DURATION_TICKS;

        // Hunger cost only in survival / adventure
        if (!player.isCreative() && !player.isSpectator()) {
            player.getHungerManager().add(-HUNGER_COST * 2, player.getSaturationLevel() > 0
                    ? player.getSaturationLevel() : 0f);
        }

        // Cinematic sound layer: whoosh + low bass pulse
        player.playSound(SoundEvents.ENTITY_ENDER_DRAGON_GROWL, 0.18f, 1.7f);
        player.playSound(SoundEvents.BLOCK_BEACON_ACTIVATE,      0.35f, 1.9f);
        player.playSound(SoundEvents.ENTITY_WARDEN_HEARTBEAT,    0.22f, 1.4f);

        // Immediately reveal hostiles so the player sees them on frame 1
        applyGlowToThreats(world, player);
    }

    /**
     * Apply the {@link StatusEffects#GLOWING} effect to every hostile mob
     * within {@link #DETECT_RADIUS} blocks. Glowing makes their silhouette
     * visible through walls, exactly like the Spider-Verse effect.
     */
    private static void applyGlowToThreats(ClientWorld world, PlayerEntity player) {
        Box searchBox = player.getBoundingBox().expand(DETECT_RADIUS);
        for (Entity entity : world.getOtherEntities(
                player, searchBox,
                e -> e instanceof HostileEntity && e.isAlive())) {
            if (entity instanceof LivingEntity living) {
                living.addStatusEffect(new StatusEffectInstance(
                        StatusEffects.GLOWING,
                        GLOW_TICKS,
                        0,
                        false, false, true));
            }
        }
    }

    // ---- Read-only state for the renderer ------------------------------------
    public static boolean isActive() {
        return activeTicksRemaining > 0;
    }

    public static float getActivationProgress() {
        // 1.0 at the start of the effect, 0.0 at the end
        if (activeTicksRemaining <= 0) return 0f;
        return (float) activeTicksRemaining / DURATION_TICKS;
    }

    public static int getCooldownTicksRemaining() {
        return cooldownTicksRemaining;
    }

    public static int getCooldownTicksTotal() {
        return COOLDOWN_TICKS;
    }
}
