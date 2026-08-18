package com.spideysense.client;

import net.minecraft.client.MinecraftClient;
import net.minecraft.client.network.ClientPlayerEntity;
import net.minecraft.client.world.ClientWorld;
import net.minecraft.entity.Entity;
import net.minecraft.entity.LivingEntity;
import net.minecraft.entity.mob.HostileEntity;
import net.minecraft.entity.player.PlayerEntity;
import net.minecraft.particle.ParticleTypes;
import net.minecraft.potion.StatusEffectInstance;
import net.minecraft.potion.StatusEffects;
import net.minecraft.sound.SoundEvents;
import net.minecraft.util.math.Box;

/**
 * Core game logic for the Spidey Sense ability.
 *
 * Owns the activation timer, the cooldown timer, the time-freeze state, and
 * the per-effect cosmetic state (shake, zoom, flash, burst). Spawns the
 * comic-book pop-ups and ambient particles.
 */
public final class SpideySenseHandler {
    private SpideySenseHandler() {}

    // ----- TUNABLE CONSTANTS ----------------------------------------------------
    public static final int DURATION_TICKS = 60;          // 3 seconds of freeze
    public static final int COOLDOWN_TICKS = 20 * 30;     // 30 seconds cooldown
    public static final int DETECT_RADIUS = 30;
    public static final int HUNGER_COST = 1;             // = 1/2 drumstick
    public static final int GLOW_TICKS = DURATION_TICKS + 40;
    public static final int FLASH_TICKS = 6;
    public static final int ZOOM_TICKS = 12;
    public static final int SHAKE_TICKS = DURATION_TICKS;
    public static final int BURST_TICKS = 18;
    public static final int RANDOM_POP_INTERVAL = 8;     // ticks between random POW! pop-ups
    public static final int HOSTILE_PARTICLE_INTERVAL = 4;
    // ---------------------------------------------------------------------------

    // ----- effect lifecycle ----------------------------------------------------
    private static int activeTicksRemaining = 0;
    private static int cooldownTicksRemaining = 0;
    private static boolean timeFrozenByUs = false;

    // ----- cosmetic state ------------------------------------------------------
    private static int flashTicksRemaining = 0;
    private static int zoomTicksRemaining = 0;
    private static int shakeTicksRemaining = 0;
    private static int burstTicksRemaining = 0;
    private static int ticksSinceLastRandomPop = 0;
    private static int ticksSinceLastHostileParticle = 0;

    public static void onEndTick(MinecraftClient client) {
        ClientPlayerEntity player = client.player;
        ClientWorld world = client.world;
        if (player == null || world == null) return;

        // Cooldown always counts down.
        if (cooldownTicksRemaining > 0) cooldownTicksRemaining--;

        // ---- Press detection --------------------------------------------------
        while (SpideySenseKeybinds.activate.wasPressed()) {
            if (activeTicksRemaining <= 0 && cooldownTicksRemaining <= 0) {
                activate(player, world);
            }
        }

        // ---- Active timer ----------------------------------------------------
        if (activeTicksRemaining > 0) {
            if (!timeFrozenByUs) {
                world.getTickManager().setFrozen(true);
                timeFrozenByUs = true;
            }
            applyGlowToThreats(world, player);

            activeTicksRemaining--;

            // While the effect is alive: spawn comic pop-ups and ambient particles.
            spawnAmbientEffects(player, world, activeTicksRemaining);

            if (activeTicksRemaining == 0) {
                if (timeFrozenByUs) {
                    world.getTickManager().setFrozen(false);
                    timeFrozenByUs = false;
                }
                cooldownTicksRemaining = COOLDOWN_TICKS;
                // Finisher pop-up when the freeze ends.
                SpideySenseComicText.spawnFinisher(client);
            }
        }

        // Cosmetic timers tick regardless.
        if (flashTicksRemaining > 0) flashTicksRemaining--;
        if (zoomTicksRemaining > 0) zoomTicksRemaining--;
        if (shakeTicksRemaining > 0) shakeTicksRemaining--;
        if (burstTicksRemaining > 0) burstTicksRemaining--;

        // Age out comic pop-ups every tick.
        SpideySenseComicText.tick();
    }

    private static void activate(ClientPlayerEntity player, ClientWorld world) {
        activeTicksRemaining = DURATION_TICKS;

        // Hunger cost in survival / adventure.
        if (!player.isCreative() && !player.isSpectator()) {
            player.getHungerManager().add(-HUNGER_COST * 2, 0f);
        }

        // Cinematic sound layer: whoosh + bass pulse + heartbeat.
        player.playSound(SoundEvents.ENTITY_ENDER_DRAGON_GROWL, 0.18f, 1.7f);
        player.playSound(SoundEvents.BLOCK_BEACON_ACTIVATE,    0.35f, 1.9f);
        player.playSound(SoundEvents.ENTITY_WARDEN_HEARTBEAT,  0.22f, 1.4f);

        // Cosmetic timers.
        flashTicksRemaining = FLASH_TICKS;
        zoomTicksRemaining  = ZOOM_TICKS;
        shakeTicksRemaining = SHAKE_TICKS;
        burstTicksRemaining = BURST_TICKS;

        // Comic title (e.g. "SPIDER-SENSE!") + immediate reveal of hostiles.
        SpideySenseComicText.spawnTitle(MinecraftClient.getInstance());
        spawnWebBurst(player, world);
        applyGlowToThreats(world, player);
    }

    // ----------------------------------------------------------------- ambient

    /** Spawn random comic pop-ups, particles around hostiles, etc. */
    private static void spawnAmbientEffects(ClientPlayerEntity player, ClientWorld world, int ticksLeft) {
        MinecraftClient client = MinecraftClient.getInstance();

        // Random "POW!" / "WHAM!" pop-up every RANDOM_POP_INTERVAL ticks.
        ticksSinceLastRandomPop++;
        if (ticksSinceLastRandomPop >= RANDOM_POP_INTERVAL) {
            SpideySenseComicText.spawnRandom(client);
            ticksSinceLastRandomPop = 0;
        }

        // Red menacing flame particles around every detected hostile.
        ticksSinceLastHostileParticle++;
        if (ticksSinceLastHostileParticle >= HOSTILE_PARTICLE_INTERVAL) {
            spawnHostileParticles(world, player);
            ticksSinceLastHostileParticle = 0;
        }
    }

    // ----------------------------------------------------------------- particles

    /** A burst of web/electric sparks around the player on activation. */
    private static void spawnWebBurst(ClientPlayerEntity player, ClientWorld world) {
        int count = 40;
        for (int i = 0; i < count; i++) {
            double angle = Math.random() * Math.PI * 2;
            double dist  = Math.random() * 2.5;
            double px = player.getX() + Math.cos(angle) * dist;
            double py = player.getY() + 0.5 + Math.random() * 2.0;
            double pz = player.getZ() + Math.sin(angle) * dist;
            double vx = Math.cos(angle) * 0.1;
            double vy = 0.05 + Math.random() * 0.1;
            double vz = Math.sin(angle) * 0.1;
            world.addParticle(ParticleTypes.CRIT, px, py, pz, vx, vy, vz);
        }
    }

    /** Red flame particles around every hostile within detection range. */
    private static void spawnHostileParticles(ClientWorld world, PlayerEntity player) {
        Box box = player.getBoundingBox().expand(DETECT_RADIUS);
        for (Entity e : world.getOtherEntities(player, box,
                ent -> ent instanceof HostileEntity && ent.isAlive())) {
            for (int i = 0; i < 4; i++) {
                double angle = Math.random() * Math.PI * 2;
                double dist  = Math.random() * 1.5;
                double px = e.getX() + Math.cos(angle) * dist;
                double py = e.getY() + 0.2 + Math.random() * e.getHeight();
                double pz = e.getZ() + Math.sin(angle) * dist;
                world.addParticle(ParticleTypes.FLAME, px, py, pz, 0, 0.04, 0);
            }
        }
    }

    /** Apply {@link StatusEffects#GLOWING} to every hostile within range. */
    private static void applyGlowToThreats(ClientWorld world, PlayerEntity player) {
        Box box = player.getBoundingBox().expand(DETECT_RADIUS);
        for (Entity entity : world.getOtherEntities(player, box,
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

    // ----------------------------------------------------------------- read-only state

    public static boolean isActive() {
        return activeTicksRemaining > 0;
    }

    public static float getActivationProgress() {
        if (activeTicksRemaining <= 0) return 0f;
        return (float) activeTicksRemaining / DURATION_TICKS;
    }

    public static int getCooldownTicksRemaining() {
        return cooldownTicksRemaining;
    }

    public static int getCooldownTicksTotal() {
        return COOLDOWN_TICKS;
    }

    public static float getFlashProgress() {
        return flashTicksRemaining <= 0 ? 0f : (float) flashTicksRemaining / FLASH_TICKS;
    }

    /**
     * Zoom pulse: peaks at ~1.06 over the first ZOOM_TICKS ticks and falls
     * back to 1.0. We model it with a sine arch so the HUD punches in then
     * snaps back smoothly.
     */
    public static float getZoomPulse() {
        if (zoomTicksRemaining <= 0) return 0f;
        float t = 1f - (float) zoomTicksRemaining / ZOOM_TICKS;   // 0 → 1
        return 1f + (float) Math.sin(t * Math.PI) * 0.06f;
    }

    public static float getShakeIntensity() {
        if (shakeTicksRemaining <= 0) return 0f;
        return (float) shakeTicksRemaining / SHAKE_TICKS;
    }

    public static float getBurstProgress() {
        if (burstTicksRemaining <= 0) return 0f;
        return 1f - (float) burstTicksRemaining / BURST_TICKS;
    }
}
