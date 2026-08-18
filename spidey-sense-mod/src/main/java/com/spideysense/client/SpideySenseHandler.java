package com.spideysense.client;

import net.minecraft.client.MinecraftClient;
import net.minecraft.client.network.ClientPlayerEntity;
import net.minecraft.client.world.ClientWorld;
import net.minecraft.entity.Entity;
import net.minecraft.entity.LivingEntity;
import net.minecraft.entity.mob.HostileEntity;
import net.minecraft.entity.player.PlayerEntity;
import net.minecraft.entity.projectile.ProjectileEntity;
import net.minecraft.particle.ParticleTypes;
import net.minecraft.potion.StatusEffectInstance;
import net.minecraft.potion.StatusEffects;
import net.minecraft.sound.SoundEvents;
import net.minecraft.util.math.Box;
import net.minecraft.util.math.Vec3d;

/**
 * Core game logic for the Spidey Sense ability.
 *
 * Now with:
 *   - HOLD-TO-CHARGE mechanic: hold V to build up the "venom blast",
 *     release to trigger. Longer hold = longer duration, longer cooldown,
 *     louder effects.
 *   - Incoming-projectile detection during the effect — fires an
 *     "INCOMING!" comic pop-up when something is flying at the player.
 *   - Continuous bioelectric vibration particles around the player while
 *     charging AND while the effect is active.
 *   - Charge-up ticking sound that accelerates as you near max charge.
 */
public final class SpideySenseHandler {
    private SpideySenseHandler() {}

    // ----- TUNABLE CONSTANTS ----------------------------------------------------
    public static final int DURATION_TICKS = 60;          // 3 seconds of freeze (full charge)
    public static final int COOLDOWN_TICKS = 20 * 30;     // 30 seconds cooldown (full charge)
    public static final int DETECT_RADIUS = 30;
    public static final int HUNGER_COST = 1;
    public static final int GLOW_TICKS = DURATION_TICKS + 40;

    // Charge mechanic.
    public static final int CHARGE_DURATION = 30;        // 1.5s to full charge
    public static final int QUICK_DURATION = 20;         // 1s quick-effect duration
    public static final int QUICK_COOLDOWN = 20 * 5;     // 5s quick-effect cooldown
    public static final int MAX_HOLD = CHARGE_DURATION * 2;  // auto-release after 3s

    // Cosmetic timers (in ticks).
    public static final int FLASH_TICKS = 6;
    public static final int ZOOM_TICKS = 12;
    public static final int SHAKE_TICKS = DURATION_TICKS;
    public static final int BURST_TICKS = 18;

    // Periodic checks.
    public static final int RANDOM_POP_INTERVAL = 8;
    public static final int HOSTILE_PARTICLE_INTERVAL = 4;
    public static final int INCOMING_CHECK_INTERVAL = 5;
    public static final int VIBRATION_INTERVAL = 2;
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
    private static int ticksSinceIncomingCheck = 0;
    private static int ticksSinceLastVibration = 0;

    // ----- charge state --------------------------------------------------------
    private static boolean isCharging = false;
    private static int chargeTicks = 0;

    public static void onEndTick(MinecraftClient client) {
        ClientPlayerEntity player = client.player;
        ClientWorld world = client.world;
        if (player == null || world == null) return;

        // Cooldown always counts down.
        if (cooldownTicksRemaining > 0) cooldownTicksRemaining--;

        var keybind = SpideySenseKeybinds.activate;

        // ---- Charging state machine ------------------------------------------
        // Start charging on the press tick if we're ready.
        if (keybind.wasPressed()
                && activeTicksRemaining <= 0
                && cooldownTicksRemaining <= 0
                && !isCharging) {
            isCharging = true;
            chargeTicks = 0;
            ticksSinceLastVibration = 0;
            player.playSound(SoundEvents.BLOCK_BEACON_AMBIENT, 0.25f, 1.8f);
        }

        // While charging: tick up, play buildup, spawn vibration particles.
        if (isCharging) {
            chargeTicks++;

            // Charge tick — interval shortens as we approach max.
            int interval = Math.max(2, 8 - chargeTicks / 5);
            if (chargeTicks % interval == 0) {
                player.playSound(SoundEvents.UI_BUTTON_CLICK, 0.3f, 1.5f + chargeTicks * 0.05f);
            }

            // Vibration particles around the player while charging.
            ticksSinceLastVibration++;
            if (ticksSinceLastVibration >= VIBRATION_INTERVAL) {
                spawnVibrationParticles(world, player);
                ticksSinceLastVibration = 0;
            }

            // Auto-release on max hold, or release as soon as the player lets go.
            if (chargeTicks >= MAX_HOLD || !keybind.isPressed()) {
                triggerEffect(player, world, client);
            }
        }

        // ---- Active effect --------------------------------------------------
        if (activeTicksRemaining > 0) {
            if (!timeFrozenByUs) {
                world.getTickManager().setFrozen(true);
                timeFrozenByUs = true;
            }
            applyGlowToThreats(world, player);

            // Continuous vibration particles while the effect is up.
            ticksSinceLastVibration++;
            if (ticksSinceLastVibration >= VIBRATION_INTERVAL) {
                spawnVibrationParticles(world, player);
                ticksSinceLastVibration = 0;
            }

            // Periodic checks.
            ticksSinceLastRandomPop++;
            if (ticksSinceLastRandomPop >= RANDOM_POP_INTERVAL) {
                SpideySenseComicText.spawnRandom(client);
                ticksSinceLastRandomPop = 0;
            }
            ticksSinceLastHostileParticle++;
            if (ticksSinceLastHostileParticle >= HOSTILE_PARTICLE_INTERVAL) {
                spawnHostileParticles(world, player);
                ticksSinceLastHostileParticle = 0;
            }
            ticksSinceIncomingCheck++;
            if (ticksSinceIncomingCheck >= INCOMING_CHECK_INTERVAL) {
                checkIncomingProjectiles(world, player);
                ticksSinceIncomingCheck = 0;
            }

            activeTicksRemaining--;

            if (activeTicksRemaining == 0) {
                if (timeFrozenByUs) {
                    world.getTickManager().setFrozen(false);
                    timeFrozenByUs = false;
                }
                // Cooldown was already set by triggerEffect() at release time.
                SpideySenseComicText.spawnFinisher(client);
            }
        }

        // Cosmetic timers.
        if (flashTicksRemaining > 0) flashTicksRemaining--;
        if (zoomTicksRemaining > 0) zoomTicksRemaining--;
        if (shakeTicksRemaining > 0) shakeTicksRemaining--;
        if (burstTicksRemaining > 0) burstTicksRemaining--;

        // Age comic pop-ups.
        SpideySenseComicText.tick();
    }

    // ----------------------------------------------------------------- effect trigger

    /**
     * Triggered when the player releases V (or holds to max).
     * Effect duration, cooldown, sounds, and cosmetics scale with how long they held.
     */
    private static void triggerEffect(ClientPlayerEntity player, ClientWorld world, MinecraftClient client) {
        isCharging = false;
        float charge = Math.min(1f, (float) chargeTicks / CHARGE_DURATION);
        chargeTicks = 0;

        // Require a tiny minimum hold so accidental taps do nothing.
        if (charge < 0.10f) {
            player.playSound(SoundEvents.BLOCK_DISPENSER_FAIL, 0.3f, 1.6f);
            return;
        }

        // Duration + cooldown scale with charge.
        activeTicksRemaining = (int) (QUICK_DURATION + charge * (DURATION_TICKS - QUICK_DURATION));
        cooldownTicksRemaining = (int) (QUICK_COOLDOWN + charge * (COOLDOWN_TICKS - QUICK_COOLDOWN));

        // Hunger cost scales with charge (free for very brief pulses).
        if (charge > 0.30f && !player.isCreative() && !player.isSpectator()) {
            player.getHungerManager().add(-HUNGER_COST * 2, 0f);
        }

        // Sounds — louder and lower-pitched with more charge.
        float vol = 0.5f + 0.5f * charge;
        player.playSound(SoundEvents.ENTITY_ENDER_DRAGON_GROWL, 0.18f * vol, 1.7f);
        player.playSound(SoundEvents.BLOCK_BEACON_ACTIVATE,    0.35f * vol, 1.9f);
        player.playSound(SoundEvents.ENTITY_WARDEN_HEARTBEAT,  0.22f * vol, 1.4f);

        // Cosmetic timers.
        flashTicksRemaining = FLASH_TICKS;
        zoomTicksRemaining  = ZOOM_TICKS;
        shakeTicksRemaining = SHAKE_TICKS;
        burstTicksRemaining = BURST_TICKS;

        // Comic title + reveal hostiles + initial particle burst.
        SpideySenseComicText.spawnTitle(client);
        spawnWebBurst(player, world);
        applyGlowToThreats(world, player);
    }

    // ----------------------------------------------------------------- particles

    /** A burst of web/electric sparks around the player on activation. */
    private static void spawnWebBurst(ClientPlayerEntity player, ClientWorld world) {
        for (int i = 0; i < 40; i++) {
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

    /** Continuous bioelectric vibration particles around the player. */
    private static void spawnVibrationParticles(ClientWorld world, PlayerEntity player) {
        for (int i = 0; i < 2; i++) {
            double angle = Math.random() * Math.PI * 2;
            double dist  = 1.4 + Math.random() * 0.8;
            double px = player.getX() + Math.cos(angle) * dist;
            double py = player.getY() + 0.2 + Math.random() * 2.0;
            double pz = player.getZ() + Math.sin(angle) * dist;
            world.addParticle(ParticleTypes.CRIT, px, py, pz, 0, 0.05, 0);
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
                        GLOW_TICKS, 0,
                        false, false, true));
            }
        }
    }

    /** Detect projectiles flying toward the player and warn them. */
    private static void checkIncomingProjectiles(ClientWorld world, PlayerEntity player) {
        Box box = player.getBoundingBox().expand(40);
        Vec3d playerCenter = player.getPos().add(0, 1, 0);
        for (Entity e : world.getOtherEntities(player, box,
                ent -> ent instanceof ProjectileEntity && ent.isAlive())) {
            Vec3d vel = e.getVelocity();
            if (vel.length() < 0.15) continue;
            Vec3d toPlayer = playerCenter.subtract(e.getPos());
            double dist = toPlayer.length();
            if (dist > 30) continue;
            // Is the projectile moving toward the player?
            if (vel.dotProduct(toPlayer.normalize()) > 0.4) {
                SpideySenseComicText.spawnIncoming();
                break;   // only warn once per check
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

    public static boolean isCharging() {
        return isCharging;
    }

    public static float getChargeProgress() {
        if (!isCharging) return 0f;
        return Math.min(1f, (float) chargeTicks / CHARGE_DURATION);
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

    public static float getZoomPulse() {
        if (zoomTicksRemaining <= 0) return 0f;
        float t = 1f - (float) zoomTicksRemaining / ZOOM_TICKS;
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
