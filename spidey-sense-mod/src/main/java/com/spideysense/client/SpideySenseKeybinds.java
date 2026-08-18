package com.spideysense.client;

import net.fabricmc.fabric.api.client.keybinding.v1.KeyBindingHelper;
import net.minecraft.client.option.KeyBinding;
import net.minecraft.client.util.InputUtil;
import org.lwjgl.glfw.GLFW;

/**
 * Keybinding registration for the Spidey Sense ability.
 * Default key: V. Changeable in Minecraft's Controls menu.
 */
public final class SpideySenseKeybinds {
    private SpideySenseKeybinds() {}

    public static final String CATEGORY = "category.spideysense";
    public static final String ACTIVATE_KEY = "key.spideysense.activate";

    public static KeyBinding activate;

    public static void register() {
        activate = KeyBindingHelper.registerKeyBinding(new KeyBinding(
                ACTIVATE_KEY,
                InputUtil.Type.KEYSYM,
                GLFW.GLFW_KEY_V,
                CATEGORY
        ));
    }
}
