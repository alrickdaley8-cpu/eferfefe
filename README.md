# Greedy Growers 🌳⚡

A web tribute to the Roblox game **Greedy Growers** — buy seeds at the river, plant
them in your plot, watch the trees grow… and harvest **before the lightning** turns
your fortune to charcoal.

## How to play

1. **Buy a seed at the river** — open the Shop (or click the river stall) and buy a seed bag.
2. **Plant it** — your seed bags sit at the bottom of the garden. Tap a bag, then tap an empty plot.
3. **Watch it grow** — trees grow taller and more valuable over time. The tag above each tree shows its current value.
4. **Harvest before the lightning** — storms roll in with a warning. Tall trees are the tastiest target for Zeus, and an unharvested tree that gets struck is lost forever (you only salvage 5% as charcoal). Bank "good enough" payouts.

## Features

- **9 seed tiers**, from River Seeds to Solar Seeds, each growing bigger and rarer
- **Dynamic weather**: clear skies → storm warning → lightning storm that hunts your tallest trees
- **Upgrades**: extra plots, fertilizer (+15% value each), growth boosters, and lightning rods (-12% risk each, max 5)
- **Greed streak 🔥**: harvest trees at 60%+ growth to stack up to +50% value — harvest too early and the streak resets
- **Rebirth system**: sacrifice everything for permanent Greed (+75% value & +15% growth speed per level)
- **Redeemable codes** (try `ILOVECATS`, `RIVER`, `GREEDY`, `BANJO`)
- **Local save** — progress persists in your browser, plus sound effects and toasts

## Files

- `index.html` — layout & UI
- `style.css` — styling
- `game.js` — game logic, canvas rendering, weather, save system

## Run

Serve the folder and open it — no build step:

```bash
python3 -m http.server 8080
```

Double-click the 💸 lifetime counter in the top bar to reset your save.
