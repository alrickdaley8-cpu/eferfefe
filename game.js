/* =========================================================
   GREEDY GROWERS — a web tribute to the Roblox game
   Buy seeds at the river → plant in your plot → watch the
   tree grow → harvest BEFORE the lightning turns it to ash.
   ========================================================= */
"use strict";

/* ---------------- data ---------------- */

const TIERS = [
  // { name, ico, cost, base, max, rate(s), leaf colors (light,dark), glow }
  { name: "River Seed",    ico: "🌰", cost: 10,        base: 5,      max: 30,      rate: 45,  leaf: ["#6fbf4f", "#3e8f2e"] },
  { name: "Willow Seed",   ico: "🌿", cost: 60,        base: 30,     max: 200,     rate: 55,  leaf: ["#9ccc65", "#689f38"] },
  { name: "Maple Seed",    ico: "🍁", cost: 250,       base: 130,    max: 1000,    rate: 65,  leaf: ["#ff8f5a", "#d95c3f"] },
  { name: "Golden Seed",   ico: "🌟", cost: 1000,      base: 550,    max: 5200,    rate: 75,  leaf: ["#ffd54f", "#f0a832"], glow: true },
  { name: "Crystal Seed",  ico: "💎", cost: 4500,      base: 2400,   max: 28000,   rate: 90,  leaf: ["#4dd0e1", "#26a5b8"], glow: true },
  { name: "Greedy Seed",   ico: "🍇", cost: 18000,     base: 10000,  max: 120000,  rate: 110, leaf: ["#ab47bc", "#7b2f8f"], glow: true },
  { name: "Titanic Seed",  ico: "🌳", cost: 75000,     base: 42000,  max: 560000,  rate: 140, leaf: ["#a5433a", "#6f2a26"] },
  { name: "Cosmic Seed",   ico: "🪐", cost: 300000,    base: 170000, max: 2400000, rate: 180, leaf: ["#7b8cff", "#4a56c8"], glow: true },
  { name: "Solar Seed",    ico: "☀️", cost: 1200000,   base: 700000, max: 10500000,rate: 240, leaf: ["#fff176", "#ffb300"], glow: true },
];

const UPGRADES = [
  {
    key: "plots", name: "Extra Plot", ico: "🟫",
    desc: "Unlock another dirt plot so you can grow more trees at once.",
    costs: [150, 1200, 5000, 18000, 65000, 200000, 600000, 1600000],
    max: 9, // total plots including the starting one
  },
  {
    key: "fert", name: "Fertilizer", ico: "🧪",
    desc: "+15% tree value per level. Banjo-approved compost.",
    cost: (l) => 200 * Math.pow(2.2, l), max: 10,
  },
  {
    key: "growth", name: "Growth Booster", ico: "⚡",
    desc: "+20% growth speed per level. Greed waits for no one.",
    cost: (l) => 150 * Math.pow(2.3, l), max: 10,
  },
  {
    key: "rod", name: "Lightning Rod", ico: "⛩️",
    desc: "-12% lightning risk per level. Cap at 5 (max -60%).",
    cost: (l) => 300 * Math.pow(3, l), max: 5,
  },
];

const CODES = {
  "ILOVECATS": { reward: 500,  label: "Starter coins" },
  "RIVER":     { reward: 250,  label: "River bonus" },
  "GREEDY":    { reward: 1000, label: "Greed bonus" },
  "BANJO":     { reward: 2000, label: "Creator's blessing" },
};

const fmt = (n) => {
  if (n >= 1e9) return (n / 1e9).toFixed(n >= 1e10 ? 0 : 2) + "B";
  if (n >= 1e6) return (n / 1e6).toFixed(n >= 1e7 ? 0 : 2) + "M";
  if (n >= 1e4) return (n / 1e3).toFixed(1) + "K";
  return Math.floor(n).toLocaleString();
};
const rnd = (a, b) => a + Math.random() * (b - a);
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

/* ---------------- state ---------------- */

const SAVE_KEY = "greedy-growers-v1";

function defaultState() {
  return {
    coins: 25,                 // a little pocket money
    lifetime: 0,               // total coins ever earned
    greed: 0,                  // rebirth level
    streak: 0,                 // greed streak
    codes: [],
    upgrades: { plots: 1, fert: 0, growth: 0, rod: 0 },
    plots: [{ tier: null, growth: 0, burnt: false, plantedAt: 0 }],
    inventory: {},             // tierIdx -> count of seed bags
    selected: null,            // tierIdx of the seed bag in hand
    stats: { harvested: 0, lost: 0, bestTree: 0, storms: 0, strikes: 0, burntSalvage: 0 },
    settings: { sound: true },
  };
}

let S = load();
function load() {
  try {
    const raw = localStorage.getItem(SAVE_KEY);
    if (!raw) return defaultState();
    const d = JSON.parse(raw);
    const base = defaultState();
    return { ...base, ...d, upgrades: { ...base.upgrades, ...(d.upgrades || {}) },
             stats: { ...base.stats, ...(d.stats || {}) }, settings: { ...base.settings, ...(d.settings || {}) } };
  } catch (e) { return defaultState(); }
}
function save() {
  try { localStorage.setItem(SAVE_KEY, JSON.stringify(S)); } catch (e) {}
}

/* ensure plots array matches owned count */
function syncPlots() {
  const want = Math.min(S.upgrades.plots, UPGRADES[0].max);
  while (S.plots.length < want) S.plots.push({ tier: null, growth: 0, burnt: false, plantedAt: 0 });
  while (S.plots.length > want) S.plots.pop();
}
syncPlots();

/* ---------------- derived helpers ---------------- */

function easeGrowth(g) { return Math.pow(g, 1.15); }
function greedMult()  { return 1 + 0.75 * S.greed; }
function fertMult()   { return 1 + 0.15 * S.upgrades.fert; }
function growthMult() { return (1 + 0.2 * S.upgrades.growth) * (1 + 0.15 * S.greed); }
function streakMult() { return 1 + 0.1 * S.streak; }

function treeValue(plot) {
  const t = TIERS[plot.tier];
  const e = easeGrowth(plot.growth);
  let v = t.base + (t.max - t.base) * e;
  v *= fertMult() * greedMult() * streakMult();
  return Math.max(1, Math.round(v));
}
function burntSalvage(plot) {
  return plot.salvage ?? Math.max(1, Math.round(treeValue(plot) * 0.05));
}

function rebirthRequirement() {
  return 1000000 * Math.pow(2.5, S.greed);
}

/* ---------------- weather ---------------- */

const WEATHER = {
  phase: "calm",     // calm | warn | storm
  t: rnd(45, 75),    // seconds left in phase
  warnT: 8,          // warning length
  stormT: 0,
  tick: 0,           // strike tick timer
  flash: 0,          // lightning flash alpha
  bolt: null,        // {x0,y0,x1,y1,age}
};

/* ---------------- particles & floating text ---------------- */

let floats = [];   // {x,y,txt,color,t,life,size}
let sparks = [];   // {x,y,vx,vy,t,life,color,size}
let smokes = [];   // {x,y,t,life,size}
let rain = [];     // {x,y,len,spd}
let clouds = [];   // {x,y,s,spd,storm}

for (let i = 0; i < 5; i++) {
  clouds.push({ x: Math.random() * 1600, y: 20 + Math.random() * 90, s: 0.7 + Math.random() * 0.9, spd: 6 + Math.random() * 8, storm: false });
}
for (let i = 0; i < 70; i++) rain.push({ x: Math.random() * 2000, y: Math.random() * 1200, len: 12 + Math.random() * 10, spd: 500 + Math.random() * 300 });

function addFloat(x, y, txt, color, size = 15) {
  floats.push({ x, y, txt, color, t: 0, life: 1.6, size });
}
function addSparks(x, y, n, color) {
  for (let i = 0; i < n; i++) {
    const a = Math.random() * Math.PI * 2, sp = rnd(20, 90);
    sparks.push({ x, y, vx: Math.cos(a) * sp, vy: Math.sin(a) * sp - 60, t: 0, life: rnd(0.5, 1.1), color, size: rnd(1.5, 3.5) });
  }
}
function addSmoke(x, y) {
  for (let i = 0; i < 6; i++) smokes.push({ x: x + rnd(-8, 8), y: y + rnd(-4, 4), t: 0, life: rnd(1.2, 2.2), size: rnd(3, 7) });
}

/* ---------------- audio ---------------- */

let AC = null;
function ac() { if (!AC) AC = new (window.AudioContext || window.webkitAudioContext)(); if (AC.state === "suspended") AC.resume(); return AC; }
function tone(freq, dur, type = "sine", vol = 0.15, slide = 0) {
  if (!S.settings.sound) return;
  try {
    const a = ac(), o = a.createOscillator(), g = a.createGain();
    o.type = type; o.frequency.value = freq;
    if (slide) o.frequency.exponentialRampToValueAtTime(Math.max(30, freq + slide), a.currentTime + dur);
    g.gain.setValueAtTime(vol, a.currentTime);
    g.gain.exponentialRampToValueAtTime(0.0001, a.currentTime + dur);
    o.connect(g).connect(a.destination);
    o.start(); o.stop(a.currentTime + dur + 0.02);
  } catch (e) {}
}
function noiseBurst(dur, vol = 0.2, lp = 400) {
  if (!S.settings.sound) return;
  try {
    const a = ac(), len = Math.max(1, Math.floor(a.sampleRate * dur));
    const buf = a.createBuffer(1, len, a.sampleRate);
    const d = buf.getChannelData(0);
    for (let i = 0; i < len; i++) d[i] = (Math.random() * 2 - 1) * (1 - i / len);
    const src = a.createBufferSource(); src.buffer = buf;
    const f = a.createBiquadFilter(); f.type = "lowpass"; f.frequency.value = lp;
    const g = a.createGain(); g.gain.value = vol;
    src.connect(f).connect(g).connect(a.destination);
    src.start();
  } catch (e) {}
}
const sfx = {
  coin:    () => { tone(880, 0.09, "square", 0.06); setTimeout(() => tone(1318, 0.12, "square", 0.06), 60); },
  plant:   () => { tone(180, 0.12, "triangle", 0.2, 140); },
  buy:     () => { tone(520, 0.07, "square", 0.07); setTimeout(() => tone(780, 0.09, "square", 0.07), 70); },
  buyErr:  () => { tone(200, 0.15, "sawtooth", 0.08, -60); },
  thunder: () => { noiseBurst(0.9, 0.35, 180); tone(70, 0.8, "sine", 0.3, -25); },
  warn:    () => { tone(660, 0.25, "sawtooth", 0.08); setTimeout(() => tone(660, 0.25, "sawtooth", 0.08), 300); },
  harvest: () => { tone(784, 0.1, "triangle", 0.14); setTimeout(() => tone(988, 0.1, "triangle", 0.14), 80); setTimeout(() => tone(1319, 0.16, "triangle", 0.14), 160); },
  lost:    () => { tone(300, 0.3, "sawtooth", 0.1, -120); },
  ui:      () => { tone(440, 0.05, "sine", 0.05); },
  rebirth: () => { [523, 659, 784, 1047].forEach((f, i) => setTimeout(() => tone(f, 0.18, "triangle", 0.15), i * 120)); },
};

/* ---------------- DOM refs ---------------- */

const $ = (id) => document.getElementById(id);
const el = {
  coin: $("coinVal"), life: $("lifeVal"), greed: $("greedVal"), streak: $("streakVal"),
  streakStat: $("streakStat"), weatherChip: $("weatherChip"), weatherIco: $("weatherIco"), weatherText: $("weatherText"),
  soundBtn: $("soundBtn"), helpBtn: $("helpBtn"), shopList: $("shopList"), upgradeList: $("upgradeList"),
  invItems: $("invItems"), invEmpty: $("invEmpty"), selectedHint: $("selectedHint"), selectedName: $("selectedName"),
  codeInput: $("codeInput"), codeBtn: $("codeBtn"), codeHint: $("codeHint"), codeList: $("codeList"),
  rebirthReq: $("rebirthReq"), rebirthBar: $("rebirthBar"), rebirthPct: $("rebirthPct"), rebirthBtn: $("rebirthBtn"),
  statGrid: $("statGrid"), toasts: $("toasts"),
  modal: $("modal"), modalX: $("modalX"), modalGo: $("modalGo"),
  confirmModal: $("confirmModal"), confirmTitle: $("confirmTitle"), confirmBody: $("confirmBody"),
  confirmYes: $("confirmYes"), confirmNo: $("confirmNo"),
};

/* ---------------- toasts ---------------- */

function toast(msg, kind = "") {
  const t = document.createElement("div");
  t.className = "toast " + kind;
  t.textContent = msg;
  el.toasts.appendChild(t);
  setTimeout(() => { t.classList.add("out"); setTimeout(() => t.remove(), 350); }, 2600);
  while (el.toasts.children.length > 4) el.toasts.firstChild.remove();
}

/* ---------------- HUD rendering ---------------- */

function renderHUD() {
  el.coin.textContent = fmt(S.coins);
  el.life.textContent = fmt(S.lifetime);
  el.greed.textContent = S.greed;
  el.streak.textContent = S.streak;
  el.streakStat.classList.toggle("streak-on", S.streak > 0);
  el.soundBtn.textContent = S.settings.sound ? "🔊" : "🔇";

  const w = WEATHER;
  el.weatherChip.classList.toggle("warn", w.phase === "warn");
  el.weatherChip.classList.toggle("storm", w.phase === "storm");
  if (w.phase === "calm") { el.weatherIco.textContent = "☀️"; el.weatherText.textContent = `Clear — next storm in ${Math.ceil(w.t)}s`; }
  else if (w.phase === "warn") { el.weatherIco.textContent = "🌩️"; el.weatherText.textContent = `Storm in ${Math.ceil(w.t)}s — harvest!`; }
  else { el.weatherIco.textContent = "⛈️"; el.weatherText.textContent = `STORM! Protect your trees (${Math.ceil(w.t)}s)`; }
}

function renderInventory() {
  el.invItems.innerHTML = "";
  const entries = Object.entries(S.inventory).map(([k, c]) => [+k, c]).filter(([, c]) => c > 0).sort((a, b) => a[0] - b[0]);
  if (entries.length === 0) { el.invEmpty.classList.remove("hidden"); el.invItems.classList.add("hidden"); }
  else {
    el.invEmpty.classList.add("hidden"); el.invItems.classList.remove("hidden");
    for (const [idx, count] of entries) {
      const t = TIERS[idx];
      const chip = document.createElement("div");
      chip.className = "seedchip" + (S.selected === idx ? " selected" : "");
      chip.innerHTML = `<span class="seed-ico">${t.ico}</span><span>${t.name.replace(" Seed", "")}</span><span class="seed-count">×${count}</span>`;
      chip.onclick = (e) => { e.stopPropagation(); toggleSelect(idx); };
      el.invItems.appendChild(chip);
    }
  }
  el.selectedHint.classList.toggle("hidden", S.selected === null);
  if (S.selected !== null) el.selectedName.textContent = TIERS[S.selected].name;
}

function renderShop() {
  el.shopList.innerHTML = "";
  TIERS.forEach((t, i) => {
    const item = document.createElement("div");
    item.className = "shop-item";
    const can = S.coins >= t.cost;
    item.innerHTML = `
      <div class="shop-ico">${t.ico}</div>
      <div class="shop-info">
        <div class="shop-name">${t.name}</div>
        <div class="shop-desc">Grows to ${fmt(t.max)} · ${Math.round(t.rate)}s to mature</div>
      </div>
      <button class="shop-btn ${can ? "" : "poor"}">Buy ${fmt(t.cost)} 🪙</button>`;
    const btn = item.querySelector(".shop-btn");
    btn.onclick = (e) => { e.stopPropagation(); buySeed(i); };
    el.shopList.appendChild(item);
  });
}

function renderUpgrades() {
  el.upgradeList.innerHTML = "";
  for (const u of UPGRADES) {
    const item = document.createElement("div");
    item.className = "shop-item";
    const lvl = S.upgrades[u.key];
    const maxed = u.key === "plots" ? lvl >= u.max : lvl >= u.max;
    let costTxt, can = false;
    if (maxed) costTxt = "MAX";
    else {
      const cost = u.key === "plots" ? u.costs[lvl - 1] : u.cost(lvl);
      can = S.coins >= cost;
      costTxt = `Lv ${lvl} → ${fmt(cost)} 🪙`;
    }
    item.innerHTML = `
      <div class="shop-ico">${u.ico}</div>
      <div class="shop-info">
        <div class="shop-name">${u.name} <span class="owned-tag">Lv ${lvl}/${u.max}</span></div>
        <div class="shop-desc">${u.desc}</div>
      </div>
      <button class="shop-btn ${can ? "" : "poor"}" ${maxed ? "disabled" : ""}>${costTxt}</button>`;
    const btn = item.querySelector(".shop-btn");
    btn.onclick = (e) => { e.stopPropagation(); buyUpgrade(u.key); };
    if (maxed) btn.style.cursor = "default";
    el.upgradeList.appendChild(item);
  }
}

function renderCodes() {
  const redeemed = new Set(S.codes);
  el.codeList.innerHTML = "";
  const names = Object.keys(CODES);
  if (names.length === 0) el.codeList.innerHTML = `<span style="color:var(--muted);font-size:12px">No codes yet. Stare at the sky.</span>`;
  names.forEach((c) => {
    const tag = document.createElement("div");
    tag.className = "code-tag" + (redeemed.has(c) ? " redeemed" : "");
    tag.textContent = c + (redeemed.has(c) ? " ✓" : "");
    el.codeList.appendChild(tag);
  });
  el.codeHint.textContent = redeemed.size === 0 ? "Tip: try ILOVECATS, RIVER, GREEDY, BANJO" : "";
}

function renderRebirth() {
  const req = rebirthRequirement();
  const pct = clamp(S.lifetime / req, 0, 1);
  el.rebirthReq.innerHTML = `Reach <b>${fmt(req)}</b> lifetime earnings to rebirth. You have <b>${fmt(S.lifetime)}</b>.`;
  el.rebirthBar.style.width = (pct * 100).toFixed(1) + "%";
  el.rebirthPct.textContent = (pct * 100).toFixed(1) + "% — next Greed level grants +75% value & +15% growth speed";
  el.rebirthBtn.disabled = S.lifetime < req;
  el.rebirthBtn.textContent = `Rebirth (→ Greed ${S.greed + 1})`;

  const st = S.stats;
  const rows = [
    ["Trees harvested", st.harvested], ["Trees lost to lightning", st.lost],
    ["Lightning strikes", st.strikes], ["Storms survived", st.storms],
    ["Best single tree", "$" + fmt(st.bestTree)], ["Charcoal salvaged", "$" + fmt(st.burntSalvage)],
    ["Greed level", S.greed], ["Total upgrades", S.upgrades.fert + S.upgrades.growth + S.upgrades.rod + (S.upgrades.plots - 1)],
  ];
  el.statGrid.innerHTML = rows.map(([k, v]) => `<div class="stat-cell">${k}<b>${v}</b></div>`).join("");
}

function renderAll() {
  renderHUD(); renderInventory(); renderShop(); renderUpgrades(); renderCodes(); renderRebirth();
}

/* ---------------- actions ---------------- */

function buySeed(idx) {
  const t = TIERS[idx];
  if (!t) return;
  if (S.coins < t.cost) { sfx.buyErr(); toast("Not enough coins for " + t.name + "!", "bad"); return; }
  S.coins -= t.cost;
  S.inventory[idx] = (S.inventory[idx] || 0) + 1;
  sfx.buy();
  toast(`Bought ${t.name} for ${fmt(t.cost)} 🪙`);
  renderAll();
  save();
}

function toggleSelect(idx) {
  S.selected = S.selected === idx ? null : idx;
  if (S.selected !== null) sfx.ui();
  renderAll();
}

function plantSeed(idx, plotIdx) {
  const plot = S.plots[plotIdx];
  if (plot.tier !== null || plot.burnt) return;
  if (S.selected === null) return;
  if ((S.inventory[S.selected] || 0) <= 0) { S.selected = null; renderAll(); return; }
  plot.tier = S.selected; plot.growth = 0; plot.plantedAt = performance.now(); plot.burnt = false;
  S.inventory[S.selected]--;
  if (S.inventory[S.selected] <= 0) delete S.inventory[S.selected];
  sfx.plant();
  const pt = plotRects[plotIdx];
  addFloat(pt.cx, pt.y - 14, "🌰 planted!", "#a8e6a3");
  toast(`${TIERS[plot.tier].name} planted on plot ${plotIdx + 1}`);
  renderAll(); save();
}

function harvestPlot(plotIdx) {
  const plot = S.plots[plotIdx];
  if (plot.tier === null || plot.burnt) return;
  const v = treeValue(plot);
  const g = plot.growth;
  S.coins += v; S.lifetime += v;
  S.stats.harvested++; S.stats.bestTree = Math.max(S.stats.bestTree, v);

  // greed streak: late harvests stack it, early harvests kill it
  if (g >= 0.6) {
    S.streak = Math.min(5, S.streak + 1);
    if (S.streak === 5) toast("🔥 MAX GREED STREAK! +50% value on the next trees!", "gold");
  } else {
    if (S.streak > 0) toast("Harvested too early — greed streak reset 💔", "bad");
    S.streak = 0;
  }

  const pt = plotRects[plotIdx];
  addFloat(pt.cx, pt.y - 26, "+$" + fmt(v), "#8dffa3", 19);
  addSparks(pt.cx, pt.y - 20, 10, "#ffd166");
  sfx.harvest();
  plot.tier = null; plot.growth = 0; plot.burnt = false;
  renderAll(); save();
}

function buyUpgrade(key) {
  const u = UPGRADES.find((x) => x.key === key);
  const lvl = S.upgrades[key];
  const maxed = key === "plots" ? lvl >= u.max : lvl >= u.max;
  if (maxed) return;
  const cost = key === "plots" ? u.costs[lvl - 1] : u.cost(lvl);
  if (S.coins < cost) { sfx.buyErr(); toast("Not enough coins!", "bad"); return; }
  S.coins -= cost;
  S.upgrades[key]++;
  sfx.buy();
  toast(`${u.name} → Lv ${S.upgrades[key]}`);
  if (key === "plots") { syncPlots(); layout(); toast(`New plot unlocked! ${S.upgrades.plots} total 🟫`, "gold"); }
  renderAll(); save();
}

function doRebirth() {
  if (S.lifetime < rebirthRequirement()) return;
  const earned = S.lifetime;
  S.greed++;
  S.coins = 25; S.lifetime = 0; S.streak = 0;
  S.upgrades = { plots: 1, fert: 0, growth: 0, rod: 0 };
  S.inventory = {};
  S.plots = [{ tier: null, growth: 0, burnt: false, plantedAt: 0 }];
  S.selected = null;
  layout();
  sfx.rebirth();
  toast(`😈 REBIRTH! Greed level ${S.greed} — all future trees worth +${75 * S.greed}%`, "gold");
  renderAll(); save();
}

function redeemCode() {
  const code = el.codeInput.value.trim().toUpperCase();
  if (!code) return;
  if (!CODES[code]) { sfx.buyErr(); toast(`"${code}" is not a real code. Greedy.`, "bad"); return; }
  if (S.codes.includes(code)) { sfx.buyErr(); toast(`"${code}" was already redeemed!`, "bad"); return; }
  S.codes.push(code);
  S.coins += CODES[code].reward;
  sfx.coin();
  toast(`🎁 Code "${code}" redeemed: +${fmt(CODES[code].reward)} coins! (${CODES[code].label})`, "gold");
  el.codeInput.value = "";
  renderAll(); save();
}

/* ---------------- canvas ---------------- */

const canvas = $("game");
const ctx = canvas.getContext("2d");
let W = 0, H = 0, dpr = 1;
let groundY = 0, riverW = 0;
let plotRects = [];
let shackRect = null;
let hoverPlot = -1;
let hoverShack = false;
let lastT = performance.now();
let lastTouchT = 0; // guard against touchstart + synthetic click double-fire

function layout() {
  const rect = canvas.getBoundingClientRect();
  dpr = Math.min(window.devicePixelRatio || 1, 2);
  W = rect.width; H = rect.height;
  canvas.width = Math.round(W * dpr);
  canvas.height = Math.round(H * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  groundY = H * 0.46;
  riverW = Math.min(150, W * 0.24);

  // plots: 3 columns
  const cols = 3;
  const rows = Math.ceil(S.upgrades.plots / cols);
  const pad = 14;
  const areaX = riverW + pad + 6;
  const areaW = W - areaX - pad;
  const areaH = H - groundY - pad * 2;
  const plotW = areaW / cols;
  const plotH = areaH / rows;
  plotRects = [];
  for (let i = 0; i < S.upgrades.plots; i++) {
    const c = i % cols, r = Math.floor(i / cols);
    const x = areaX + c * plotW + plotW * 0.06;
    const y = groundY + pad + r * plotH + plotH * 0.05;
    const w = plotW * 0.88, h = plotH * 0.9;
    plotRects.push({ x, y, w, h, cx: x + w / 2, cy: y + h / 2 });
  }
  shackRect = { x: 10, y: groundY + 10, w: riverW * 0.8, h: 92 };
}

function treeHeight(g, s) { return s * (0.45 + 0.85 * g); }

function drawTree(plot, cx, baseY, g, time, plotW) {
  const s = Math.min(plotW * 0.9, 96);
  const h = treeHeight(g, s);
  const sway = Math.sin(time * 1.4 + cx * 0.05) * (2 + 6 * g);
  const topX = cx + sway;
  const topY = baseY - h;

  if (g < 0.12) { // sprout
    ctx.fillStyle = "#5fbf4f";
    ctx.beginPath(); ctx.ellipse(cx - 5, baseY - 12, 5, 9, -0.5, 0, Math.PI * 2); ctx.fill();
    ctx.beginPath(); ctx.ellipse(cx + 5, baseY - 12, 5, 9, 0.5, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = "#7a5230"; ctx.lineWidth = 2.5;
    ctx.beginPath(); ctx.moveTo(cx, baseY); ctx.lineTo(cx, baseY - 10); ctx.stroke();
    return;
  }

  // trunk
  const trunkW = s * (0.07 + 0.05 * g);
  const trunkH = h * 0.38;
  ctx.fillStyle = "#7a5230";
  ctx.fillRect(cx - trunkW / 2, baseY - trunkH, trunkW, trunkH);

  // branches
  ctx.strokeStyle = "#6b4626"; ctx.lineWidth = 2.5;
  ctx.beginPath(); ctx.moveTo(cx, baseY - trunkH * 0.6); ctx.lineTo(cx + trunkW * 1.4, baseY - trunkH * 0.9); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(cx, baseY - trunkH * 0.75); ctx.lineTo(cx - trunkW * 1.4, baseY - trunkH * 1.05); ctx.stroke();

  // canopy
  const tier = TIERS[plot.tier];
  const [light, dark] = tier.leaf;
  const canH = h - trunkH;
  const canW = s * (0.5 + 0.45 * g);
  const n = 3 + Math.floor(g * 3);

  ctx.save();
  ctx.translate(topX, topY + canH * 0.55);
  if (tier.glow) { ctx.shadowColor = light; ctx.shadowBlur = 14 + 10 * g; }
  for (let i = 0; i < n; i++) {
    const a = (i / n) * Math.PI * 2 + time * 0.3;
    const px = Math.cos(a) * canW * 0.32;
    const py = Math.sin(a) * canH * 0.26;
    const r = canW * (0.3 + 0.1 * Math.sin(i * 2.7 + time));
    ctx.fillStyle = i % 2 ? light : dark;
    ctx.beginPath(); ctx.arc(px, py, r, 0, Math.PI * 2); ctx.fill();
    // top highlight
    ctx.fillStyle = "rgba(255,255,255,.18)";
    ctx.beginPath(); ctx.arc(px - r * 0.25, py - r * 0.3, r * 0.45, 0, Math.PI * 2); ctx.fill();
  }
  ctx.restore();

  // sparkles when fully grown
  if (g >= 0.97) {
    const fl = Math.sin(time * 6 + cx) * 0.5 + 0.5;
    ctx.fillStyle = `rgba(255,240,150,${0.5 + fl * 0.5})`;
    ctx.font = "13px serif"; ctx.textAlign = "center";
    ctx.fillText("✦", topX, topY + 6);
  }
  // fruits for higher tiers
  if (tier.tier >= 3 && g > 0.5) {
    ctx.fillStyle = "rgba(255,120,90,.9)";
    for (let i = 0; i < 3; i++) {
      const a = time * 0.5 + i * 2.1;
      ctx.beginPath(); ctx.arc(topX + Math.cos(a) * canW * 0.2, baseY - trunkH + 6 + i * 6, 2.5, 0, Math.PI * 2); ctx.fill();
    }
  }
}

function drawSky(time, storminess) {
  const g = ctx.createLinearGradient(0, 0, 0, groundY);
  if (storminess > 0.5) g.addColorStop(0, "#3c4454"); else g.addColorStop(0, "#5eb2e8");
  g.addColorStop(1, storminess > 0.5 ? "#6b6f86" : "#cfeaf7");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, W, groundY);

  // sun
  if (storminess <= 0.5) {
    ctx.save();
    ctx.globalAlpha = 0.9;
    ctx.fillStyle = "#ffe98a";
    ctx.shadowColor = "#ffe98a"; ctx.shadowBlur = 30;
    ctx.beginPath(); ctx.arc(W - 70, 55, 26, 0, Math.PI * 2); ctx.fill();
    ctx.restore();
  }

  // clouds
  for (const c of clouds) {
    c.storm = storminess > 0.5;
    c.x += c.spd * (1 / 60);
    if (c.x > W + 160) c.x = -160;
    const col = c.storm ? "#4d5568" : "rgba(255,255,255,.92)";
    ctx.fillStyle = col;
    const s = c.s;
    ctx.beginPath();
    ctx.arc(c.x, c.y, 22 * s, 0, Math.PI * 2);
    ctx.arc(c.x + 24 * s, c.y - 10 * s, 18 * s, 0, Math.PI * 2);
    ctx.arc(c.x + 48 * s, c.y, 20 * s, 0, Math.PI * 2);
    ctx.arc(c.x + 24 * s, c.y + 8 * s, 20 * s, 0, Math.PI * 2);
    ctx.fill();
    if (c.storm) { // little lightning bolt on storm cloud
      ctx.strokeStyle = "#ffe066"; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(c.x + 30 * s, c.y + 10 * s); ctx.lineTo(c.x + 26 * s, c.y + 22 * s); ctx.lineTo(c.x + 30 * s, c.y + 20 * s); ctx.lineTo(c.x + 24 * s, c.y + 32 * s); ctx.stroke();
    }
  }
}

function drawGround() {
  const g = ctx.createLinearGradient(0, groundY, 0, H);
  g.addColorStop(0, "#7cb75e");
  g.addColorStop(1, "#5d9a48");
  ctx.fillStyle = g;
  ctx.fillRect(0, groundY, W, H - groundY);

  // grass tufts
  ctx.strokeStyle = "rgba(60,120,50,.5)"; ctx.lineWidth = 1.5;
  for (let i = 0; i < 24; i++) {
    const gx = (i * 97.3 + 30) % W, gy = groundY + 18 + ((i * 53.7) % (H - groundY - 36));
    if (gx < riverW + 20) continue;
    ctx.beginPath(); ctx.moveTo(gx, gy);
    ctx.quadraticCurveTo(gx + 3, gy - 7, gx + 6, gy - 9);
    ctx.moveTo(gx, gy); ctx.quadraticCurveTo(gx - 3, gy - 7, gx - 6, gy - 9);
    ctx.stroke();
  }
}

function drawRiver(time) {
  // water
  const g = ctx.createLinearGradient(0, groundY, 0, H);
  g.addColorStop(0, "#5f9ed1"); g.addColorStop(1, "#3f7fb8");
  ctx.fillStyle = g;
  ctx.fillRect(0, groundY, riverW, H - groundY);

  // animated flow
  ctx.strokeStyle = "rgba(255,255,255,.28)"; ctx.lineWidth = 2;
  for (let i = 0; i < 6; i++) {
    const y = groundY + 20 + i * ((H - groundY) / 7) + Math.sin(time * 0.8 + i * 1.3) * 4;
    ctx.beginPath();
    for (let x = -8; x <= riverW + 8; x += 6) {
      const yy = y + Math.sin(x * 0.18 + time * 2 + i) * 3;
      if (x === -8) ctx.moveTo(x, yy); else ctx.lineTo(x, yy);
    }
    ctx.stroke();
  }
  // bank
  ctx.fillStyle = "#8a6a44";
  ctx.fillRect(riverW - 5, groundY, 8, H - groundY);
}

function drawShack(time) {
  const r = shackRect;
  // sign
  ctx.fillStyle = "#6b4626";
  ctx.fillRect(r.x + 4, r.y - 26, r.w - 8, 30);
  ctx.fillStyle = "#f5e6c8";
  ctx.fillRect(r.x + 9, r.y - 21, r.w - 18, 20);
  ctx.fillStyle = "#5d3a1a";
  ctx.font = "bold 10px sans-serif"; ctx.textAlign = "center";
  ctx.fillText("RIVER SEEDS", r.x + r.w / 2, r.y - 8);

  // stall body
  ctx.fillStyle = "#9a7446";
  ctx.fillRect(r.x, r.y + 2, r.w, 34);
  ctx.fillStyle = "#8a6540";
  ctx.fillRect(r.x, r.y + 2, r.w, 8);
  // counter
  ctx.fillStyle = "#b98c54";
  ctx.fillRect(r.x - 3, r.y + 30, r.w + 6, 8);
  // seed jars
  ["🌰", "🌿", "🍁"].forEach((em, i) => {
    ctx.font = "13px serif"; ctx.textAlign = "center";
    ctx.fillText(em, r.x + 12 + i * (r.w / 3), r.y + 24);
  });
  // bobbing
  const bob = Math.sin(time * 2) * 2;
  ctx.font = "16px serif"; ctx.textAlign = "center";
  ctx.fillText("🪙", r.x + r.w / 2, r.y + 58 + bob);
  if (hoverShack) {
    ctx.strokeStyle = "#ffd166"; ctx.lineWidth = 2;
    ctx.strokeRect(r.x - 4, r.y - 32, r.w + 8, 96);
  }
}

function drawPlot(i, time) {
  const p = plotRects[i];
  const plot = S.plots[i];

  // dirt
  ctx.fillStyle = "#8a6a44";
  roundRect(p.x, p.y, p.w, p.h, 10); ctx.fill();
  ctx.fillStyle = "#7a5c3c";
  roundRect(p.x + 3, p.y + 3, p.w - 6, p.h - 6, 8); ctx.fill();
  // inner darker soil
  ctx.fillStyle = "#6a4e32";
  roundRect(p.x + 7, p.y + 7, p.w - 14, p.h - 14, 7); ctx.fill();

  // wooden frame
  ctx.strokeStyle = "#9a7446"; ctx.lineWidth = 4;
  roundRect(p.x, p.y, p.w, p.h, 10); ctx.stroke();

  if (plot.burnt) {
    // charred remains
    ctx.fillStyle = "#3a332b";
    roundRect(p.cx - 10, p.cy + 8, 20, 10, 4); ctx.fill();
    ctx.fillStyle = "#2c2620";
    ctx.fillRect(p.cx - 3, p.cy - 14, 6, 24);
    ctx.fillStyle = "rgba(120,110,95,.8)";
    ctx.font = "11px sans-serif"; ctx.textAlign = "center";
    ctx.fillText("🌫️ salvage +" + fmt(burntSalvage(plot)), p.cx, p.y + 12);
    return;
  }

  if (plot.tier !== null) {
    drawTree(plot, p.cx, p.y + p.h - 10, plot.growth, time, p.w);

    // value tag
    const v = treeValue(plot);
    const full = plot.growth >= 1;
    const bw = 74, bh = 20;
    const bx = clamp(p.cx - bw / 2, p.x + 2, p.x + p.w - bw - 2);
    const by = p.y - bh + 4;
    ctx.fillStyle = full ? "#ffd166" : "rgba(20,32,24,.85)";
    roundRect(bx, by, bw, bh, 9); ctx.fill();
    ctx.strokeStyle = full ? "#f0a832" : "rgba(255,255,255,.25)"; ctx.lineWidth = 1.5;
    roundRect(bx, by, bw, bh, 9); ctx.stroke();
    ctx.fillStyle = full ? "#3a2c00" : "#ffffff";
    ctx.font = `bold ${full ? 13 : 12}px sans-serif`; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText((full ? "✦ $": "$") + fmt(v), bx + bw / 2, by + bh / 2 + 1);
    ctx.textBaseline = "alphabetic";

    // growth bar under tag
    const gbw = 60, gbx = p.cx - gbw / 2, gby = by + bh + 2;
    ctx.fillStyle = "rgba(0,0,0,.45)";
    roundRect(gbx, gby, gbw, 5, 3); ctx.fill();
    ctx.fillStyle = plot.growth >= 1 ? "#ffd166" : "#7bd389";
    roundRect(gbx, gby, gbw * plot.growth, 5, 3); ctx.fill();

    // hover highlight
    if (hoverPlot === i) {
      ctx.strokeStyle = "rgba(255,255,255,.85)"; ctx.lineWidth = 2.5;
      roundRect(p.x - 3, p.y - 3, p.w + 6, p.h + 6, 12); ctx.stroke();
    }
  } else {
    // empty plot hint
    if (S.selected !== null && hoverPlot === i) {
      ctx.fillStyle = "rgba(255,255,255,.9)";
      ctx.font = "bold 13px sans-serif"; ctx.textAlign = "center";
      ctx.fillText("Plant " + TIERS[S.selected].ico, p.cx, p.cy + 4);
      ctx.strokeStyle = "#ffd166"; ctx.lineWidth = 2;
      roundRect(p.x - 3, p.y - 3, p.w + 6, p.h + 6, 12); ctx.stroke();
    }
    ctx.fillStyle = "rgba(255,255,255,.25)";
    ctx.font = "bold 16px sans-serif"; ctx.textAlign = "center";
    ctx.fillText("+", p.cx, p.cy + 6);
  }
}

function roundRect(x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function drawWeather(time, dt) {
  if (WEATHER.phase !== "storm") return;
  // rain
  ctx.strokeStyle = "rgba(180,210,255,.45)"; ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (const r of rain) {
    r.y += r.spd * dt; r.x -= r.spd * 0.25 * dt;
    if (r.y > H) { r.y = -20; r.x = Math.random() * W; }
    ctx.moveTo(r.x, r.y); ctx.lineTo(r.x - 3, r.y + r.len);
  }
  ctx.stroke();

  // flash
  if (WEATHER.flash > 0) {
    ctx.fillStyle = `rgba(255,255,255,${WEATHER.flash})`;
    ctx.fillRect(0, 0, W, H);
    WEATHER.flash -= dt * 1.6;
  }
  // bolt
  if (WEATHER.bolt) {
    const b = WEATHER.bolt;
    ctx.strokeStyle = "rgba(255,240,150,.95)"; ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(b.x0, b.y0);
    let x = b.x0, y = b.y0;
    const segs = 7;
    for (let i = 0; i < segs; i++) {
      const tx = b.x1, ty = b.y1;
      x += (tx - x) / (segs - i) + rnd(-14, 14);
      y += (ty - y) / (segs - i) + rnd(-4, 6);
      ctx.lineTo(x, y);
    }
    ctx.lineTo(b.x1, b.y1);
    ctx.stroke();
    ctx.strokeStyle = "rgba(255,255,255,.6)"; ctx.lineWidth = 6;
    ctx.stroke();
    b.age += dt;
    if (b.age > 0.18) WEATHER.bolt = null;
  }
}

function drawFloats(dt) {
  ctx.textAlign = "center";
  for (let i = floats.length - 1; i >= 0; i--) {
    const f = floats[i];
    f.t += dt;
    if (f.t >= f.life) { floats.splice(i, 1); continue; }
    const a = 1 - f.t / f.life;
    ctx.globalAlpha = a;
    ctx.font = `bold ${f.size}px sans-serif`;
    ctx.fillStyle = "rgba(0,0,0,.6)";
    ctx.fillText(f.txt, f.x + 1, f.y + 1);
    ctx.fillStyle = f.color;
    ctx.fillText(f.txt, f.x, f.y);
    ctx.globalAlpha = 1;
  }
  // sparks
  for (let i = sparks.length - 1; i >= 0; i--) {
    const s = sparks[i];
    s.t += dt;
    if (s.t >= s.life) { sparks.splice(i, 1); continue; }
    s.x += s.vx * dt; s.y += s.vy * dt; s.vy += 220 * dt;
    ctx.globalAlpha = 1 - s.t / s.life;
    ctx.fillStyle = s.color;
    ctx.beginPath(); ctx.arc(s.x, s.y, s.size, 0, Math.PI * 2); ctx.fill();
    ctx.globalAlpha = 1;
  }
  // smoke
  for (let i = smokes.length - 1; i >= 0; i--) {
    const s = smokes[i];
    s.t += dt;
    if (s.t >= s.life) { smokes.splice(i, 1); continue; }
    const a = 0.35 * (1 - s.t / s.life);
    ctx.fillStyle = `rgba(90,90,95,${a})`;
    ctx.beginPath(); ctx.arc(s.x, s.y - s.t * 26, s.size + s.t * 9, 0, Math.PI * 2); ctx.fill();
  }
}

/* ---------------- weather simulation ---------------- */

function updateWeather(dt) {
  const w = WEATHER;
  w.t -= dt;
  if (w.phase === "calm" && w.t <= 0) {
    w.phase = "warn"; w.t = w.warnT;
    sfx.warn();
    toast("🌩️ Storm incoming in 8 seconds! Harvest your tall trees!", "gold");
    renderHUD();
  } else if (w.phase === "warn" && w.t <= 0) {
    w.phase = "storm"; w.t = rnd(10, 16); w.tick = 0.6;
    S.stats.storms++;
    sfx.thunder();
    toast("⚡ THE STORM HITS! Lightning is hunting your trees!", "bad");
    renderHUD();
  } else if (w.phase === "storm") {
    if (w.t <= 0) {
      w.phase = "calm"; w.t = rnd(50, 85);
      toast("☀️ The storm passes. The greedy survive.", "");
      renderHUD();
    } else {
      w.tick -= dt;
      if (w.tick <= 0) {
        w.tick = rnd(1.1, 1.8);
        strikeChance();
      }
    }
  }
}

function strikeChance() {
  // each unharvested tree has a weighted chance; taller = tastier target
  const candidates = [];
  let totalW = 0;
  S.plots.forEach((p, i) => {
    if (p.tier === null || p.burnt) return;
    const g = p.growth;
    if (g < 0.2) return; // tiny trees are safe
    let w = (g - 0.2) / 0.8;
    if (g >= 1) w *= 1.5;
    candidates.push({ i, w });
    totalW += w;
  });
  if (candidates.length === 0) return;

  const rodMult = Math.max(0.4, 1 - 0.12 * S.upgrades.rod);
  // base storm ferocity: ~0.30 strike events per tick scaled by tree weight
  const p = 0.30 * rodMult;
  if (Math.random() > p) return;

  let r = Math.random() * totalW;
  let pick = candidates[0].i;
  for (const c of candidates) { r -= c.w; if (r <= 0) { pick = c.i; break; } }

  strikeTree(pick);
}

function strikeTree(plotIdx) {
  const plot = S.plots[plotIdx];
  const p = plotRects[plotIdx];
  const v = treeValue(plot);
  const salvage = Math.max(1, Math.round(v * 0.05));

  S.coins += salvage;
  S.stats.strikes++;
  S.stats.lost += v - salvage;
  S.stats.burntSalvage += salvage;

  WEATHER.flash = 0.55;
  WEATHER.bolt = { x0: p.cx + rnd(-60, 60), y0: 0, x1: p.cx, y1: p.y + p.h / 2, age: 0 };

  const wasFull = plot.growth >= 0.97;
  plot.burnt = true; plot.growth = 0; // keep tier; store salvage for the charred plot
  plot.salvage = salvage;

  addFloat(p.cx, p.y - 30, "⚡ -$" + fmt(v), "#ff8a8a", 19);
  addSmoke(p.cx, p.y + p.h / 2);
  sfx.thunder();
  sfx.lost();
  toast(`⚡ Lightning destroyed your ${TIERS[plot.tier].name}! Lost $${fmt(v)}${wasFull ? " — so greedy." : ""}`, "bad");

  // streak broken by greed
  if (S.streak > 0) { S.streak = 0; toast("Greed streak lost with the tree 💔", "bad"); }

  renderAll(); save();
}

function salvageBurnt(plotIdx) {
  const plot = S.plots[plotIdx];
  if (!plot.burnt) return;
  const v = burntSalvage(plot);
  S.coins += v;
  const p = plotRects[plotIdx];
  addFloat(p.cx, p.y - 20, "+$" + fmt(v) + " charcoal", "#c9b98a");
  sfx.coin();
  plot.tier = null; plot.burnt = false; plot.salvage = undefined;
  renderAll(); save();
}

/* ---------------- growth simulation ---------------- */

function updateGrowth(dt) {
  for (const plot of S.plots) {
    if (plot.tier === null || plot.burnt) continue;
    const rate = (1 / TIERS[plot.tier].rate) * growthMult();
    plot.growth = Math.min(1, plot.growth + rate * dt);
  }
}

/* ---------------- input ---------------- */

function canvasPos(e) {
  const rect = canvas.getBoundingClientRect();
  const t = e.touches ? e.touches[0] : e;
  return { x: t.clientX - rect.left, y: t.clientY - rect.top };
}

function hitTest(x, y) {
  if (shackRect && x >= shackRect.x - 4 && x <= shackRect.x + shackRect.w + 4 && y >= shackRect.y - 32 && y <= shackRect.y + 64) return "shack";
  for (let i = 0; i < plotRects.length; i++) {
    const p = plotRects[i];
    if (x >= p.x && x <= p.x + p.w && y >= p.y && y <= p.y + p.h) return "plot:" + i;
  }
  return null;
}

function handleClick(x, y) {
  const hit = hitTest(x, y);
  if (!hit) { if (S.selected !== null) { S.selected = null; renderAll(); } return; }
  if (hit === "shack") { openTab("shop"); sfx.ui(); return; }
  const plotIdx = +hit.split(":")[1];
  const plot = S.plots[plotIdx];

  if (plot.burnt) { salvageBurnt(plotIdx); return; }
  if (plot.tier !== null) { harvestPlot(plotIdx); return; }
  if (S.selected !== null) plantSeed(S.selected, plotIdx);
  else toast("No seed selected — buy one at the river shop first 🌊");
}

function onPointer(e) {
  const { x, y } = canvasPos(e);
  if (e.type === "mousemove") {
    const hit = hitTest(x, y);
    hoverPlot = hit && hit.startsWith("plot:") ? +hit.split(":")[1] : -1;
    hoverShack = hit === "shack";
    canvas.style.cursor = (hoverPlot >= 0 || hoverShack) ? "pointer" : "default";
  } else if (e.type === "click") {
    if (performance.now() - lastTouchT < 500) return; // already handled by touchstart
    handleClick(x, y);
  }
}

canvas.addEventListener("mousemove", onPointer);
canvas.addEventListener("click", onPointer);
canvas.addEventListener("touchstart", (e) => {
  if (e.touches.length === 1) {
    lastTouchT = performance.now();
    const { x, y } = canvasPos(e);
    handleClick(x, y);
  }
}, { passive: true });

/* ---------------- tabs & modals ---------------- */

function openTab(name) {
  document.querySelectorAll(".tabpane").forEach((t) => t.classList.add("hidden"));
  $("tab-" + name).classList.remove("hidden");
  document.querySelectorAll(".navbtn").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
}

document.querySelectorAll(".navbtn").forEach((b) => {
  b.addEventListener("click", () => { openTab(b.dataset.tab); sfx.ui(); });
});

function showModal(m) { m.classList.remove("hidden"); }
function hideModal(m) { m.classList.add("hidden"); }

el.modalGo.onclick = () => hideModal(el.modal);
el.modalX.onclick = () => hideModal(el.modal);
el.helpBtn.onclick = () => { showModal(el.modal); };
$("modal").addEventListener("click", (e) => { if (e.target === el.modal) hideModal(el.modal); });

let confirmCb = null;
function confirmDialog(title, body, cb, yesLabel = "Yes") {
  el.confirmTitle.textContent = title;
  el.confirmBody.textContent = body;
  $("confirmYes").textContent = yesLabel;
  confirmCb = cb;
  showModal(el.confirmModal);
}
el.confirmYes.onclick = () => { hideModal(el.confirmModal); if (confirmCb) confirmCb(); };
el.confirmNo.onclick = () => hideModal(el.confirmModal);
el.confirmModal.addEventListener("click", (e) => { if (e.target === el.confirmModal) hideModal(el.confirmModal); });

$("cancelSel").onclick = () => { S.selected = null; renderAll(); };

el.soundBtn.onclick = () => {
  S.settings.sound = !S.settings.sound;
  if (S.settings.sound) { ac(); sfx.ui(); }
  renderAll(); save();
};

el.codeBtn.onclick = redeemCode;
el.codeInput.addEventListener("keydown", (e) => { if (e.key === "Enter") redeemCode(); });

el.rebirthBtn.onclick = () => {
  confirmDialog("😈 Rebirth?",
    "You'll lose all coins, plots, upgrades and seed bags, but keep Greed level " + S.greed + " and unlock Greed " + (S.greed + 1) + " (+75% value, +15% growth speed forever).",
    doRebirth, "Rebirth now");
};

/* reset progress */
$("lifeStat").addEventListener("dblclick", () => {
  confirmDialog("💀 Reset everything?",
    "This wipes your entire save (coins, plots, upgrades, codes, stats). There is no undo.",
    () => { localStorage.removeItem(SAVE_KEY); location.reload(); }, "Wipe it");
});

/* ---------------- main loop ---------------- */

function frame(now) {
  const dt = Math.min(0.05, (now - lastT) / 1000);
  lastT = now;
  const time = now / 1000;

  updateGrowth(dt);
  updateWeather(dt);

  const storminess = WEATHER.phase === "storm" ? 1 : WEATHER.phase === "warn" ? 0.55 : 0;

  drawSky(time, storminess);
  drawGround();
  drawRiver(time);
  drawShack(time);
  for (let i = 0; i < plotRects.length; i++) drawPlot(i, time);
  drawWeather(time, dt);
  drawFloats(dt);

  requestAnimationFrame(frame);
}

/* ---------------- boot ---------------- */

window.addEventListener("resize", layout);
window.addEventListener("load", () => {
  layout();
  renderAll();
  if (!localStorage.getItem(SAVE_KEY)) showModal(el.modal); // first-run how-to
  requestAnimationFrame((t) => { lastT = t; requestAnimationFrame(frame); });
});

// autosave
setInterval(save, 5000);
window.addEventListener("beforeunload", save);
document.addEventListener("visibilitychange", () => { if (document.hidden) save(); });

// hide selected seed when clicking anywhere else
document.addEventListener("click", (e) => {
  if (S.selected !== null && !e.target.closest(".seedchip") && !e.target.closest("#game") && !e.target.closest("#selectedHint")) {
    S.selected = null;
    renderAll();
  }
});
