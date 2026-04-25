/**
 * image_composer.js  v1 - Eric_Composer_Studio
 *
 * Interactive image compositor.
 *   • Up to 6 image+mask (or RGBA) layers on a configurable canvas.
 *   • Per-layer: position, uniform scale, X/Y scale, rotation, flip H/V,
 *                blend mode, opacity, drop shadow, brightness/contrast/saturation,
 *                BG mode (cover-fit, mask ignored), visibility, z-order.
 *   • Procedural background (solid / gradient / paper / canvas / checker / dots / perlin).
 *   • Canvas resolution = aspect ratio × megapixels (0.5 / 1 / 2 / 4 MP).
 *   • Shift-drag to snap to centre/edges. Right-click layer for context menu.
 */

import { app } from "../../scripts/app.js";

const NODE_TYPE = "EricImageComposer";
const W_COMP    = "composition_data";
const MAX_SLOTS = 6;

const RATIO_BAR_H = 34;
const DETAIL_BAR_H = 34;
const SHADOW_BAR_H = 30;
const BG_BAR_H     = 34;
const MARGIN       = 10;
const MIN_COMP     = 80;
const MOV_R        = 9;
const EDGE_R       = 6;
const LG_RESIZE    = 16;
const ROT_R        = 7;
const ROT_DIST     = 30;

const SLOT_ROW_H  = 30;
const LIST_HDR_H  = 18;
const LIST_H      = LIST_HDR_H + MAX_SLOTS * SLOT_ROW_H + 6;

const SLOT_COLORS = ["#4A9EFF", "#FF6B35", "#4CAF50", "#E040FB", "#FFD740", "#26C6DA"];

const BLEND_MODES = [
    "normal", "multiply", "screen", "overlay", "soft_light", "hard_light",
    "add", "subtract", "difference", "darken", "lighten",
    "color_dodge", "color_burn", "hue", "saturation", "color", "luminosity",
];

// Canvas2D globalCompositeOperation mapping.
// "subtract" has no exact Canvas2D equivalent — we approximate via difference.
const BLEND_CANVAS = {
    normal: "source-over", multiply: "multiply", screen: "screen",
    overlay: "overlay", soft_light: "soft-light", hard_light: "hard-light",
    add: "plus-lighter", subtract: "difference",
    difference: "difference", darken: "darken", lighten: "lighten",
    color_dodge: "color-dodge", color_burn: "color-burn",
    hue: "hue", saturation: "saturation", color: "color", luminosity: "luminosity",
};

const BG_TYPES = [
    {v: "solid",        l: "Solid"},
    {v: "gradient_lin", l: "Linear gradient"},
    {v: "gradient_rad", l: "Radial gradient"},
    {v: "paper",        l: "Paper"},
    {v: "canvas",       l: "Canvas weave"},
    {v: "checker",      l: "Checker"},
    {v: "dots",         l: "Dots"},
    {v: "perlin",       l: "Perlin noise"},
];

const MP_PRESETS = [
    {v: 0.25, l: "0.25 MP"}, {v: 0.5, l: "0.5 MP"},
    {v: 1.0,  l: "1 MP"},    {v: 2.0, l: "2 MP"},
    {v: 4.0,  l: "4 MP"},    {v: 8.0, l: "8 MP"},
];

const RATIOS = [
    {label:"3:1",  w:3,  h:1 }, {label:"2:1",  w:2,  h:1 },
    {label:"16:9", w:16, h:9 }, {label:"4:3",  w:4,  h:3 },
    {label:"3:2",  w:3,  h:2 }, {label:"5:4",  w:5,  h:4 },
    {label:"7:5",  w:7,  h:5 }, {label:"1:1",  w:1,  h:1 },
    {label:"5:7",  w:5,  h:7 }, {label:"4:5",  w:4,  h:5 },
    {label:"3:4",  w:3,  h:4 }, {label:"2:3",  w:2,  h:3 },
    {label:"9:16", w:9,  h:16}, {label:"1:2",  w:1,  h:2 },
    {label:"1:3",  w:1,  h:3 },
];

function findClosestRatio(cw, ch) {
    if (!ch) return RATIOS[7];
    const aspect = cw / ch;
    let best = RATIOS[7], bestDiff = Infinity;
    for (const r of RATIOS) {
        const diff = Math.abs(r.w / r.h - aspect);
        if (diff < bestDiff) { best = r; bestDiff = diff; }
    }
    return best;
}

// ── Widget helpers ───────────────────────────────────────────────────────────
const _gw  = (n,k)   => n.widgets?.find(w => w.name === k);
const gval = (n,k,d) => _gw(n,k)?.value ?? d;
const sval = (n,k,v) => { const w = _gw(n,k); if (w) w.value = v; };
const getComp = n => { try { return JSON.parse(gval(n,W_COMP,"")||"null"); } catch { return null; } };
const setComp = (n,d) => sval(n, W_COMP, JSON.stringify(d));

// ── Defaults ────────────────────────────────────────────────────────────────
function defaultComp() {
    const dw = 640, dh = 640;
    const mp = 1.0;
    const {w: cw, h: ch} = dimsFromAspectMP(dw, dh, mp);
    return {
        display_w: dw, display_h: dh,
        canvas_w:  cw, canvas_h:  ch,
        megapixels: mp,
        background: {
            type: "solid",
            color1: [40, 40, 48], color2: [20, 20, 28],
            angle: 0.0, scale: 1.0,
        },
        slots: Array.from({length: MAX_SLOTS}, (_, i) => ({
            id: `s${i}`, label: `Layer ${i+1}`,
            is_bg: false, visible: true, z_order: i,
            x: 0.5, y: 0.5,
            scale: 0.5, scale_x: 1.0, scale_y: 1.0,
            rotation: 0.0, flip_h: false, flip_v: false,
            opacity: 0.5, blend_mode: "normal",
            brightness: 0.0, contrast: 0.0, saturation: 0.0,
            shadow: {
                enabled: false, angle: 135.0, distance: 8.0,
                blur: 12.0, opacity: 0.6, color: [0, 0, 0],
            },
            src_w: 0, src_h: 0,
        })),
    };
}

function dimsFromAspectMP(dw, dh, mp) {
    const aspect = dw / dh;
    const totalPx = mp * 1_000_000;
    let w = Math.sqrt(totalPx * aspect);
    let h = w / aspect;
    w = Math.max(64, Math.round(w / 8) * 8);
    h = Math.max(64, Math.round(h / 8) * 8);
    return {w, h};
}

function ensureComp(node) {
    let cd = getComp(node);
    if (!cd || !cd.slots || cd.slots.length !== MAX_SLOTS) {
        cd = defaultComp();
        setComp(node, cd);
    }
    if (!cd.background) cd.background = defaultComp().background;
    return cd;
}

function getDispDims(node, availW) {
    const cd   = getComp(node);
    const refW = cd?.display_w || availW;
    const refH = cd?.display_h || availW;
    const scale = Math.min(availW / refW, availW / refH);
    return {
        w: Math.max(Math.round(refW * scale), MIN_COMP),
        h: Math.max(Math.round(refH * scale), MIN_COMP),
    };
}

function setDispDims(node, dw, dh) {
    const cd = ensureComp(node);
    cd.display_w = Math.round(dw);
    cd.display_h = Math.round(dh);
    const {w, h} = dimsFromAspectMP(dw, dh, cd.megapixels ?? 1.0);
    cd.canvas_w = w; cd.canvas_h = h;
    setComp(node, cd);
}

function setMegapixels(node, mp) {
    const cd = ensureComp(node);
    cd.megapixels = mp;
    const {w, h} = dimsFromAspectMP(cd.display_w, cd.display_h, mp);
    cd.canvas_w = w; cd.canvas_h = h;
    setComp(node, cd);
}

// ── Geometry helpers ────────────────────────────────────────────────────────
const lerp2 = (a, b) => [(a[0]+b[0])/2, (a[1]+b[1])/2];
const dist2 = (a, b) => Math.hypot(b[0]-a[0], b[1]-a[1]);
const dot2  = (a, b) => a[0]*b[0] + a[1]*b[1];

function getSlotTransform(slot, ca) {
    const src_w = slot.src_w || 1, src_h = slot.src_h || 1;
    const max_dim = Math.max(src_w, src_h);
    const base_px = (slot.scale || 0.5) * ca.w / max_dim;
    const sx = base_px * (slot.scale_x || 1.0) * (slot.flip_h ? -1 : 1);
    const sy = base_px * (slot.scale_y || 1.0) * (slot.flip_v ? -1 : 1);
    const rad = (slot.rotation || 0) * Math.PI / 180;
    const cx = (slot.x ?? 0.5) * ca.w + ca.x;
    const cy = (slot.y ?? 0.5) * ca.h + ca.y;
    const cos_r = Math.cos(rad), sin_r = Math.sin(rad);
    const hw = src_w / 2 * Math.abs(sx), hh = src_h / 2 * Math.abs(sy);
    function tr(lx, ly) {
        return [cx + lx * cos_r - ly * sin_r, cy + lx * sin_r + ly * cos_r];
    }
    const tl = tr(-hw, -hh), tr_ = tr(hw, -hh), br = tr(hw, hh), bl = tr(-hw, hh);
    const top_mid = lerp2(tl, tr_);
    const rot_hnd = [
        top_mid[0] - sin_r * ROT_DIST,
        top_mid[1] + cos_r * ROT_DIST,
    ];
    return {
        cx, cy, sx, sy, rad, cos_r, sin_r, hw, hh,
        tl, tr: tr_, br, bl, top: top_mid,
        right:  lerp2(tr_, br), bottom: lerp2(br, bl),
        left:   lerp2(bl, tl),  rot_hnd,
        src_w, src_h,
    };
}

// ── Image loading ──────────────────────────────────────────────────────────
function loadSlotImage(node, slotIdx) {
    const nodeId = String(node.id);
    const t = Date.now();
    // masked thumb (used for non-BG layers)
    const url = `/eric_composer_studio/imgcomp_slot/${nodeId}/${slotIdx}?t=${t}`;
    const img = new Image();
    img.onload = () => {
        node._icImages[slotIdx] = img;
        const cd = ensureComp(node);
        if (cd.slots[slotIdx]) {
            const slot = cd.slots[slotIdx];
            const wasFresh = slot.src_w === 0;
            slot.src_w = img.naturalWidth;
            slot.src_h = img.naturalHeight;
            if (wasFresh && img.naturalWidth > 0) {
                slot.scale   = 0.5;
                slot.opacity = 0.5;
            }
            setComp(node, cd);
        }
        node._icRedraw?.();
    };
    img.onerror = () => { node._icImages[slotIdx] = null; };
    img.src = url;

    // raw thumb (mask ignored) — used when this slot is set as BG
    const urlRaw = `/eric_composer_studio/imgcomp_slot_raw/${nodeId}/${slotIdx}?t=${t}`;
    const imgRaw = new Image();
    imgRaw.onload = () => {
        node._icImagesRaw[slotIdx] = imgRaw;
        node._icRedraw?.();
    };
    imgRaw.onerror = () => { node._icImagesRaw[slotIdx] = null; };
    imgRaw.src = urlRaw;
}

function loadAllImages(node) {
    const nodeId = String(node.id);
    fetch(`/eric_composer_studio/imgcomp_composition/${nodeId}`)
        .then(r => r.json())
        .then(data => {
            const avail = data.slots_available || [];
            let any = false;
            for (let i = 0; i < MAX_SLOTS; i++) {
                if (avail[i]) { loadSlotImage(node, i); any = true; }
                else { node._icImages[i] = null; node._icImagesRaw[i] = null; }
            }
            if (!any) node._icRedraw?.();
        })
        .catch(() => {});
}

// ── Drawing utilities ───────────────────────────────────────────────────────
function drawSliderBar(ctx, x, y, w, h, val, trackCol, fillCol) {
    ctx.fillStyle = trackCol || "#111"; ctx.fillRect(x, y, w, h);
    ctx.strokeStyle = "#2A3A4A"; ctx.lineWidth = 0.5; ctx.strokeRect(x, y, w, h);
    ctx.fillStyle = fillCol;
    ctx.fillRect(x, y, Math.max(0, Math.min(w, w * Math.max(0, Math.min(1, val)))), h);
}

function drawSmallBtn(ctx, x, y, w, h, label, col, active) {
    ctx.fillStyle = active ? "#1A2A3A" : "#121218";
    ctx.strokeStyle = col; ctx.lineWidth = active ? 1.5 : 1;
    ctx.fillRect(x, y, w, h); ctx.strokeRect(x, y, w, h);
    ctx.fillStyle = col; ctx.font = `bold ${Math.min(h-1, 11)}px monospace`;
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText(label, x + w/2, y + h/2);
}

function drawRndRect(ctx, x, y, w, h, r) {
    if (ctx.roundRect) ctx.roundRect(x, y, w, h, r);
    else ctx.rect(x, y, w, h);
}

function colorToHex(arr) {
    const c = (v) => Math.max(0, Math.min(255, Math.round(v))).toString(16).padStart(2, "0");
    return `#${c(arr[0])}${c(arr[1])}${c(arr[2])}`;
}
function hexToColor(hex) {
    const m = /^#?([0-9a-f]{6})$/i.exec(hex);
    if (!m) return [0, 0, 0];
    const n = parseInt(m[1], 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

// ── Background rendering (preview only) ─────────────────────────────────────
// We render the BG into an offscreen canvas so Canvas2D blend modes work correctly
// against a real image layer below. This is a visual-only approximation of the
// Python procedural BG; full accuracy comes from the rendered node output.
function renderBgPreview(bg, w, h, off) {
    off.width = w; off.height = h;
    const c = off.getContext("2d");
    c.clearRect(0, 0, w, h);
    const col1 = colorToHex(bg.color1 || [40,40,48]);
    const col2 = colorToHex(bg.color2 || [20,20,28]);
    const type = bg.type || "solid";
    const scale = bg.scale || 1.0;

    if (type === "solid") {
        c.fillStyle = col1;
        c.fillRect(0, 0, w, h);
        return;
    }
    if (type === "gradient_lin") {
        const rad = (bg.angle || 0) * Math.PI / 180;
        const dx = Math.cos(rad), dy = Math.sin(rad);
        const cx = w/2, cy = h/2;
        const len = Math.abs(dx) * w/2 + Math.abs(dy) * h/2;
        const g = c.createLinearGradient(cx - dx*len, cy - dy*len, cx + dx*len, cy + dy*len);
        g.addColorStop(0, col1); g.addColorStop(1, col2);
        c.fillStyle = g; c.fillRect(0, 0, w, h); return;
    }
    if (type === "gradient_rad") {
        const r = Math.max(w, h) * 0.5 * Math.max(scale, 0.1);
        const g = c.createRadialGradient(w/2, h/2, 0, w/2, h/2, r);
        g.addColorStop(0, col1); g.addColorStop(1, col2);
        c.fillStyle = g; c.fillRect(0, 0, w, h); return;
    }
    if (type === "checker") {
        const sz = Math.max(2, Math.round(32 * scale));
        c.fillStyle = col1; c.fillRect(0, 0, w, h);
        c.fillStyle = col2;
        for (let y = 0; y < h; y += sz)
            for (let x = ((y / sz) & 1) * sz; x < w; x += sz * 2)
                c.fillRect(x, y, sz, sz);
        return;
    }
    if (type === "dots") {
        const p = Math.max(6, Math.round(24 * scale));
        c.fillStyle = col1; c.fillRect(0, 0, w, h);
        c.fillStyle = col2;
        const r = Math.max(1.5, p * 0.25);
        for (let y = p/2; y < h; y += p)
            for (let x = p/2; x < w; x += p) {
                c.beginPath(); c.arc(x, y, r, 0, Math.PI * 2); c.fill();
            }
        return;
    }
    // paper/canvas/perlin — approximate with tiled random noise
    c.fillStyle = col1; c.fillRect(0, 0, w, h);
    const img = c.getImageData(0, 0, w, h);
    const d = img.data;
    const c1 = hexToColor(col1), c2 = hexToColor(col2);
    const period = type === "canvas" ? Math.max(2, Math.round(4 * scale)) : 0;
    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            const i = (y * w + x) * 4;
            let t = Math.random() * 0.4;
            if (period > 0) {
                const weave = 0.5 * (Math.sin(x * 2*Math.PI/period) + Math.sin(y * 2*Math.PI/period)) * 0.5 + 0.5;
                t = 0.4 * weave + 0.6 * t;
            }
            d[i    ] = c1[0] * (1-t) + c2[0] * t;
            d[i + 1] = c1[1] * (1-t) + c2[1] * t;
            d[i + 2] = c1[2] * (1-t) + c2[2] * t;
            d[i + 3] = 255;
        }
    }
    c.putImageData(img, 0, 0);
}

// ── Main draw ───────────────────────────────────────────────────────────────
function drawAll(ctx, node, W) {
    const cd = ensureComp(node);
    const st = node._icst;
    const winW = W - 2 * MARGIN;
    const dims = getDispDims(node, winW);
    const cw = dims.w, ch = dims.h;
    const listY = W;
    const panelH = W + LIST_H;
    st._listY = listY;

    const ca = {
        x: MARGIN + Math.round((winW - cw) / 2),
        y: MARGIN + Math.round((winW - ch) / 2),
        w: cw, h: ch,
    };

    // Panel background
    ctx.fillStyle = "#06060F"; ctx.fillRect(0, 0, W, panelH);
    ctx.strokeStyle = "#1E2A38"; ctx.lineWidth = 1;
    ctx.strokeRect(1, 1, W-2, panelH-2);

    // Canvas window
    ctx.fillStyle = "#030308";
    ctx.fillRect(MARGIN, MARGIN, winW, winW);
    ctx.strokeStyle = "#1A2430"; ctx.lineWidth = 1;
    ctx.strokeRect(MARGIN, MARGIN, winW, winW);

    // Composition area: procedural BG preview
    if (!st._bgOff) st._bgOff = document.createElement("canvas");
    renderBgPreview(cd.background, ca.w, ca.h, st._bgOff);
    ctx.drawImage(st._bgOff, ca.x, ca.y);

    ctx.strokeStyle = "#334455"; ctx.lineWidth = 1;
    ctx.strokeRect(ca.x, ca.y, ca.w, ca.h);

    // Grid
    ctx.save(); ctx.strokeStyle = "#141E2A55"; ctx.lineWidth = 0.5;
    for (let i = 1; i < 3; i++) {
        const gx = ca.x + ca.w * i / 3, gy = ca.y + ca.h * i / 3;
        ctx.beginPath(); ctx.moveTo(gx, ca.y); ctx.lineTo(gx, ca.y + ca.h); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(ca.x, gy); ctx.lineTo(ca.x + ca.w, gy); ctx.stroke();
    }
    ctx.restore();

    // Output dimensions label
    ctx.fillStyle = "#88AACC"; ctx.font = "10px monospace";
    ctx.textAlign = "center"; ctx.textBaseline = "bottom";
    ctx.fillText(`${cd.canvas_w} × ${cd.canvas_h}  (${((cd.canvas_w*cd.canvas_h)/1e6).toFixed(2)} MP)`,
                 ca.x + ca.w/2, ca.y + ca.h - 2);

    // Any layers?
    const hasAny = cd.slots.some((s, i) => s.src_w > 0 && node._icImages[i]);
    if (!hasAny) {
        ctx.fillStyle = "#445566"; ctx.font = "12px sans-serif";
        ctx.textAlign = "center"; ctx.textBaseline = "middle";
        ctx.fillText("Connect images (and optional masks) and queue workflow",
                     ca.x + ca.w/2, ca.y + ca.h/2 - 8);
        ctx.font = "10px monospace"; ctx.fillStyle = "#2A3A4A";
        ctx.fillText("image_1..6 + optional mask_1..6", ca.x + ca.w/2, ca.y + ca.h/2 + 10);
    }

    // Clip layer rendering to canvas area
    ctx.save();
    ctx.beginPath(); ctx.rect(ca.x, ca.y, ca.w, ca.h); ctx.clip();

    // Find BG slot (drawn first, ignores mask)
    let bgSlotIdx = -1;
    for (let i = 0; i < MAX_SLOTS; i++) {
        if (cd.slots[i].is_bg && node._icImages[i]) { bgSlotIdx = i; break; }
    }

    // Build draw order: BG first, then other slots in z_order ascending.
    const others = cd.slots.map((s, i) => ({s, i}))
        .filter(({i}) => i !== bgSlotIdx)
        .sort((a, b) => (a.s.z_order||0) - (b.s.z_order||0));
    const ordered = bgSlotIdx >= 0
        ? [{s: cd.slots[bgSlotIdx], i: bgSlotIdx}, ...others]
        : others;

    for (const {s: slot, i} of ordered) {
        if (!slot.visible || !slot.src_w) continue;
        const isBg = (i === bgSlotIdx);
        // For BG mode, prefer the raw (mask-ignored) thumb so users see the
        // full input frame to position other layers against.
        const img = (isBg && node._icImagesRaw[i]) ? node._icImagesRaw[i] : node._icImages[i];
        if (!img) continue;
        const tf = getSlotTransform(slot, ca);

        // Drop shadow
        const sh = slot.shadow || {};
        if (sh.enabled) {
            const angleRad = (sh.angle ?? 135) * Math.PI / 180;
            const dist = sh.distance ?? 8;
            const dx = Math.cos(angleRad) * dist;
            const dy = -Math.sin(angleRad) * dist;
            ctx.save();
            ctx.globalAlpha = (sh.opacity ?? 0.6) * (slot.opacity ?? 1);
            ctx.filter = `blur(${(sh.blur ?? 12).toFixed(1)}px)`;
            ctx.translate(tf.cx + dx, tf.cy + dy);
            ctx.rotate(tf.rad);
            ctx.scale(tf.sx, tf.sy);
            // Tint: draw image silhouette as shadow colour by using a temp canvas
            if (!st._shadowOff) st._shadowOff = document.createElement("canvas");
            const off = st._shadowOff;
            off.width = img.naturalWidth; off.height = img.naturalHeight;
            const oc = off.getContext("2d");
            oc.clearRect(0, 0, off.width, off.height);
            oc.drawImage(img, 0, 0);
            // For BG mode the mask is ignored, so use opaque rectangle.
            if (isBg) {
                oc.globalCompositeOperation = "source-over";
            } else {
                oc.globalCompositeOperation = "source-in";
            }
            oc.fillStyle = colorToHex(sh.color || [0, 0, 0]);
            oc.fillRect(0, 0, off.width, off.height);
            ctx.drawImage(off, -tf.src_w/2, -tf.src_h/2, tf.src_w, tf.src_h);
            ctx.restore();
        }

        ctx.save();
        ctx.globalCompositeOperation = BLEND_CANVAS[slot.blend_mode] || "source-over";
        ctx.globalAlpha = Math.max(0, Math.min(1, slot.opacity ?? 1));
        const bcs = slot.brightness || slot.contrast || slot.saturation;
        if (bcs) {
            const b = 1 + (slot.brightness || 0);
            const c = 1 + (slot.contrast   || 0);
            const s = 1 + (slot.saturation || 0);
            ctx.filter = `brightness(${b}) contrast(${c}) saturate(${s})`;
        }
        ctx.translate(tf.cx, tf.cy);
        ctx.rotate(tf.rad);
        ctx.scale(tf.sx, tf.sy);
        // BG mode draws full image rectangle (mask ignored).
        // Non-BG draws the image with its alpha (mask) applied.
        ctx.drawImage(img, -tf.src_w/2, -tf.src_h/2, tf.src_w, tf.src_h);
        ctx.restore();
    }
    ctx.restore(); // end clip

    // Slot handles (BG slot included so user can size/position it)
    const selIdx = st._selIdx;
    cd.slots.forEach((slot, i) => {
        if (!slot.visible || !slot.src_w) return;
        const tf = getSlotTransform(slot, ca);
        const col = SLOT_COLORS[i] || "#888";
        const sel = (i === selIdx);

        ctx.save();
        ctx.strokeStyle = sel ? "#FFF" : col;
        ctx.fillStyle = sel ? "#FFFFFF44" : col + "44";
        ctx.lineWidth = sel ? 2 : 1.5;
        ctx.beginPath(); ctx.arc(tf.cx, tf.cy, MOV_R, 0, Math.PI * 2);
        ctx.fill(); ctx.stroke();
        // draw layer index number inside
        ctx.fillStyle = "#FFF"; ctx.font = "bold 9px monospace";
        ctx.textAlign = "center"; ctx.textBaseline = "middle";
        ctx.fillText(`${i+1}`, tf.cx, tf.cy);
        ctx.restore();

        if (sel) {
            ctx.save();
            ctx.strokeStyle = col + "AA"; ctx.lineWidth = 1;
            ctx.setLineDash([3, 3]); ctx.beginPath();
            ctx.moveTo(tf.tl[0], tf.tl[1]); ctx.lineTo(tf.tr[0], tf.tr[1]);
            ctx.lineTo(tf.br[0], tf.br[1]); ctx.lineTo(tf.bl[0], tf.bl[1]);
            ctx.closePath(); ctx.stroke();
            ctx.setLineDash([]); ctx.restore();

            _drawEdgeHnd(ctx, tf.right,  col, "X");
            _drawEdgeHnd(ctx, tf.left,   col, "X");
            _drawEdgeHnd(ctx, tf.top,    col, "Y");
            _drawEdgeHnd(ctx, tf.bottom, col, "Y");

            ctx.save();
            ctx.fillStyle = "#1A2A3A"; ctx.strokeStyle = "#88AAFF"; ctx.lineWidth = 1.5;
            ctx.fillRect(tf.br[0]-EDGE_R, tf.br[1]-EDGE_R, EDGE_R*2, EDGE_R*2);
            ctx.strokeRect(tf.br[0]-EDGE_R, tf.br[1]-EDGE_R, EDGE_R*2, EDGE_R*2);
            ctx.restore();

            ctx.save();
            ctx.strokeStyle = col + "99"; ctx.lineWidth = 1;
            ctx.setLineDash([2, 2]);
            ctx.beginPath(); ctx.moveTo(tf.top[0], tf.top[1]);
            ctx.lineTo(tf.rot_hnd[0], tf.rot_hnd[1]); ctx.stroke();
            ctx.setLineDash([]);
            ctx.fillStyle = st._dragMode === "rotate" && st._dragData?.slotIdx === i
                            ? "#FFCC44CC" : "#2A1A0ACC";
            ctx.strokeStyle = "#FFCC44"; ctx.lineWidth = 1.5;
            ctx.beginPath(); ctx.arc(tf.rot_hnd[0], tf.rot_hnd[1], ROT_R, 0, Math.PI * 2);
            ctx.fill(); ctx.stroke();
            ctx.fillStyle = "#FFCC44"; ctx.font = `bold ${ROT_R + 3}px sans-serif`;
            ctx.textAlign = "center"; ctx.textBaseline = "middle";
            ctx.fillText("↻", tf.rot_hnd[0], tf.rot_hnd[1] + 1);
            ctx.restore();
        }
    });

    // Resize corner
    const rx = ca.x + ca.w, ry = ca.y + ca.h;
    const isRes = st._dragMode === "resize";
    ctx.save();
    ctx.fillStyle = isRes ? "#5599FFCC" : "#33445588";
    ctx.beginPath(); ctx.moveTo(rx - LG_RESIZE*2, ry); ctx.lineTo(rx, ry-LG_RESIZE*2); ctx.lineTo(rx, ry);
    ctx.closePath(); ctx.fill();
    ctx.fillStyle = isRes ? "#88BBFF" : "#556677";
    ctx.beginPath(); ctx.arc(rx, ry, 4, 0, Math.PI*2); ctx.fill();
    ctx.restore();
    const cr = findClosestRatio(cw, ch);
    ctx.save(); ctx.font = isRes ? "bold 11px monospace" : "11px monospace";
    ctx.fillStyle = isRes ? "#88CCFF" : "#446688";
    ctx.textAlign = "left"; ctx.textBaseline = "top";
    ctx.fillText(`≈ ${cr.label}`, rx+4, ry-20);
    ctx.fillStyle = "#334455"; ctx.font = "9px monospace";
    ctx.fillText(`${Math.round(cw)}×${Math.round(ch)}px`, rx+4, ry-8);
    ctx.restore();

    st._caArea = ca;

    // Layer list
    _drawList(ctx, node, W, cd, selIdx);
}

function _drawEdgeHnd(ctx, pos, col, lbl) {
    ctx.save();
    ctx.fillStyle = "#1A2A3A"; ctx.strokeStyle = col; ctx.lineWidth = 1.5;
    ctx.fillRect(pos[0]-EDGE_R, pos[1]-EDGE_R, EDGE_R*2, EDGE_R*2);
    ctx.strokeRect(pos[0]-EDGE_R, pos[1]-EDGE_R, EDGE_R*2, EDGE_R*2);
    ctx.fillStyle = col; ctx.font = "bold 7px monospace";
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText(lbl, pos[0], pos[1]);
    ctx.restore();
}

function _drawList(ctx, node, W, cd, selIdx) {
    const st = node._icst;
    const lx = MARGIN, lw = W - 2*MARGIN;
    const LIST_Y = st._listY;

    ctx.fillStyle = "#0D1A2A";
    ctx.fillRect(lx, LIST_Y, lw, LIST_H);
    ctx.strokeStyle = "#1E2A38"; ctx.lineWidth = 1;
    ctx.strokeRect(lx, LIST_Y, lw, LIST_H);

    ctx.fillStyle = "#2A4A6A"; ctx.font = "10px monospace";
    ctx.textAlign = "left"; ctx.textBaseline = "middle";
    ctx.fillText("Layers — click to select · right-click for menu (z-order, reset, etc.)",
                 lx+6, LIST_Y + LIST_HDR_H/2);

    st._zones = {};

    cd.slots.forEach((slot, i) => {
        const rowY = LIST_Y + LIST_HDR_H + i * SLOT_ROW_H + 2;
        const rowH = SLOT_ROW_H - 2;
        const col  = SLOT_COLORS[i];
        const sel  = (i === selIdx);
        const hasD = slot.src_w > 0;

        ctx.fillStyle = sel ? "#0A1828" : "#080F18";
        ctx.fillRect(lx+2, rowY, lw-4, rowH);
        if (sel) {
            ctx.strokeStyle = col + "99"; ctx.lineWidth = 1;
            ctx.strokeRect(lx+2, rowY, lw-4, rowH);
        }
        st._zones[`row_${i}`] = {x:lx+2, y:rowY, w:lw-4, h:rowH};

        const midY = rowY + rowH/2;
        const rX = lx + 6;

        // Color dot
        ctx.fillStyle = hasD ? col : "#334455";
        ctx.strokeStyle = "#000"; ctx.lineWidth = 0.5;
        ctx.beginPath(); ctx.arc(rX+5, midY, 5, 0, Math.PI*2); ctx.fill(); ctx.stroke();

        // Label
        ctx.fillStyle = hasD ? "#99BBCC" : "#445566";
        ctx.font = "bold 11px monospace"; ctx.textAlign = "left"; ctx.textBaseline = "middle";
        ctx.fillText(slot.label, rX+16, midY);

        // Dims
        ctx.fillStyle = "#446688"; ctx.font = "10px monospace";
        const dimTxt = hasD ? `${slot.src_w}×${slot.src_h}` : "no input";
        ctx.fillText(dimTxt, rX+90, midY);

        // Blend mode + flip indicators
        const blendTxt = (slot.blend_mode || "normal").replace("_"," ");
        const flipTxt  = (slot.flip_h ? " ↔" : "") + (slot.flip_v ? " ↕" : "");
        const shadowTxt = slot.shadow?.enabled ? " · shadow" : "";
        ctx.fillStyle = "#88AABB"; ctx.font = "10px monospace";
        ctx.fillText(`${blendTxt}${flipTxt}${shadowTxt}`, rX+185, midY);

        // Right side: BG  · eye · opacity slider · opacity %
        // Layout from right edge inward.
        const btnH = 16, btnY = rowY + (rowH - btnH)/2;
        const rightEdge = lx + lw - 8;

        // Opacity %
        const pctTxt = `${Math.round((slot.opacity ?? 1)*100)}%`;
        ctx.fillStyle = "#88AABB"; ctx.font = "bold 10px monospace";
        ctx.textAlign = "right"; ctx.textBaseline = "middle";
        ctx.fillText(pctTxt, rightEdge, midY);
        const pctW = 36;

        // Opacity slider
        const opW = 70;
        const opX = rightEdge - pctW - opW;
        drawSliderBar(ctx, opX, midY-4, opW, 8, slot.opacity ?? 1.0, "#111", "#446688");
        st._zones[`op_${i}`] = {x:opX, y:midY-5, w:opW, h:10, type:"op"};

        // Eye
        const eyeW = 22;
        const eyeX = opX - eyeW - 6;
        drawSmallBtn(ctx, eyeX, btnY, eyeW, btnH,
                     slot.visible ? "👁" : "◌", "#668899", slot.visible);
        st._zones[`eye_${i}`] = {x:eyeX, y:btnY, w:eyeW, h:btnH};

        // BG
        const bgW = 26;
        const bgX = eyeX - bgW - 4;
        drawSmallBtn(ctx, bgX, btnY, bgW, btnH, "BG",
                     slot.is_bg ? "#66DD66" : "#445566", slot.is_bg);
        st._zones[`bg_${i}`] = {x:bgX, y:btnY, w:bgW, h:btnH};
    });
}

// ── Hit testing ─────────────────────────────────────────────────────────────
function _hr(mx, my, z) { return mx>=z.x && mx<=z.x+z.w && my>=z.y && my<=z.y+z.h; }

function hitTest(mx, my, node) {
    const cd = ensureComp(node);
    const st = node._icst;
    const ca = st._caArea;

    // Check button/widget zones BEFORE row zones so clicking a button inside
    // a row doesn't get swallowed by the row's "select layer" zone.
    const zones = st._zones || {};
    const entries = Object.entries(zones);
    const buttons = entries.filter(([k]) => !k.startsWith("row_"));
    const rows    = entries.filter(([k]) =>  k.startsWith("row_"));
    for (const [key, zone] of [...buttons, ...rows]) {
        if (_hr(mx, my, zone)) return {type:"zone", key, zone};
    }
    if (!ca) return null;

    const rx = ca.x + ca.w, ry = ca.y + ca.h;
    if (Math.hypot(mx-rx, my-ry) < LG_RESIZE + 4) return {type:"resize"};

    const selIdx = st._selIdx;

    const checkOrder = selIdx !== null
        ? [selIdx, ...cd.slots.map((_,i)=>i).filter(i=>i!==selIdx)]
        : cd.slots.map((_,i)=>i);

    for (const i of checkOrder) {
        const slot = cd.slots[i];
        if (!slot.visible || !slot.src_w) continue;
        const tf = getSlotTransform(slot, ca);

        if (i === selIdx) {
            if (dist2([mx,my], tf.rot_hnd) < ROT_R+4)  return {type:"rotate",  slotIdx:i};
            if (dist2([mx,my], tf.br)     < EDGE_R+4)  return {type:"scale_u", slotIdx:i};
            if (dist2([mx,my], tf.right)  < EDGE_R+3)  return {type:"scale_x", slotIdx:i, axis:[tf.cos_r, tf.sin_r], cx:tf.cx, cy:tf.cy};
            if (dist2([mx,my], tf.left)   < EDGE_R+3)  return {type:"scale_x", slotIdx:i, axis:[tf.cos_r, tf.sin_r], cx:tf.cx, cy:tf.cy};
            if (dist2([mx,my], tf.top)    < EDGE_R+3)  return {type:"scale_y", slotIdx:i, axis:[-tf.sin_r, tf.cos_r], cx:tf.cx, cy:tf.cy};
            if (dist2([mx,my], tf.bottom) < EDGE_R+3)  return {type:"scale_y", slotIdx:i, axis:[-tf.sin_r, tf.cos_r], cx:tf.cx, cy:tf.cy};
        }
        if (dist2([mx,my], [tf.cx, tf.cy]) < MOV_R+3)  return {type:"move", slotIdx:i};
    }
    return null;
}

function getCanvasXY(e, canvas) {
    const rect = canvas.getBoundingClientRect();
    const dpr  = window.devicePixelRatio || 1;
    return [
        (e.clientX - rect.left) * (canvas.width  / dpr / rect.width),
        (e.clientY - rect.top)  * (canvas.height / dpr / rect.height),
    ];
}

// ── Snap helper ─────────────────────────────────────────────────────────────
function snapPos(x, y, shiftKey) {
    if (!shiftKey) return [x, y];
    const snaps = [0.0, 0.25, 0.5, 0.75, 1.0];
    const thresh = 0.025;
    let sx = x, sy = y;
    for (const s of snaps) {
        if (Math.abs(x - s) < thresh) sx = s;
        if (Math.abs(y - s) < thresh) sy = s;
    }
    return [sx, sy];
}

// ── Context menu ────────────────────────────────────────────────────────────
function showContextMenu(e, node, slotIdx) {
    e.preventDefault(); e.stopPropagation();
    const existing = document.querySelector(".eric-ic-ctx");
    if (existing) existing.remove();

    const menu = document.createElement("div");
    menu.className = "eric-ic-ctx";
    menu.style.cssText = [
        "position:fixed", `left:${e.clientX}px`, `top:${e.clientY}px`,
        "background:#12121E", "border:1px solid #2A3A4A", "border-radius:4px",
        "padding:4px 0", "z-index:99999", "font:11px monospace",
        "box-shadow:0 4px 12px #0008", "color:#99BBCC", "min-width:180px",
    ].join(";");

    const item = (label, fn) => {
        const d = document.createElement("div");
        d.textContent = label;
        d.style.cssText = "padding:4px 12px;cursor:pointer;";
        d.onmouseenter = () => d.style.background = "#1A2A3A";
        d.onmouseleave = () => d.style.background = "";
        d.onclick = () => { fn(); menu.remove(); };
        menu.appendChild(d);
    };
    const sep = () => {
        const d = document.createElement("div");
        d.style.cssText = "height:1px;background:#2A3A4A;margin:3px 0;";
        menu.appendChild(d);
    };

    const cd = ensureComp(node);
    const slot = cd.slots[slotIdx];

    item("Bring to front", () => {
        const maxZ = Math.max(...cd.slots.map(s => s.z_order || 0));
        slot.z_order = maxZ + 1;
        setComp(node, cd); node._icRedraw?.();
    });
    item("Send to back", () => {
        const minZ = Math.min(...cd.slots.map(s => s.z_order || 0));
        slot.z_order = minZ - 1;
        setComp(node, cd); node._icRedraw?.();
    });
    sep();
    item("Reset transform", () => {
        slot.x = 0.5; slot.y = 0.5;
        slot.scale = 0.5; slot.scale_x = 1.0; slot.scale_y = 1.0;
        slot.rotation = 0.0; slot.flip_h = false; slot.flip_v = false;
        setComp(node, cd); node._icRedraw?.();
    });
    item("Reset blend & opacity", () => {
        slot.opacity = 1.0; slot.blend_mode = "normal";
        setComp(node, cd); node._icRedraw?.();
        node._icSyncDetail?.();
    });
    item("Reset colour adjustments", () => {
        slot.brightness = 0; slot.contrast = 0; slot.saturation = 0;
        setComp(node, cd); node._icRedraw?.();
        node._icSyncDetail?.();
    });
    item("Center on canvas", () => {
        slot.x = 0.5; slot.y = 0.5;
        setComp(node, cd); node._icRedraw?.();
    });
    item("Fit to canvas (cover)", () => {
        slot.scale = 1.0; slot.scale_x = 1.0; slot.scale_y = 1.0;
        slot.x = 0.5; slot.y = 0.5;
        setComp(node, cd); node._icRedraw?.();
    });
    sep();
    item(slot.visible ? "Hide" : "Show", () => {
        slot.visible = !slot.visible;
        setComp(node, cd); node._icRedraw?.();
    });
    item("Toggle as background", () => {
        const was = slot.is_bg;
        cd.slots.forEach(s => s.is_bg = false);
        if (!was) {
            slot.is_bg = true;
            if (slot.src_w && slot.src_h) {
                const max_dim = Math.max(slot.src_w, slot.src_h);
                const cover = Math.max(
                    cd.canvas_w / slot.src_w,
                    cd.canvas_h / slot.src_h,
                );
                slot.scale   = cover * max_dim / cd.canvas_w;
                slot.scale_x = 1.0; slot.scale_y = 1.0;
                slot.x = 0.5; slot.y = 0.5; slot.rotation = 0;
            }
        }
        setComp(node, cd); node._icRedraw?.();
    });

    document.body.appendChild(menu);
    const off = (ev) => {
        if (!menu.contains(ev.target)) { menu.remove(); document.removeEventListener("mousedown", off, true); }
    };
    setTimeout(() => document.addEventListener("mousedown", off, true), 0);
}

// ── Setup ───────────────────────────────────────────────────────────────────
function setup(node) {
    node._icImages = Array(MAX_SLOTS).fill(null);
    node._icImagesRaw = Array(MAX_SLOTS).fill(null);
    node._icst = {
        _selIdx: 0, _dragMode: null, _dragData: null,
        _caArea: null, _zones: {}, _listY: null,
        _bgOff: null, _shadowOff: null,
    };

    const wrapper = document.createElement("div");
    wrapper.style.cssText = "width:100%;overflow:hidden;user-select:none;pointer-events:all;display:flex;flex-direction:column;";

    // ── Top bar (ratio + MP + reset/reload) ──────────────────────────────────
    const topBar = document.createElement("div");
    topBar.style.cssText = [
        "display:flex", "align-items:center", "gap:6px",
        "padding:4px 10px", "background:#0A0A18", "border-bottom:1px solid #1E2A38",
        `height:${RATIO_BAR_H}px`, "box-sizing:border-box", "flex-shrink:0",
    ].join(";");
    const lbl = (t) => { const s = document.createElement("span"); s.textContent = t;
        s.style.cssText = "color:#446688;font:11px monospace;white-space:nowrap;"; return s; };
    const mkSelect = (opts, valFn) => {
        const s = document.createElement("select");
        s.style.cssText = "background:#12121E;color:#88AACC;border:1px solid #2A3A4A;border-radius:3px;font:11px monospace;padding:2px 4px;cursor:pointer;pointer-events:all;";
        opts.forEach(o => { const op = document.createElement("option");
            op.value = typeof o === "string" ? o : o.v;
            op.textContent = typeof o === "string" ? o : o.l; s.appendChild(op); });
        return s;
    };

    topBar.appendChild(lbl("Snap:"));
    const snapSelect = mkSelect([{v:"", l:"- free -"}, ...RATIOS.map(r => ({v:`${r.w}:${r.h}`, l:r.label}))]);
    snapSelect.addEventListener("change", () => {
        const v = snapSelect.value; if (!v) return;
        const [rw, rh] = v.split(":").map(Number);
        const dims = getDispDims(node, canvas.clientWidth - 2*MARGIN);
        setDispDims(node, dims.w, Math.round(dims.w * rh / rw));
        redraw();
    });
    topBar.appendChild(snapSelect);

    topBar.appendChild(lbl("MP:"));
    const mpSelect = mkSelect(MP_PRESETS);
    mpSelect.addEventListener("change", () => {
        setMegapixels(node, parseFloat(mpSelect.value));
        redraw();
    });
    topBar.appendChild(mpSelect);

    const resetBtn = document.createElement("button");
    resetBtn.textContent = "↺ Reset all";
    resetBtn.title = "Reset all layer transforms and blend settings";
    resetBtn.style.cssText = "margin-left:auto;background:#0A1A0A;color:#668844;border:1px solid #334422;border-radius:3px;font:11px monospace;padding:2px 8px;cursor:pointer;pointer-events:all;";
    resetBtn.addEventListener("click", (e) => {
        e.preventDefault(); e.stopPropagation();
        const cd = ensureComp(node);
        const fresh = defaultComp();
        cd.slots.forEach((s, i) => {
            const f = fresh.slots[i];
            Object.assign(s, {
                x:f.x, y:f.y, scale:f.scale, scale_x:f.scale_x, scale_y:f.scale_y,
                rotation:f.rotation, flip_h:f.flip_h, flip_v:f.flip_v,
                opacity:f.opacity, blend_mode:f.blend_mode,
                brightness:f.brightness, contrast:f.contrast, saturation:f.saturation,
                shadow: {...f.shadow},
            });
        });
        setComp(node, cd); redraw(); syncDetail();
    });
    topBar.appendChild(resetBtn);

    const reloadBtn = document.createElement("button");
    reloadBtn.textContent = "⟳ Reload";
    reloadBtn.title = "Reload layer thumbnails from server after queuing";
    reloadBtn.style.cssText = "background:#0A1A2A;color:#6699BB;border:1px solid #2A3A4A;border-radius:3px;font:11px monospace;padding:2px 8px;cursor:pointer;pointer-events:all;";
    reloadBtn.addEventListener("click", (e) => {
        e.preventDefault(); e.stopPropagation();
        reloadBtn.textContent = "⟳ Loading...";
        reloadBtn.disabled = true;
        loadAllImages(node);
        setTimeout(() => { reloadBtn.textContent = "⟳ Reload"; reloadBtn.disabled = false; }, 1500);
    });
    topBar.appendChild(reloadBtn);

    wrapper.appendChild(topBar);

    // ── Canvas ───────────────────────────────────────────────────────────────
    const canvas = document.createElement("canvas");
    canvas.style.cssText = "display:block;width:100%;pointer-events:all;touch-action:none;cursor:default;flex-shrink:0;";
    wrapper.appendChild(canvas);

    // ── Detail row (selected layer) ──────────────────────────────────────────
    const detailBar = document.createElement("div");
    detailBar.style.cssText = [
        "display:flex", "align-items:center", "gap:6px", "flex-wrap:wrap",
        "padding:4px 10px", "background:#0A0A18", "border-top:1px solid #1E2A38",
        `min-height:${DETAIL_BAR_H}px`, "box-sizing:border-box", "flex-shrink:0",
    ].join(";");

    const layerLabel = document.createElement("span");
    layerLabel.style.cssText = "color:#88AACC;font:bold 11px monospace;min-width:60px;";
    detailBar.appendChild(layerLabel);

    detailBar.appendChild(lbl("Blend:"));
    const blendSelect = mkSelect(BLEND_MODES.map(m => ({v:m, l:m.replace("_"," ")})));
    blendSelect.addEventListener("change", () => {
        const st = node._icst; if (st._selIdx == null) return;
        const cd = ensureComp(node);
        cd.slots[st._selIdx].blend_mode = blendSelect.value;
        setComp(node, cd); redraw();
    });
    detailBar.appendChild(blendSelect);

    const mkNumRange = (min, max, step, val, onChange, width=90) => {
        const inp = document.createElement("input");
        inp.type = "range"; inp.min = min; inp.max = max; inp.step = step; inp.value = val;
        inp.style.cssText = `width:${width}px;pointer-events:all;`;
        inp.addEventListener("input", () => onChange(parseFloat(inp.value)));
        return inp;
    };

    detailBar.appendChild(lbl("Opacity:"));
    const opInput = mkNumRange(0, 1, 0.01, 0.5, (v) => {
        const st = node._icst; if (st._selIdx == null) return;
        const cd = ensureComp(node); cd.slots[st._selIdx].opacity = v;
        setComp(node, cd); redraw();
        opVal.textContent = `${Math.round(v*100)}%`;
    });
    detailBar.appendChild(opInput);
    const opVal = document.createElement("span");
    opVal.style.cssText = "color:#88AACC;font:10px monospace;min-width:34px;";
    detailBar.appendChild(opVal);

    const mkToggle = (label, title, onToggle) => {
        const b = document.createElement("button");
        b.textContent = label; b.title = title;
        b.style.cssText = "background:#12121E;color:#88AACC;border:1px solid #2A3A4A;border-radius:3px;font:11px monospace;padding:2px 8px;cursor:pointer;pointer-events:all;";
        b.dataset.active = "0";
        b.addEventListener("click", (e) => { e.stopPropagation(); onToggle(b); });
        return b;
    };
    const setToggleState = (b, active) => {
        b.dataset.active = active ? "1" : "0";
        b.style.background = active ? "#1A3A1A" : "#12121E";
        b.style.color      = active ? "#66DD66" : "#88AACC";
        b.style.borderColor = active ? "#44AA44" : "#2A3A4A";
    };

    const flipHBtn = mkToggle("↔ H", "Flip horizontally", (b) => {
        const st = node._icst; if (st._selIdx == null) return;
        const cd = ensureComp(node);
        cd.slots[st._selIdx].flip_h = !cd.slots[st._selIdx].flip_h;
        setToggleState(b, cd.slots[st._selIdx].flip_h);
        setComp(node, cd); redraw();
    });
    const flipVBtn = mkToggle("↕ V", "Flip vertically", (b) => {
        const st = node._icst; if (st._selIdx == null) return;
        const cd = ensureComp(node);
        cd.slots[st._selIdx].flip_v = !cd.slots[st._selIdx].flip_v;
        setToggleState(b, cd.slots[st._selIdx].flip_v);
        setComp(node, cd); redraw();
    });
    detailBar.appendChild(flipHBtn); detailBar.appendChild(flipVBtn);

    // B/C/S sliders
    const mkBcsRow = (key, name, col) => {
        const wrap = document.createElement("span");
        wrap.style.cssText = "display:inline-flex;align-items:center;gap:3px;";
        const lb = document.createElement("span");
        lb.textContent = name; lb.style.cssText = `color:${col};font:10px monospace;`;
        wrap.appendChild(lb);
        const inp = mkNumRange(-1, 1, 0.01, 0, (v) => {
            const st = node._icst; if (st._selIdx == null) return;
            const cd = ensureComp(node); cd.slots[st._selIdx][key] = v;
            setComp(node, cd); redraw();
            vl.textContent = v.toFixed(2);
        }, 70);
        wrap.appendChild(inp);
        const vl = document.createElement("span");
        vl.style.cssText = "color:#88AACC;font:9px monospace;min-width:28px;";
        wrap.appendChild(vl);
        return {wrap, inp, vl};
    };
    const bRow = mkBcsRow("brightness", "B", "#DDBB55");
    const cRow = mkBcsRow("contrast",   "C", "#BBDD88");
    const sRow = mkBcsRow("saturation", "S", "#88BBDD");
    detailBar.appendChild(bRow.wrap);
    detailBar.appendChild(cRow.wrap);
    detailBar.appendChild(sRow.wrap);

    wrapper.appendChild(detailBar);

    // ── Shadow row ───────────────────────────────────────────────────────────
    const shadowBar = document.createElement("div");
    shadowBar.style.cssText = [
        "display:flex", "align-items:center", "gap:6px", "flex-wrap:wrap",
        "padding:4px 10px", "background:#0A0A18", "border-top:1px solid #1E2A38",
        `min-height:${SHADOW_BAR_H}px`, "box-sizing:border-box", "flex-shrink:0",
    ].join(";");

    const shEnableBtn = mkToggle("Drop shadow", "Toggle drop shadow", (b) => {
        const st = node._icst; if (st._selIdx == null) return;
        const cd = ensureComp(node);
        const sh = cd.slots[st._selIdx].shadow ||= {};
        sh.enabled = !sh.enabled;
        setToggleState(b, sh.enabled);
        setComp(node, cd); redraw();
    });
    shadowBar.appendChild(shEnableBtn);

    const mkShParam = (name, key, min, max, step, width=70) => {
        const wrap = document.createElement("span");
        wrap.style.cssText = "display:inline-flex;align-items:center;gap:3px;";
        const lb = lbl(name); wrap.appendChild(lb);
        const inp = mkNumRange(min, max, step, min, (v) => {
            const st = node._icst; if (st._selIdx == null) return;
            const cd = ensureComp(node);
            const sh = cd.slots[st._selIdx].shadow ||= {};
            sh[key] = v;
            setComp(node, cd); redraw();
            vl.textContent = v.toFixed(key === "opacity" ? 2 : 0);
        }, width);
        wrap.appendChild(inp);
        const vl = document.createElement("span");
        vl.style.cssText = "color:#88AACC;font:9px monospace;min-width:28px;";
        wrap.appendChild(vl);
        return {wrap, inp, vl};
    };
    const shAngle = mkShParam("Ang",  "angle",    -180, 180, 1);
    const shDist  = mkShParam("Dist", "distance", 0, 80, 1);
    const shBlur  = mkShParam("Blur", "blur",     0, 60, 1);
    const shOp    = mkShParam("Op",   "opacity",  0, 1, 0.01);
    shadowBar.appendChild(shAngle.wrap);
    shadowBar.appendChild(shDist.wrap);
    shadowBar.appendChild(shBlur.wrap);
    shadowBar.appendChild(shOp.wrap);

    shadowBar.appendChild(lbl("Col"));
    const shColor = document.createElement("input");
    shColor.type = "color"; shColor.value = "#000000";
    shColor.style.cssText = "width:28px;height:20px;border:1px solid #2A3A4A;background:#12121E;cursor:pointer;pointer-events:all;";
    shColor.addEventListener("input", () => {
        const st = node._icst; if (st._selIdx == null) return;
        const cd = ensureComp(node);
        const sh = cd.slots[st._selIdx].shadow ||= {};
        sh.color = hexToColor(shColor.value);
        setComp(node, cd); redraw();
    });
    shadowBar.appendChild(shColor);

    wrapper.appendChild(shadowBar);

    // ── Background row ───────────────────────────────────────────────────────
    const bgBar = document.createElement("div");
    bgBar.style.cssText = [
        "display:flex", "align-items:center", "gap:6px", "flex-wrap:wrap",
        "padding:4px 10px", "background:#080814", "border-top:1px solid #1E2A38",
        `min-height:${BG_BAR_H}px`, "box-sizing:border-box", "flex-shrink:0",
    ].join(";");

    bgBar.appendChild(lbl("Background:"));
    const bgSelect = mkSelect(BG_TYPES.map(b => ({v:b.v, l:b.l})));
    bgSelect.addEventListener("change", () => {
        const cd = ensureComp(node); cd.background.type = bgSelect.value;
        setComp(node, cd); redraw();
    });
    bgBar.appendChild(bgSelect);

    bgBar.appendChild(lbl("C1"));
    const bgCol1 = document.createElement("input");
    bgCol1.type = "color"; bgCol1.style.cssText = "width:28px;height:20px;border:1px solid #2A3A4A;background:#12121E;cursor:pointer;pointer-events:all;";
    bgCol1.addEventListener("input", () => {
        const cd = ensureComp(node); cd.background.color1 = hexToColor(bgCol1.value);
        setComp(node, cd); redraw();
    });
    bgBar.appendChild(bgCol1);

    bgBar.appendChild(lbl("C2"));
    const bgCol2 = document.createElement("input");
    bgCol2.type = "color"; bgCol2.style.cssText = "width:28px;height:20px;border:1px solid #2A3A4A;background:#12121E;cursor:pointer;pointer-events:all;";
    bgCol2.addEventListener("input", () => {
        const cd = ensureComp(node); cd.background.color2 = hexToColor(bgCol2.value);
        setComp(node, cd); redraw();
    });
    bgBar.appendChild(bgCol2);

    const mkBgParam = (name, key, min, max, step, width=80) => {
        const wrap = document.createElement("span");
        wrap.style.cssText = "display:inline-flex;align-items:center;gap:3px;";
        wrap.appendChild(lbl(name));
        const inp = mkNumRange(min, max, step, min, (v) => {
            const cd = ensureComp(node); cd.background[key] = v;
            setComp(node, cd); redraw();
            vl.textContent = v.toFixed(key === "scale" ? 2 : 0);
        }, width);
        wrap.appendChild(inp);
        const vl = document.createElement("span");
        vl.style.cssText = "color:#88AACC;font:9px monospace;min-width:28px;";
        wrap.appendChild(vl);
        return {wrap, inp, vl};
    };
    const bgAngle = mkBgParam("Ang", "angle", 0, 360, 1);
    const bgScale = mkBgParam("Sc", "scale", 0.1, 5, 0.01);
    bgBar.appendChild(bgAngle.wrap);
    bgBar.appendChild(bgScale.wrap);

    wrapper.appendChild(bgBar);

    // ── Sync detail UI to selected slot / comp ───────────────────────────────
    function syncDetail() {
        const cd = ensureComp(node);
        const st = node._icst;
        const i = st._selIdx;
        if (i == null || !cd.slots[i]) {
            layerLabel.textContent = "(none)";
            return;
        }
        const slot = cd.slots[i];
        layerLabel.textContent = `Layer ${i+1}`;
        layerLabel.style.color = SLOT_COLORS[i];
        blendSelect.value = slot.blend_mode || "normal";
        opInput.value = slot.opacity ?? 1;
        opVal.textContent = `${Math.round((slot.opacity ?? 1)*100)}%`;
        setToggleState(flipHBtn, !!slot.flip_h);
        setToggleState(flipVBtn, !!slot.flip_v);
        bRow.inp.value = slot.brightness || 0; bRow.vl.textContent = (slot.brightness||0).toFixed(2);
        cRow.inp.value = slot.contrast   || 0; cRow.vl.textContent = (slot.contrast  ||0).toFixed(2);
        sRow.inp.value = slot.saturation || 0; sRow.vl.textContent = (slot.saturation||0).toFixed(2);
        const sh = slot.shadow || {};
        setToggleState(shEnableBtn, !!sh.enabled);
        shAngle.inp.value = sh.angle ?? 135; shAngle.vl.textContent = Math.round(sh.angle ?? 135);
        shDist.inp.value  = sh.distance ?? 8; shDist.vl.textContent  = Math.round(sh.distance ?? 8);
        shBlur.inp.value  = sh.blur ?? 12;    shBlur.vl.textContent  = Math.round(sh.blur ?? 12);
        shOp.inp.value    = sh.opacity ?? 0.6; shOp.vl.textContent   = (sh.opacity ?? 0.6).toFixed(2);
        shColor.value = colorToHex(sh.color || [0,0,0]);
    }
    function syncBg() {
        const cd = ensureComp(node);
        bgSelect.value = cd.background.type || "solid";
        bgCol1.value = colorToHex(cd.background.color1 || [40,40,48]);
        bgCol2.value = colorToHex(cd.background.color2 || [20,20,28]);
        bgAngle.inp.value = cd.background.angle ?? 0; bgAngle.vl.textContent = Math.round(cd.background.angle ?? 0);
        bgScale.inp.value = cd.background.scale ?? 1; bgScale.vl.textContent = (cd.background.scale ?? 1).toFixed(2);
        mpSelect.value = String(cd.megapixels || 1.0);
    }
    node._icSyncDetail = () => { syncDetail(); syncBg(); };

    let widget;
    function redraw() {
        const dpr = window.devicePixelRatio || 1;
        const W = Math.max(canvas.clientWidth || 600, 200);
        const panelH = W + LIST_H;
        canvas.width  = W * dpr;
        canvas.height = panelH * dpr;
        canvas.style.height = panelH + "px";
        const ctx = canvas.getContext("2d");
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        drawAll(ctx, node, W);
        const cd = getComp(node);
        if (cd?.display_w && cd?.display_h) {
            const cl = findClosestRatio(cd.display_w, cd.display_h);
            snapSelect.value = `${cl.w}:${cl.h}`;
        }
    }
    node._icRedraw = redraw;

    const _ro = new ResizeObserver(() => { redraw(); });
    _ro.observe(canvas);

    // ── Mouse handlers ───────────────────────────────────────────────────────
    canvas.addEventListener("mousedown", e => {
        if (e.button === 2) return; // let contextmenu handle
        e.stopPropagation();
        const [mx, my] = getCanvasXY(e, canvas);
        const hit = hitTest(mx, my, node);
        const cd  = ensureComp(node);
        const st  = node._icst;

        if (!hit) {
            // clicking in list area might hit a row; hitTest covered that via zones.
            // clicking in canvas area deselects handle but keep selection.
            redraw(); return;
        }

        if (hit.type === "resize") {
            st._dragMode = "resize";
            const dims = getDispDims(node, canvas.clientWidth - 2*MARGIN);
            st._dragData = {startMx:mx, startMy:my, startW:dims.w, startH:dims.h};
            return;
        }

        if (hit.type === "zone") {
            const {key, zone} = hit;
            const parseIdx = str => parseInt(str.split("_").pop());

            if (key.startsWith("row_")) {
                st._selIdx = parseIdx(key); redraw(); syncDetail(); return;
            }
            if (key.startsWith("bg_")) {
                const i = parseIdx(key); const was = cd.slots[i].is_bg;
                cd.slots.forEach(s => s.is_bg = false);
                if (!was) {
                    cd.slots[i].is_bg = true;
                    // Default BG to a cover-fit scale so it fills the frame.
                    const slot = cd.slots[i];
                    if (slot.src_w && slot.src_h) {
                        const max_dim = Math.max(slot.src_w, slot.src_h);
                        const cover = Math.max(
                            slot.src_w ? cd.canvas_w / slot.src_w : 1,
                            slot.src_h ? cd.canvas_h / slot.src_h : 1,
                        );
                        // UI scale convention: max_dim * scale * canvas_w/max_dim → drawn px
                        // To get image side = canvas side: scale = cover * max_dim / canvas_w
                        slot.scale   = cover * max_dim / cd.canvas_w;
                        slot.scale_x = 1.0; slot.scale_y = 1.0;
                        slot.x = 0.5; slot.y = 0.5; slot.rotation = 0;
                    }
                }
                setComp(node, cd); redraw(); return;
            }
            if (key.startsWith("eye_")) {
                const i = parseIdx(key);
                cd.slots[i].visible = !cd.slots[i].visible;
                setComp(node, cd); redraw(); return;
            }
            if (zone.type === "op") {
                const i = parseIdx(key); st._selIdx = i; syncDetail();
                st._dragMode = "op_slider";
                st._dragData = {slotIdx:i, startMx:mx, startVal:cd.slots[i].opacity ?? 1, barW:zone.w};
                return;
            }
            return;
        }

        if (hit.type === "move") {
            st._selIdx = hit.slotIdx;
            const slot = cd.slots[hit.slotIdx];
            const ca = st._caArea;
            st._dragMode = "move";
            st._dragData = {slotIdx:hit.slotIdx, startMx:mx, startMy:my,
                startX:slot.x ?? 0.5, startY:slot.y ?? 0.5, caW:ca.w, caH:ca.h};
            syncDetail(); redraw(); return;
        }
        if (hit.type === "rotate") {
            const ca = st._caArea;
            const tf = getSlotTransform(cd.slots[hit.slotIdx], ca);
            st._dragMode = "rotate";
            st._dragData = {
                slotIdx: hit.slotIdx, cx: tf.cx, cy: tf.cy,
                startRot: cd.slots[hit.slotIdx].rotation || 0,
                startAngle: Math.atan2(my - tf.cy, mx - tf.cx),
            };
            return;
        }
        if (hit.type === "scale_u") {
            const slot = cd.slots[hit.slotIdx];
            const ca = st._caArea;
            const tf = getSlotTransform(slot, ca);
            st._dragMode = "scale_u";
            st._dragData = {slotIdx:hit.slotIdx, startScale:slot.scale||1.0,
                startDist:Math.max(dist2([mx,my],[tf.cx,tf.cy]), 2)};
            return;
        }
        if (hit.type === "scale_x" || hit.type === "scale_y") {
            const slot = cd.slots[hit.slotIdx];
            const startDist = Math.max(Math.abs(dot2([mx-hit.cx, my-hit.cy], hit.axis)), 2);
            st._dragMode = hit.type;
            st._dragData = {
                slotIdx: hit.slotIdx, axis: hit.axis, cx: hit.cx, cy: hit.cy,
                startDist,
                startScale: hit.type === "scale_x" ? (slot.scale_x||1.0) : (slot.scale_y||1.0),
            };
            return;
        }
        redraw();
    });

    canvas.addEventListener("mousemove", e => {
        const st = node._icst; if (!st._dragMode) return;
        e.stopPropagation();
        const [mx, my] = getCanvasXY(e, canvas);
        const cd = ensureComp(node);
        const dd = st._dragData;

        if (st._dragMode === "resize") {
            const nw = Math.max(MIN_COMP, dd.startW + mx - dd.startMx);
            const nh = Math.max(MIN_COMP, dd.startH + my - dd.startMy);
            setDispDims(node, nw, nh); redraw(); return;
        }
        if (st._dragMode === "op_slider") {
            const v = Math.max(0, Math.min(1, dd.startVal + (mx - dd.startMx) / dd.barW));
            cd.slots[dd.slotIdx].opacity = v; setComp(node, cd); redraw();
            if (dd.slotIdx === st._selIdx) syncDetail();
            return;
        }
        if (st._dragMode === "move") {
            // Allow positions outside the canvas so layers can be pushed past
            // the edges (useful for cropping subjects that sit near a corner of
            // their source frame). Clamp generously to keep the drag bounded.
            let nx = Math.max(-2, Math.min(3, dd.startX + (mx-dd.startMx)/dd.caW));
            let ny = Math.max(-2, Math.min(3, dd.startY + (my-dd.startMy)/dd.caH));
            [nx, ny] = snapPos(nx, ny, e.shiftKey);
            cd.slots[dd.slotIdx].x = nx;
            cd.slots[dd.slotIdx].y = ny;
            setComp(node, cd); redraw(); return;
        }
        if (st._dragMode === "rotate") {
            const currentAngle = Math.atan2(my - dd.cy, mx - dd.cx);
            const delta = (currentAngle - dd.startAngle) * 180 / Math.PI;
            let newRot = dd.startRot + delta;
            if (e.shiftKey) newRot = Math.round(newRot / 15) * 15;
            newRot = ((newRot + 180) % 360 + 360) % 360 - 180;
            cd.slots[dd.slotIdx].rotation = newRot;
            setComp(node, cd); redraw(); return;
        }
        if (st._dragMode === "scale_u") {
            const slot = cd.slots[dd.slotIdx];
            const ca = node._icst._caArea;
            const tf = getSlotTransform(slot, ca);
            const r = dist2([mx,my],[tf.cx,tf.cy]) / dd.startDist;
            slot.scale = Math.max(0.005, dd.startScale * r);
            setComp(node, cd); redraw(); return;
        }
        if (st._dragMode === "scale_x") {
            const d = Math.abs(dot2([mx-dd.cx, my-dd.cy], dd.axis));
            cd.slots[dd.slotIdx].scale_x = Math.max(0.01, dd.startScale * d / dd.startDist);
            setComp(node, cd); redraw(); return;
        }
        if (st._dragMode === "scale_y") {
            const d = Math.abs(dot2([mx-dd.cx, my-dd.cy], dd.axis));
            cd.slots[dd.slotIdx].scale_y = Math.max(0.01, dd.startScale * d / dd.startDist);
            setComp(node, cd); redraw(); return;
        }
    });

    canvas.addEventListener("mouseup",    e => { node._icst._dragMode = null; e.stopPropagation(); });
    canvas.addEventListener("mouseleave", () => { node._icst._dragMode = null; });
    canvas.addEventListener("wheel", e => e.stopPropagation());

    // Right-click: context menu
    canvas.addEventListener("contextmenu", e => {
        e.preventDefault(); e.stopPropagation();
        const [mx, my] = getCanvasXY(e, canvas);
        const hit = hitTest(mx, my, node);
        let idx = null;
        if (hit?.type === "zone" && hit.key.startsWith("row_")) {
            idx = parseInt(hit.key.split("_").pop());
        } else if (hit?.type === "move" || hit?.type === "rotate" ||
                   hit?.type === "scale_u" || hit?.type === "scale_x" || hit?.type === "scale_y") {
            idx = hit.slotIdx;
        } else if (node._icst._selIdx != null) {
            idx = node._icst._selIdx;
        }
        if (idx == null) return;
        node._icst._selIdx = idx;
        redraw(); syncDetail();
        showContextMenu(e, node, idx);
    });

    // Double-click opacity bar in list → reset
    canvas.addEventListener("dblclick", e => {
        e.stopPropagation();
        const [mx, my] = getCanvasXY(e, canvas);
        const st = node._icst;
        for (const [key, zone] of Object.entries(st._zones || {})) {
            if (!_hr(mx, my, zone)) continue;
            if (zone.type === "op") {
                const i = parseInt(key.split("_").pop());
                const cd = ensureComp(node); cd.slots[i].opacity = 1.0;
                setComp(node, cd); redraw(); syncDetail(); return;
            }
        }
    });

    // Reload thumbnails after Python executes
    const origOnExecuted = node.onExecuted;
    node.onExecuted = function(output) {
        origOnExecuted?.call(this, output);
        setTimeout(() => loadAllImages(node), 200);
    };
    app.api.addEventListener("executed", (e) => {
        if (String(e.detail?.node) === String(node.id)) {
            setTimeout(() => loadAllImages(node), 200);
        }
    });

    function tryHide(n) {
        const w = _gw(node, W_COMP);
        if (w) { w.computeSize = () => [0,-4]; w.draw = () => {}; }
        if (n < 12) setTimeout(() => tryHide(n + 1), 200);
        else app.graph?.setDirtyCanvas(true, true);
    }
    setTimeout(() => tryHide(0), 80);

    widget = node.addDOMWidget("_image_canvas", "custom", wrapper, {
        serialize:  false,
        hideOnZoom: false,
        getValue:   () => "",
        setValue:   () => {},
    });
    widget.computeSize = (w) => [w, RATIO_BAR_H + w + LIST_H + DETAIL_BAR_H + SHADOW_BAR_H + BG_BAR_H + 14];

    setTimeout(() => { loadAllImages(node); syncDetail(); syncBg(); }, 400);
    return widget;
}

// ── Extension registration ──────────────────────────────────────────────────
app.registerExtension({
    name: "Eric.ImageComposer",
    async nodeCreated(node) {
        if (node.comfyClass !== NODE_TYPE) return;
        // Default large size — roughly 2× what the depth/pose composers use,
        // so the canvas window is comfortable to work in without zooming.
        if (!node.size || node.size[0] < 300) {
            const W = 1800;
            node.size = [W, RATIO_BAR_H + W + LIST_H + DETAIL_BAR_H + SHADOW_BAR_H + BG_BAR_H + 14];
        }
        setup(node);
    },
});
