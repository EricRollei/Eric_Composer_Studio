/**
 * depth_composer.js  v1 - Eric_Composer_Studio
 *
 * Interactive depth map compositor.
 * Up to 4 depth patches + optional masks composited onto a configurable canvas.
 * Per-element controls: position, uniform scale, X/Y scale (edge handles),
 * rotation, depth_placement, gradient_scale, BG toggle, visibility, z-order.
 */

import { app } from "../../scripts/app.js";

const NODE_TYPE   = "EricDepthComposer";
const W_COMP      = "composition_data";
const MAX_SLOTS   = 4;
// Canvas window is always a square: winW = winH = W - 2*MARGIN
// panelH, listY, ctrlY are computed per-draw and stored in node._dcst

const RATIO_BAR_H  = 34;
const MARGIN       = 10;
const MIN_COMP     = 80;
const PIXEL_SCALE  = 1024 / 640;   // 640px display → 1024px output
const MOV_R        = 9;
const EDGE_R       = 6;
const LG_RESIZE    = 16;
const ROT_R        = 7;    // rotate handle radius
const ROT_DIST     = 30;   // pixels above tf.top along the slot's local up axis

const SLOT_ROW_H  = 52;
const LIST_HDR_H  = 18;
const LIST_H      = LIST_HDR_H + MAX_SLOTS * SLOT_ROW_H + 6;   // 18+208+6 = 232
const CTRL_BAR_H  = 38;
// panelH(W) = W + LIST_H + CTRL_BAR_H  (since 2*MARGIN + (W-2*MARGIN) + LIST_H + CTRL_BAR_H)
// listY(W)  = W   (= MARGIN + winW + MARGIN = MARGIN + (W-2*MARGIN) + MARGIN = W)
// ctrlY(W)  = W + LIST_H

const SLOT_COLORS = ["#4A9EFF", "#FF6B35", "#4CAF50", "#E040FB"];

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
    if (!ch) return RATIOS[7];   // default: 1:1
    const aspect = cw / ch;
    let best = RATIOS[7], bestDiff = Infinity;
    for (const r of RATIOS) {
        const diff = Math.abs(r.w / r.h - aspect);
        if (diff < bestDiff) { best = r; bestDiff = diff; }
    }
    return best;
}

// ── Widget accessors ─────────────────────────────────────────────────────────
const _gw  = (n,k)   => n.widgets?.find(w => w.name === k);
const gval = (n,k,d) => _gw(n,k)?.value ?? d;
const sval = (n,k,v) => { const w = _gw(n,k); if (w) w.value = v; };
const getComp = n => { try { return JSON.parse(gval(n,W_COMP,"")||"null"); } catch { return null; } };
const setComp = (n,d) => sval(n, W_COMP, JSON.stringify(d));

// ── Default composition data ─────────────────────────────────────────────────
function defaultComp(dw, dh) {
    const w = dw || 640, h = dh || w;   // square default → 1024×1024 output
    return {
        display_w:        w,
        display_h:        h,
        canvas_w:  Math.round(w * PIXEL_SCALE / 8) * 8,
        canvas_h:  Math.round(h * PIXEL_SCALE / 8) * 8,
        background_depth: 0.10,
        slots: Array.from({length: MAX_SLOTS}, (_, i) => ({
            id: `s${i}`, label: `Slot ${i+1}`, is_bg: false,
            depth_placement: 0.70,
            gradient_scale: 1.0, visible: true, z_order: i,
            x: 0.5, y: 0.5, scale: 0.5, scale_x: 1.0, scale_y: 1.0,
            rotation: 0.0, src_w: 0, src_h: 0,
        })),
    };
}

function ensureComp(node) {
    let cd = getComp(node);
    if (!cd || !cd.slots || cd.slots.length !== MAX_SLOTS) {
        cd = defaultComp(cd?.display_w, cd?.display_h);
        setComp(node, cd);
    }
    return cd;
}

// ── Display dimensions ───────────────────────────────────────────────────────
// Content is contain-fitted within the square canvas window (winW × winW).
// The stored display_w/h define the target aspect ratio; the rendered area
// is always ≤ the window — letterboxed vertically or pillarboxed horizontally.
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
    cd.canvas_w  = Math.round(dw * PIXEL_SCALE / 8) * 8;
    cd.canvas_h  = Math.round(dh * PIXEL_SCALE / 8) * 8;
    setComp(node, cd);
}

// ── Geometry helpers ─────────────────────────────────────────────────────────
const lerp2  = (a, b) => [(a[0]+b[0])/2, (a[1]+b[1])/2];
const dist2  = (a, b) => Math.hypot(b[0]-a[0], b[1]-a[1]);
const dot2   = (a, b) => a[0]*b[0] + a[1]*b[1];

function getSlotTransform(slot, ca) {
    const src_w = slot.src_w || 1, src_h = slot.src_h || 1;
    // slot.scale is canvas-relative: max(src_w,src_h) * slot.scale maps to ca.w pixels.
    // Python uses the same convention (canvas_w / max_dim), so the preview matches output.
    const max_dim = Math.max(src_w, src_h);
    const base_px = (slot.scale || 0.5) * ca.w / max_dim;
    const sx    = base_px * (slot.scale_x || 1.0);
    const sy    = base_px * (slot.scale_y || 1.0);
    const rad   = (slot.rotation || 0) * Math.PI / 180;
    const cx    = (slot.x ?? 0.5) * ca.w + ca.x;
    const cy    = (slot.y ?? 0.5) * ca.h + ca.y;
    const cos_r = Math.cos(rad), sin_r = Math.sin(rad);
    const hw    = src_w / 2 * sx, hh = src_h / 2 * sy;

    function tr(lx, ly) {
        return [cx + lx * cos_r - ly * sin_r, cy + lx * sin_r + ly * cos_r];
    }
    const tl = tr(-hw,-hh), tr_ = tr(hw,-hh), br = tr(hw,hh), bl = tr(-hw,hh);
    const top_mid = lerp2(tl, tr_);
    // Rotate handle: ROT_DIST px above top-center along slot's local up axis (-sin_r, cos_r)
    const rot_hnd = [
        top_mid[0] - sin_r * ROT_DIST,
        top_mid[1] + cos_r * ROT_DIST,
    ];
    return {
        cx, cy, sx, sy, rad, cos_r, sin_r, hw, hh,
        tl, tr: tr_, br, bl,
        top: top_mid,
        right:  lerp2(tr_, br),
        bottom: lerp2(br, bl),
        left:   lerp2(bl, tl),
        rot_hnd,
        src_w, src_h,
    };
}

// ── Image loading ─────────────────────────────────────────────────────────────
function loadSlotImage(node, slotIdx) {
    // Always derive nodeId from node.id directly — widget timing is unreliable
    const nodeId = String(node.id);
    const url    = `/eric_composer_studio/depth_slot/${nodeId}/${slotIdx}?t=${Date.now()}`;
    const img    = new Image();
    img.onload = () => {
        console.log(`[DepthComposer] slot ${slotIdx} loaded: ${img.naturalWidth}×${img.naturalHeight}`);
        node._dcImages[slotIdx] = img;
        // Update src dimensions in composition data from actual image
        const cd = ensureComp(node);
        if (cd.slots[slotIdx]) {
            const slot      = cd.slots[slotIdx];
            const wasFresh  = slot.src_w === 0;   // true = newly assigned, not a reload
            slot.src_w = img.naturalWidth;
            slot.src_h = img.naturalHeight;
            // On first load only: set initial scale so image is half the canvas size.
            // This guarantees all handles are visible and reachable immediately.
            // After that the user has full control; we never touch scale again.
            if (wasFresh && img.naturalWidth > 0) {
                // Canvas-relative initial scale: 0.5 → image max_dim = 50% of canvas.
                // Python uses the same normalization so preview matches output exactly.
                slot.scale = 0.5;
            }
            setComp(node, cd);
        }
        node._dcRedraw?.();
    };
    img.onerror = () => { console.warn(`[DepthComposer] slot ${slotIdx} image load failed: ${url}`); node._dcImages[slotIdx] = null; };
    img.src = url;
}

function loadAllImages(node) {
    const nodeId = String(node.id);
    console.log(`[DepthComposer] loadAllImages node=${nodeId}`);
    fetch(`/eric_composer_studio/depth_composition/${nodeId}`)
        .then(r => r.json())
        .then(data => {
            const avail = data.slots_available || [];
            console.log(`[DepthComposer] slots_available=`, avail);
            let anyAvail = false;
            for (let i = 0; i < MAX_SLOTS; i++) {
                if (avail[i]) { loadSlotImage(node, i); anyAvail = true; }
                else node._dcImages[i] = null;
            }
            if (!anyAvail) node._dcRedraw?.();
        })
        .catch(err => { console.error(`[DepthComposer] loadAllImages fetch error:`, err); });
}

// ── Drawing helpers ───────────────────────────────────────────────────────────
function drawSliderBar(ctx, x, y, w, h, val, trackCol, fillCol) {
    ctx.fillStyle = "#111"; ctx.fillRect(x, y, w, h);
    ctx.strokeStyle = "#2A3A4A"; ctx.lineWidth = 0.5;
    ctx.strokeRect(x, y, w, h);
    ctx.fillStyle = fillCol;
    ctx.fillRect(x, y, Math.max(0, Math.min(w, w * Math.max(0, Math.min(1, val)))), h);
}

function drawSmallBtn(ctx, x, y, w, h, label, col) {
    ctx.fillStyle = "#121218"; ctx.strokeStyle = col; ctx.lineWidth = 1;
    ctx.fillRect(x, y, w, h); ctx.strokeRect(x, y, w, h);
    ctx.fillStyle = col; ctx.font = `bold ${h-1}px monospace`;
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText(label, x + w/2, y + h/2);
}

function drawRndRect(ctx, x, y, w, h, r) {
    if (ctx.roundRect) { ctx.roundRect(x, y, w, h, r); }
    else { ctx.rect(x, y, w, h); }
}

// ── Main canvas draw ──────────────────────────────────────────────────────────
function drawAll(ctx, node, W) {
    const cd   = ensureComp(node);
    const st   = node._dcst;
    const winW = W - 2 * MARGIN;            // square window side length
    const dims = getDispDims(node, winW);
    const cw   = dims.w, ch = dims.h;
    const listY  = W;                        // = MARGIN + winW + MARGIN
    const ctrlY  = W + LIST_H;
    const panelH = W + LIST_H + CTRL_BAR_H;
    st._listY = listY; st._ctrlY = ctrlY;
    // Center content within square window (letterbox / pillarbox)
    const ca   = {
        x: MARGIN + Math.round((winW - cw) / 2),
        y: MARGIN + Math.round((winW - ch) / 2),
        w: cw, h: ch,
    };

    // Panel background
    ctx.fillStyle = "#06060F"; ctx.fillRect(0, 0, W, panelH);
    ctx.strokeStyle = "#1E2A38"; ctx.lineWidth = 1;
    ctx.strokeRect(1, 1, W-2, panelH-2);

    // Square canvas window background (letterbox / pillarbox bars)
    ctx.fillStyle = "#030308";
    ctx.fillRect(MARGIN, MARGIN, winW, winW);
    ctx.strokeStyle = "#1A2430"; ctx.lineWidth = 1;
    ctx.strokeRect(MARGIN, MARGIN, winW, winW);

    // Composition canvas area — fill color reflects background_depth
    const bgV  = Math.round((cd.background_depth ?? 0.10) * 255);
    const bgHx = bgV.toString(16).padStart(2, "0");
    ctx.fillStyle = `#${bgHx}${bgHx}${bgHx}`; ctx.fillRect(ca.x, ca.y, ca.w, ca.h);
    ctx.strokeStyle = "#334455"; ctx.lineWidth = 1;
    ctx.strokeRect(ca.x, ca.y, ca.w, ca.h);

    // Guide grid
    ctx.save(); ctx.strokeStyle = "#141E2A"; ctx.lineWidth = 0.5;
    for (let i = 1; i < 3; i++) {
        const gx = ca.x + ca.w*i/3, gy = ca.y + ca.h*i/3;
        ctx.beginPath(); ctx.moveTo(gx, ca.y); ctx.lineTo(gx, ca.y+ca.h); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(ca.x, gy); ctx.lineTo(ca.x+ca.w, gy); ctx.stroke();
    }
    ctx.restore();

    // Output canvas dimensions label
    ctx.fillStyle = "#2A3A4A"; ctx.font = "10px monospace";
    ctx.textAlign = "center"; ctx.textBaseline = "bottom";
    ctx.fillText(`${cd.canvas_w} × ${cd.canvas_h}`, ca.x + ca.w/2, ca.y + ca.h - 2);

    // Check if any slots have data
    const hasAny = cd.slots.some((s, i) => s.src_w > 0 && node._dcImages[i]);

    if (!hasAny) {
        ctx.fillStyle = "#445566"; ctx.font = "12px sans-serif";
        ctx.textAlign = "center"; ctx.textBaseline = "middle";
        ctx.fillText("Connect depth maps and queue workflow", ca.x + ca.w/2, ca.y + ch/2 - 8);
        ctx.font = "10px monospace"; ctx.fillStyle = "#2A3A4A";
        ctx.fillText("depth_1..4 + optional mask_1..4", ca.x + ca.w/2, ca.y + ch/2 + 10);
    }

    // Clip depth patch rendering to canvas area
    ctx.save();
    ctx.beginPath(); ctx.rect(ca.x, ca.y, ca.w, ca.h); ctx.clip();

    // Sort back-to-front by depth_placement ascending.
    // Lower depth_placement = darker = further away = drawn behind (occluded).
    // BG flag only means "no mask cutout" — it does NOT force a slot behind others.
    // z_order is used as a tiebreaker when depth_placements are equal.
    const idxOrdered = cd.slots
        .map((s, i) => ({s, i}))
        .sort((a, b) => {
            const dpDiff = (a.s.depth_placement || 0.7) - (b.s.depth_placement || 0.7);
            if (Math.abs(dpDiff) > 0.001) return dpDiff;
            return (a.s.z_order || 0) - (b.s.z_order || 0);
        });

    for (const {s: slot, i} of idxOrdered) {
        if (!slot.visible || !slot.src_w) continue;
        const img = node._dcImages[i];
        if (!img) continue;
        const tf = getSlotTransform(slot, ca);
        // Approximate depth_placement as brightness for canvas preview
        const dp = slot.depth_placement ?? 0.7;
        const brightness = Math.max(0.05, dp * 2.0).toFixed(2);
        ctx.save();
        ctx.filter = `brightness(${brightness})`;
        ctx.translate(tf.cx, tf.cy);
        ctx.rotate(tf.rad);
        ctx.scale(tf.sx, tf.sy);
        ctx.drawImage(img, -tf.src_w/2, -tf.src_h/2, tf.src_w, tf.src_h);
        ctx.filter = "none";
        ctx.restore();
    }
    ctx.restore(); // end clip

    // ── Slot handles ─────────────────────────────────────────────────────────
    const selIdx = st._selIdx;
    cd.slots.forEach((slot, i) => {
        if (!slot.visible || !slot.src_w) return;
        const tf  = getSlotTransform(slot, ca);
        const col = SLOT_COLORS[i] || "#888";
        const sel = (i === selIdx);

        // Centroid handle
        ctx.save();
        ctx.strokeStyle = sel ? "#FFF" : col;
        ctx.fillStyle   = sel ? "#FFFFFF44" : col + "44";
        ctx.lineWidth   = sel ? 2 : 1.5;
        ctx.beginPath(); ctx.arc(tf.cx, tf.cy, MOV_R, 0, Math.PI * 2);
        ctx.fill(); ctx.stroke();
        ctx.restore();

        if (sel) {
            // Bounding box outline (dashed)
            ctx.save();
            ctx.strokeStyle = col + "AA"; ctx.lineWidth = 1;
            ctx.setLineDash([3, 3]);
            ctx.beginPath();
            ctx.moveTo(tf.tl[0], tf.tl[1]); ctx.lineTo(tf.tr[0], tf.tr[1]);
            ctx.lineTo(tf.br[0], tf.br[1]); ctx.lineTo(tf.bl[0], tf.bl[1]);
            ctx.closePath(); ctx.stroke();
            ctx.setLineDash([]);
            ctx.restore();

            // Edge handles: X=left/right, Y=top/bottom
            _drawEdgeHnd(ctx, tf.right,  col, "X");
            _drawEdgeHnd(ctx, tf.left,   col, "X");
            _drawEdgeHnd(ctx, tf.top,    col, "Y");
            _drawEdgeHnd(ctx, tf.bottom, col, "Y");

            // Corner handle (uniform scale) at br
            ctx.save();
            ctx.fillStyle = "#1A2A3A"; ctx.strokeStyle = "#88AAFF"; ctx.lineWidth = 1.5;
            ctx.fillRect(tf.br[0]-EDGE_R, tf.br[1]-EDGE_R, EDGE_R*2, EDGE_R*2);
            ctx.strokeRect(tf.br[0]-EDGE_R, tf.br[1]-EDGE_R, EDGE_R*2, EDGE_R*2);
            ctx.restore();

            // Rotate handle: circle above top-center, connected by a stem line
            ctx.save();
            ctx.strokeStyle = col + "99"; ctx.lineWidth = 1;
            ctx.setLineDash([2, 2]);
            ctx.beginPath();
            ctx.moveTo(tf.top[0], tf.top[1]);
            ctx.lineTo(tf.rot_hnd[0], tf.rot_hnd[1]);
            ctx.stroke();
            ctx.setLineDash([]);
            ctx.fillStyle   = st._dragMode === "rotate" && st._dragData?.slotIdx === i ? "#FFCC44CC" : "#2A1A0ACC";
            ctx.strokeStyle = "#FFCC44";
            ctx.lineWidth   = 1.5;
            ctx.beginPath(); ctx.arc(tf.rot_hnd[0], tf.rot_hnd[1], ROT_R, 0, Math.PI * 2);
            ctx.fill(); ctx.stroke();
            // ↻ glyph
            ctx.fillStyle = "#FFCC44"; ctx.font = `bold ${ROT_R + 3}px sans-serif`;
            ctx.textAlign = "center"; ctx.textBaseline = "middle";
            ctx.fillText("↻", tf.rot_hnd[0], tf.rot_hnd[1] + 1);
            ctx.restore();

            // Transform readout
            const rot = Math.round(slot.rotation || 0);
            const sxv = (slot.scale_x||1).toFixed(2), syv = (slot.scale_y||1).toFixed(2);
            if (rot !== 0 || sxv !== "1.00" || syv !== "1.00") {
                ctx.fillStyle = "#334455"; ctx.font = "9px monospace";
                ctx.textAlign = "center"; ctx.textBaseline = "top";
                ctx.fillText(`${rot}°  sx${sxv}  sy${syv}`, tf.cx, tf.cy + MOV_R + 3);
            }
        }
    });

    // Resize corner handle
    const rx = ca.x + ca.w, ry = ca.y + ca.h;
    const isRes = st._dragMode === "resize";
    ctx.save();
    ctx.fillStyle   = isRes ? "#5599FFCC" : "#33445588";
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

    // ── List panel ────────────────────────────────────────────────────────────
    _drawList(ctx, node, W, cd, selIdx);

    // ── Control bar ───────────────────────────────────────────────────────────
    _drawCtrlBar(ctx, node, W, cd);
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
    const st = node._dcst;
    const lx = MARGIN, lw = W - 2*MARGIN;
    const LIST_Y = st._listY;

    ctx.fillStyle = "#0D1A2A";
    ctx.fillRect(lx, LIST_Y, lw, LIST_H);
    ctx.strokeStyle = "#1E2A38"; ctx.lineWidth = 1;
    ctx.strokeRect(lx, LIST_Y, lw, LIST_H);

    // Header
    ctx.fillStyle = "#2A4A6A"; ctx.font = "10px monospace";
    ctx.textAlign = "left"; ctx.textBaseline = "middle";
    ctx.fillText("Depth Slots", lx+6, LIST_Y + LIST_HDR_H/2);

    st._zones = {};

    cd.slots.forEach((slot, i) => {
        const rowY  = LIST_Y + LIST_HDR_H + i * SLOT_ROW_H + 2;
        const rowH  = SLOT_ROW_H - 2;
        const col   = SLOT_COLORS[i];
        const sel   = (i === selIdx);
        const hasD  = slot.src_w > 0;
        const rX    = lx + 6;

        // Row background
        ctx.fillStyle   = sel ? "#0A1828" : "#080F18";
        ctx.fillRect(lx+2, rowY, lw-4, rowH);
        if (sel) {
            ctx.strokeStyle = col + "66"; ctx.lineWidth = 1;
            ctx.strokeRect(lx+2, rowY, lw-4, rowH);
        }

        const midY1 = rowY + 13;   // top row center
        const midY2 = rowY + 37;   // bottom row center

        // Color dot
        ctx.fillStyle   = hasD ? col : "#334455";
        ctx.strokeStyle = "#000"; ctx.lineWidth = 0.5;
        ctx.beginPath(); ctx.arc(rX+4, midY1, 5, 0, Math.PI*2); ctx.fill(); ctx.stroke();

        // Slot label + dimensions
        ctx.fillStyle = hasD ? "#99BBCC" : "#445566";
        ctx.font = "10px monospace"; ctx.textAlign = "left"; ctx.textBaseline = "middle";
        ctx.fillText(slot.label, rX+14, midY1);
        ctx.fillStyle = "#2A4A6A"; ctx.font = "8px monospace";
        ctx.fillText(hasD ? `${slot.src_w}×${slot.src_h}` : "no input", rX+14, midY1+10);

        // ── Top-right buttons: BG | 👁 | ↑ | ↓ ──────────────────────────────
        const btnH = 14, btnY = rowY + 4;
        let bx = lx + lw - 8;

        // ↓ Z-order down
        bx -= 16;
        ctx.fillStyle = "#121218"; ctx.strokeStyle = "#334455"; ctx.lineWidth = 1;
        ctx.fillRect(bx, btnY, 14, btnH); ctx.strokeRect(bx, btnY, 14, btnH);
        ctx.fillStyle = "#668899"; ctx.font = "10px sans-serif";
        ctx.textAlign = "center"; ctx.textBaseline = "middle";
        ctx.fillText("↓", bx+7, btnY+7);
        st._zones[`zdn_${i}`] = {x:bx, y:btnY, w:14, h:btnH};

        // ↑ Z-order up
        bx -= 17;
        ctx.fillStyle = "#121218"; ctx.strokeStyle = "#334455"; ctx.lineWidth = 1;
        ctx.fillRect(bx, btnY, 14, btnH); ctx.strokeRect(bx, btnY, 14, btnH);
        ctx.fillStyle = "#668899"; ctx.font = "10px sans-serif";
        ctx.textAlign = "center"; ctx.textBaseline = "middle";
        ctx.fillText("↑", bx+7, btnY+7);
        st._zones[`zup_${i}`] = {x:bx, y:btnY, w:14, h:btnH};

        // 👁 Visibility
        bx -= 25;
        const eyeW = 22;
        ctx.fillStyle   = slot.visible ? "#121218" : "#1A1A0A";
        ctx.strokeStyle = slot.visible ? "#334455" : "#555511"; ctx.lineWidth = 1;
        ctx.beginPath(); drawRndRect(ctx, bx, btnY, eyeW, btnH, 2); ctx.fill(); ctx.stroke();
        ctx.fillStyle = slot.visible ? "#668899" : "#885511";
        ctx.font = "9px sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
        ctx.fillText(slot.visible ? "👁" : "◌", bx+eyeW/2, btnY+7);
        st._zones[`eye_${i}`] = {x:bx, y:btnY, w:eyeW, h:btnH};

        // BG toggle
        bx -= 28;
        const bgW = 24, isBG = slot.is_bg || false;
        ctx.fillStyle   = isBG ? "#1A3A1A" : "#121218";
        ctx.strokeStyle = isBG ? "#44AA44" : "#334455"; ctx.lineWidth = 1;
        ctx.beginPath(); drawRndRect(ctx, bx, btnY, bgW, btnH, 2); ctx.fill(); ctx.stroke();
        ctx.fillStyle = isBG ? "#66DD66" : "#445566";
        ctx.font = "bold 8px monospace"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
        ctx.fillText("BG", bx+bgW/2, btnY+7);
        st._zones[`bg_${i}`] = {x:bx, y:btnY, w:bgW, h:btnH};

        // ── Bottom row: depth_placement | gradient_scale | rot ────────────────
        const sH = 9;
        const sY = midY2 - sH/2;
        const halfW = Math.floor((lw - 16) / 2) - 18;

        // depth_placement slider
        ctx.fillStyle = "#446688"; ctx.font = "8px monospace";
        ctx.textAlign = "left"; ctx.textBaseline = "middle";
        ctx.fillText("dp:", rX, midY2);
        const dpX = rX + 18, dpW = halfW;
        drawSliderBar(ctx, dpX, sY, dpW, sH, slot.depth_placement ?? 0.7, "#2255AA", "#4488FF");
        ctx.fillStyle = "#6699BB"; ctx.font = "8px monospace";
        ctx.textAlign = "left"; ctx.textBaseline = "middle";
        ctx.fillText((slot.depth_placement ?? 0.7).toFixed(2), dpX+dpW+3, midY2);
        st._zones[`dp_${i}`] = {x:dpX, y:sY, w:dpW, h:sH+2, type:"dp"};

        // gradient_scale slider
        const gsLX = dpX + dpW + 28;
        ctx.fillStyle = "#446688"; ctx.font = "8px monospace";
        ctx.textAlign = "left"; ctx.textBaseline = "middle";
        ctx.fillText("gs:", gsLX, midY2);
        const gsX = gsLX + 18, gsW = halfW;
        drawSliderBar(ctx, gsX, sY, gsW, sH, slot.gradient_scale ?? 1.0, "#225533", "#44AA66");
        ctx.fillStyle = "#6699BB"; ctx.font = "8px monospace";
        ctx.textAlign = "left"; ctx.textBaseline = "middle";
        ctx.fillText((slot.gradient_scale ?? 1.0).toFixed(2), gsX+gsW+3, midY2);
        st._zones[`gs_${i}`] = {x:gsX, y:sY, w:gsW, h:sH+2, type:"gs"};

        // Rotation controls
        const rotX = gsX + gsW + 28;
        ctx.fillStyle = "#446688"; ctx.font = "8px monospace";
        ctx.textAlign = "left"; ctx.textBaseline = "middle";
        ctx.fillText("rot:", rotX, midY2);
        const rotValX = rotX + 22;
        ctx.fillStyle = "#88AACC";
        ctx.fillText(`${Math.round(slot.rotation || 0)}°`, rotValX, midY2);
        drawSmallBtn(ctx, rotValX+22, sY, 13, sH+2, "-", "#885533");
        st._zones[`rot_dn_${i}`] = {x:rotValX+22, y:sY, w:13, h:sH+2};
        drawSmallBtn(ctx, rotValX+37, sY, 13, sH+2, "+", "#335588");
        st._zones[`rot_up_${i}`] = {x:rotValX+37, y:sY, w:13, h:sH+2};
    });
}

function _drawCtrlBar(ctx, node, W, cd) {
    const st  = node._dcst;
    const lx  = MARGIN, lw = W - 2*MARGIN;
    const CTRL_Y = st._ctrlY;
    const midY = CTRL_Y + CTRL_BAR_H/2;

    ctx.fillStyle = "#0A0A14";
    ctx.fillRect(lx, CTRL_Y, lw, CTRL_BAR_H);
    ctx.strokeStyle = "#1A2A3A"; ctx.lineWidth = 1;
    ctx.strokeRect(lx, CTRL_Y, lw, CTRL_BAR_H);

    let curX = lx + 8;

    // Background depth slider (key must NOT start with "bg_" — that prefix is reserved for slot BG buttons)
    ctx.fillStyle = "#446688"; ctx.font = "9px monospace";
    ctx.textAlign = "left"; ctx.textBaseline = "middle";
    ctx.fillText("BG depth:", curX, midY); curX += 58;
    const bgW = 130;
    const bgVal = cd.background_depth ?? 0.1;
    drawSliderBar(ctx, curX, midY-5, bgW, 10, bgVal / 0.5, "#1A1A2A", "#334466");
    st._zones["ctrl_bgdepth"] = {x:curX, y:midY-5, w:bgW, h:10};
    curX += bgW + 6;
    ctx.fillStyle = "#6699BB"; ctx.font = "9px monospace";
    ctx.textAlign = "left"; ctx.textBaseline = "middle";
    ctx.fillText(bgVal.toFixed(3), curX, midY); curX += 38;

    // Canvas output dims readout
    ctx.fillStyle = "#2A4A6A"; ctx.font = "9px monospace";
    ctx.textAlign = "left"; ctx.textBaseline = "middle";
    ctx.fillText(`out: ${cd.canvas_w} × ${cd.canvas_h} px`, curX, midY);
}

// ── Hit testing ───────────────────────────────────────────────────────────────
function _hr(mx, my, z) { return mx>=z.x && mx<=z.x+z.w && my>=z.y && my<=z.y+z.h; }

function hitTest(mx, my, node) {
    const cd = ensureComp(node);
    const st = node._dcst;
    const ca = st._caArea;

    // Zone buttons / sliders (list + ctrl bar)
    for (const [key, zone] of Object.entries(st._zones || {})) {
        if (Array.isArray(zone) || key === "_ratioZones") continue;
        if (key.startsWith("ratio_")) {
            if (_hr(mx, my, zone)) return {type:"ratio", r:zone.r};
            continue;
        }
        if (_hr(mx, my, zone)) return {type:"zone", key, zone};
    }

    if (!ca) return null;

    // Resize corner
    const rx = ca.x + ca.w, ry = ca.y + ca.h;
    if (Math.hypot(mx-rx, my-ry) < LG_RESIZE + 4) return {type:"resize"};

    // Slot handles
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
            if (dist2([mx,my], tf.br)     < EDGE_R+4) return {type:"scale_u", slotIdx:i};
            if (dist2([mx,my], tf.right)  < EDGE_R+3) return {type:"scale_x", slotIdx:i, axis:[tf.cos_r, tf.sin_r], cx:tf.cx, cy:tf.cy};
            if (dist2([mx,my], tf.left)   < EDGE_R+3) return {type:"scale_x", slotIdx:i, axis:[tf.cos_r, tf.sin_r], cx:tf.cx, cy:tf.cy};
            if (dist2([mx,my], tf.top)    < EDGE_R+3) return {type:"scale_y", slotIdx:i, axis:[-tf.sin_r, tf.cos_r], cx:tf.cx, cy:tf.cy};
            if (dist2([mx,my], tf.bottom) < EDGE_R+3) return {type:"scale_y", slotIdx:i, axis:[-tf.sin_r, tf.cos_r], cx:tf.cx, cy:tf.cy};
        }
        if (dist2([mx,my], [tf.cx, tf.cy]) < MOV_R+3) return {type:"move", slotIdx:i};
    }
    return null;
}

// ── Mouse coordinate helper ────────────────────────────────────────────────────
function getCanvasXY(e, canvas) {
    const rect = canvas.getBoundingClientRect();
    const dpr  = window.devicePixelRatio || 1;
    return [
        (e.clientX - rect.left) * (canvas.width  / dpr / rect.width),
        (e.clientY - rect.top)  * (canvas.height / dpr / rect.height),
    ];
}

// ── Node setup ────────────────────────────────────────────────────────────────
function setup(node) {
    node._dcImages  = [null, null, null, null];
    node._dcst = {
        _selIdx: null, _dragMode: null, _dragData: null,
        _caArea: null, _zones: {}, _listY: null, _ctrlY: null,
    };

    // ── DOM wrapper ───────────────────────────────────────────────────────────
    const wrapper = document.createElement("div");
    wrapper.style.cssText = "width:100%;overflow:hidden;user-select:none;pointer-events:all;display:flex;flex-direction:column;";

    // Ratio bar
    const ratioBar = document.createElement("div");
    ratioBar.style.cssText = [
        "display:flex","align-items:center","gap:6px",
        "padding:4px 10px","background:#0A0A18","border-bottom:1px solid #1E2A38",
        `height:${RATIO_BAR_H}px`,"box-sizing:border-box","flex-shrink:0",
    ].join(";");

    const ratioLabel = document.createElement("span");
    ratioLabel.textContent = "Snap:";
    ratioLabel.style.cssText = "color:#446688;font:11px monospace;white-space:nowrap;";

    const ratioSelect = document.createElement("select");
    ratioSelect.style.cssText = "background:#12121E;color:#88AACC;border:1px solid #2A3A4A;border-radius:3px;font:11px monospace;padding:2px 4px;cursor:pointer;pointer-events:all;";
    const freeOpt = document.createElement("option");
    freeOpt.value = ""; freeOpt.textContent = "- free -";
    ratioSelect.appendChild(freeOpt);
    RATIOS.forEach(r => {
        const o = document.createElement("option"); o.value = `${r.w}:${r.h}`; o.textContent = r.label;
        ratioSelect.appendChild(o);
    });
    ratioSelect.addEventListener("change", () => {
        const v = ratioSelect.value; if (!v) return;
        const [rw, rh] = v.split(":").map(Number);
        const dims = getDispDims(node, canvas.clientWidth - 2*MARGIN);
        setDispDims(node, dims.w, Math.round(dims.w * rh / rw));
        redraw();
    });

    ratioBar.appendChild(ratioLabel);
    ratioBar.appendChild(ratioSelect);

    // Reset button
    const resetBtn = document.createElement("button");
    resetBtn.textContent = "↺ Reset";
    resetBtn.title = "Reset all slot transforms to defaults";
    resetBtn.style.cssText = [
        "margin-left:auto","background:#0A1A0A","color:#668844",
        "border:1px solid #334422","border-radius:3px","font:11px monospace",
        "padding:2px 8px","cursor:pointer","pointer-events:all","white-space:nowrap",
    ].join(";");
    resetBtn.addEventListener("click", (e) => {
        e.preventDefault(); e.stopPropagation();
        const cd = ensureComp(node);
        cd.slots.forEach(s => {
            s.x = 0.5; s.y = 0.5; s.scale = 1.0; s.scale_x = 1.0;
            s.scale_y = 1.0; s.rotation = 0.0;
        });
        setComp(node, cd); redraw();
    });
    ratioBar.appendChild(resetBtn);

    // Reload button — fetches fresh thumbnails from server
    const reloadBtn = document.createElement("button");
    reloadBtn.textContent = "⟳ Reload";
    reloadBtn.title = "Reload slot images from server after queuing";
    reloadBtn.style.cssText = [
        "background:#0A1A2A","color:#6699BB",
        "border:1px solid #2A3A4A","border-radius:3px","font:11px monospace",
        "padding:2px 8px","cursor:pointer","pointer-events:all","white-space:nowrap",
    ].join(";");
    reloadBtn.addEventListener("click", (e) => {
        e.preventDefault(); e.stopPropagation();
        reloadBtn.textContent = "⟳ Loading...";
        reloadBtn.disabled = true;
        loadAllImages(node);
        setTimeout(() => { reloadBtn.textContent = "⟳ Reload"; reloadBtn.disabled = false; }, 1500);
    });
    ratioBar.appendChild(reloadBtn);

    wrapper.appendChild(ratioBar);

    // Main canvas
    const canvas = document.createElement("canvas");
    canvas.style.cssText = "display:block;width:100%;pointer-events:all;touch-action:none;cursor:default;flex-shrink:0;";
    wrapper.appendChild(canvas);

    let widget;  // declared before redraw so the closure can reference it

    function redraw() {
        const dpr = window.devicePixelRatio || 1;
        const W   = Math.max(canvas.clientWidth || 600, 150);
        const panelH = W + LIST_H + CTRL_BAR_H;
        canvas.width  = W * dpr;
        canvas.height = panelH * dpr;
        canvas.style.height = panelH + "px";
        const ctx = canvas.getContext("2d");
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        drawAll(ctx, node, W);
        // Sync ratio select
        const cd = getComp(node);
        if (cd?.display_w && cd?.display_h) {
            const cl = findClosestRatio(cd.display_w, cd.display_h);
            ratioSelect.value = `${cl.w}:${cl.h}`;
        }
    }
    node._dcRedraw = redraw;

    // Fire an initial redraw as soon as the canvas has a real layout width.
    // A blind 400ms timeout fires while clientWidth is still 0 (node not yet
    // in the DOM), producing a W=600 phantom draw.  The ResizeObserver fires
    // when the actual pixel width is known and triggers a correct draw.
    const _ro = new ResizeObserver(() => { redraw(); });
    _ro.observe(canvas);

    // ── Mouse handlers ────────────────────────────────────────────────────────
    canvas.addEventListener("mousedown", e => {
        e.stopPropagation();
        const [mx, my] = getCanvasXY(e, canvas);
        const hit = hitTest(mx, my, node);
        const cd  = ensureComp(node);
        const st  = node._dcst;

        if (!hit) { st._selIdx = null; redraw(); return; }

        if (hit.type === "resize") {
            st._dragMode = "resize";
            const dims = getDispDims(node, canvas.clientWidth - 2*MARGIN);
            st._dragData = {startMx:mx, startMy:my, startW:dims.w, startH:dims.h};
            return;
        }

        if (hit.type === "ratio") {
            const dims = getDispDims(node, canvas.clientWidth - 2*MARGIN);
            setDispDims(node, dims.w, Math.round(dims.w * hit.r.h / hit.r.w));
            redraw(); return;
        }

        if (hit.type === "zone") {
            const {key, zone} = hit;
            const parseIdx = str => parseInt(str.split("_").pop());

            if (key.startsWith("bg_")) {
                const i = parseIdx(key);
                const wasBG = cd.slots[i].is_bg;
                cd.slots.forEach(s => { s.is_bg = false; });
                if (!wasBG) cd.slots[i].is_bg = true;
                setComp(node, cd); redraw(); return;
            }
            if (key.startsWith("eye_")) {
                const i = parseIdx(key);
                cd.slots[i].visible = !cd.slots[i].visible;
                setComp(node, cd); redraw(); return;
            }
            if (key.startsWith("zup_")) {
                const i = parseIdx(key);
                if (i < MAX_SLOTS-1) {
                    [cd.slots[i].z_order, cd.slots[i+1].z_order] =
                    [cd.slots[i+1].z_order, cd.slots[i].z_order];
                    setComp(node, cd); redraw();
                }
                return;
            }
            if (key.startsWith("zdn_")) {
                const i = parseIdx(key);
                if (i > 0) {
                    [cd.slots[i].z_order, cd.slots[i-1].z_order] =
                    [cd.slots[i-1].z_order, cd.slots[i].z_order];
                    setComp(node, cd); redraw();
                }
                return;
            }
            if (key.startsWith("rot_dn_")) {
                const i = parseIdx(key);
                cd.slots[i].rotation = ((cd.slots[i].rotation || 0) - 5 + 180) % 360 - 180;
                if (cd.slots[i].rotation < -180) cd.slots[i].rotation += 360;
                setComp(node, cd); redraw(); return;
            }
            if (key.startsWith("rot_up_")) {
                const i = parseIdx(key);
                cd.slots[i].rotation = ((cd.slots[i].rotation || 0) + 5 + 180) % 360 - 180;
                setComp(node, cd); redraw(); return;
            }
            if (key === "ctrl_bgdepth") {
                st._dragMode = "bgdepth_slider";
                st._dragData = {startMx:mx, startVal:cd.background_depth ?? 0.1, barW:zone.w};
                return;
            }
            if (zone.type === "dp") {
                const i = parseIdx(key);
                st._selIdx  = i;
                st._dragMode = "dp_slider";
                st._dragData = {slotIdx:i, startMx:mx, startVal:cd.slots[i].depth_placement ?? 0.7, barW:zone.w};
                return;
            }
            if (zone.type === "gs") {
                const i = parseIdx(key);
                st._selIdx  = i;
                st._dragMode = "gs_slider";
                st._dragData = {slotIdx:i, startMx:mx, startVal:cd.slots[i].gradient_scale ?? 1.0, barW:zone.w};
                return;
            }
            return;
        }

        if (hit.type === "move") {
            st._selIdx  = hit.slotIdx;
            const slot  = cd.slots[hit.slotIdx];
            const ca    = st._caArea;
            st._dragMode = "move";
            st._dragData = {slotIdx:hit.slotIdx, startMx:mx, startMy:my,
                startX:slot.x ?? 0.5, startY:slot.y ?? 0.5, caW:ca.w, caH:ca.h};
            redraw(); return;
        }

        if (hit.type === "rotate") {
            const ca = st._caArea;
            const tf = getSlotTransform(cd.slots[hit.slotIdx], ca);
            st._dragMode = "rotate";
            st._dragData = {
                slotIdx:    hit.slotIdx,
                cx: tf.cx,  cy: tf.cy,
                startRot:   cd.slots[hit.slotIdx].rotation || 0,
                startAngle: Math.atan2(my - tf.cy, mx - tf.cx),
            };
            return;
        }

        if (hit.type === "scale_u") {
            const slot = cd.slots[hit.slotIdx];
            const ca   = st._caArea;
            const tf   = getSlotTransform(slot, ca);
            st._dragMode = "scale_u";
            st._dragData = {slotIdx:hit.slotIdx, startScale:slot.scale||1.0,
                startDist:Math.max(dist2([mx,my],[tf.cx,tf.cy]), 2)};
            return;
        }

        if (hit.type === "scale_x" || hit.type === "scale_y") {
            const slot      = cd.slots[hit.slotIdx];
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
        const st = node._dcst;
        if (!st._dragMode) return;
        e.stopPropagation();
        const [mx, my] = getCanvasXY(e, canvas);
        const cd  = ensureComp(node);
        const dd  = st._dragData;

        if (st._dragMode === "resize") {
            const nw = Math.max(MIN_COMP, dd.startW + mx - dd.startMx);
            const nh = Math.max(MIN_COMP, dd.startH + my - dd.startMy);
            setDispDims(node, nw, nh); redraw(); return;
        }
        if (st._dragMode === "bgdepth_slider") {
            const v = Math.max(0, Math.min(0.5, dd.startVal + (mx - dd.startMx) / dd.barW * 0.5));
            cd.background_depth = v; setComp(node, cd); redraw(); return;
        }
        if (st._dragMode === "dp_slider") {
            const v = Math.max(0, Math.min(1, dd.startVal + (mx - dd.startMx) / dd.barW));
            cd.slots[dd.slotIdx].depth_placement = v; setComp(node, cd); redraw(); return;
        }
        if (st._dragMode === "gs_slider") {
            const v = Math.max(0, Math.min(1, dd.startVal + (mx - dd.startMx) / dd.barW));
            cd.slots[dd.slotIdx].gradient_scale = v; setComp(node, cd); redraw(); return;
        }
        if (st._dragMode === "move") {
            cd.slots[dd.slotIdx].x = Math.max(0, Math.min(1, dd.startX + (mx-dd.startMx)/dd.caW));
            cd.slots[dd.slotIdx].y = Math.max(0, Math.min(1, dd.startY + (my-dd.startMy)/dd.caH));
            setComp(node, cd); redraw(); return;
        }
        if (st._dragMode === "rotate") {
            const currentAngle = Math.atan2(my - dd.cy, mx - dd.cx);
            const delta = (currentAngle - dd.startAngle) * 180 / Math.PI;
            let newRot = dd.startRot + delta;
            // Keep in -180..180 range
            newRot = ((newRot + 180) % 360 + 360) % 360 - 180;
            cd.slots[dd.slotIdx].rotation = newRot;
            setComp(node, cd); redraw(); return;
        }
        if (st._dragMode === "scale_u") {
            const slot = cd.slots[dd.slotIdx];
            const ca   = node._dcst._caArea;
            const tf   = getSlotTransform(slot, ca);
            const r    = dist2([mx,my],[tf.cx,tf.cy]) / dd.startDist;
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

    canvas.addEventListener("mouseup",    e => { node._dcst._dragMode = null; e.stopPropagation(); });
    canvas.addEventListener("mouseleave", e => { node._dcst._dragMode = null; });
    canvas.addEventListener("contextmenu", e => e.preventDefault());
    canvas.addEventListener("wheel", e => e.stopPropagation());

    // Double-click on slider resets to default
    canvas.addEventListener("dblclick", e => {
        e.stopPropagation();
        const [mx, my] = getCanvasXY(e, canvas);
        const st = node._dcst;
        const cd = ensureComp(node);
        for (const [key, zone] of Object.entries(st._zones || {})) {
            if (Array.isArray(zone) || !_hr(mx, my, zone)) continue;
            if (zone.type === "dp") {
                const i = parseInt(key.split("_").pop());
                cd.slots[i].depth_placement = 0.7;
                setComp(node, cd); redraw(); return;
            }
            if (zone.type === "gs") {
                const i = parseInt(key.split("_").pop());
                cd.slots[i].gradient_scale = 1.0;
                setComp(node, cd); redraw(); return;
            }
            if (key === "ctrl_bgdepth") {
                cd.background_depth = 0.10;
                setComp(node, cd); redraw(); return;
            }
        }
    });

    // Reload slot images after Python executes.
    // Use both node.onExecuted AND app.api 'executed' event as belt-and-suspenders
    // (non-output nodes may not reliably trigger node.onExecuted in all ComfyUI versions).
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

    // Widget hiding — use retry loop like pose_composer to handle async widget init
    function tryHide(n) {
        const w = _gw(node, W_COMP);
        // Only hide visually — do NOT set type="converted-widget" as that
        // prevents the widget value from being serialized and sent to Python.
        if (w) { w.computeSize = () => [0,-4]; w.draw = () => {}; }
        if (n < 12) setTimeout(() => tryHide(n + 1), 200);
        else app.graph?.setDirtyCanvas(true, true);
    }
    setTimeout(() => tryHide(0), 80);

    widget = node.addDOMWidget("_depth_canvas", "custom", wrapper, {
        serialize:  false,
        hideOnZoom: false,
        getValue:   () => "",
        setValue:   () => {},
    });
    widget.computeSize = (w) => [w, RATIO_BAR_H + w + LIST_H + CTRL_BAR_H + 10];

    setTimeout(() => loadAllImages(node), 400);
    return widget;
}

// ── Extension registration ────────────────────────────────────────────────────
app.registerExtension({
    name: "Eric.DepthComposer",
    async nodeCreated(node) {
        if (node.comfyClass !== NODE_TYPE) return;
        if (!node.size || node.size[0] < 300) node.size = [800, RATIO_BAR_H + 800 + LIST_H + CTRL_BAR_H + 10];
        setup(node);
    },
});
