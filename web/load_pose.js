/**
 * load_pose.js  v1 - Eric_Composer_Studio
 *
 * Adds a scrollable thumbnail gallery to the Load Pose Keypoint node.
 * Clicking a thumbnail selects that file by setting the filename widget.
 * The gallery refreshes automatically when folder_path changes.
 */

import { app } from "../../scripts/app.js";

const NODE_TYPE  = "EricLoadPoseKeypoint";
const THUMB_W    = 82;
const THUMB_H    = 82;
const THUMB_GAP  = 6;
const GALLERY_H  = 272;   // max-height of the scrollable grid
const HEADER_H   = 55;    // two-row header + wrapper padding

// BODY_LIMBS order matches controlnet_aux limbSeq / pose_composer.js exactly.
const BODY_LIMBS = [
    [1,2],[1,5],[2,3],[3,4],[5,6],[6,7],
    [1,8],[8,9],[9,10],[1,11],[11,12],[12,13],
    [1,0],[0,14],[14,16],[0,15],[15,17],
    [8,11],   // idx 17 = pelvis (skipped in DWPose rendering)
];

// Standard DWPose limb colours (RGB) matching controlnet_aux training colours.
const LIMB_COLORS = [
    "#FF0000","#FF5500","#FFAA00","#FFFF00","#AAFF00","#55FF00",
    "#00FF00","#00FF55","#00FFAA","#00FFFF","#00AAFF","#0055FF",
    "#0000FF","#5500FF","#AA00FF","#FF00FF","#FF00AA","#FF0055",
];

// ── Thumbnail renderer ───────────────────────────────────────────────────────
function renderThumb(canvas, entry) {
    const dpr = window.devicePixelRatio || 1;
    canvas.width  = THUMB_W * dpr;
    canvas.height = THUMB_H * dpr;
    canvas.style.width  = THUMB_W + "px";
    canvas.style.height = THUMB_H + "px";

    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = "#06060F";
    ctx.fillRect(0, 0, THUMB_W, THUMB_H);

    const pW  = entry.canvas_width  || 1024;
    const pH  = entry.canvas_height || 1024;
    const pad = 5;
    const sc  = Math.min((THUMB_W - pad * 2) / pW, (THUMB_H - pad * 2) / pH);
    const ox  = (THUMB_W - pW * sc) / 2;
    const oy  = (THUMB_H - pH * sc) / 2;
    const mp  = (x, y) => [ox + x * sc, oy + y * sc];

    for (const person of (entry.people || [])) {
        const flat = person.pose_keypoints_2d || [];
        const kps  = [];
        for (let i = 0; i < 54; i += 3)
            kps.push({ x: flat[i] || 0, y: flat[i + 1] || 0, c: flat[i + 2] || 0 });

        BODY_LIMBS.forEach(([a, b], idx) => {
            if (idx === 17) return;   // skip pelvis in DWPose mode
            const ka = kps[a], kb = kps[b];
            if (!ka || !kb || ka.c <= 0 || kb.c <= 0) return;
            const [ax, ay] = mp(ka.x, ka.y), [bx, by] = mp(kb.x, kb.y);
            ctx.strokeStyle = LIMB_COLORS[idx] || "#888";
            ctx.lineWidth   = 1.5;
            ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
        });
    }

    // Draw person count badge if > 1
    const n = (entry.people || []).length;
    if (n > 1) {
        ctx.fillStyle    = "#00000099";
        ctx.fillRect(THUMB_W - 18, 2, 16, 13);
        ctx.fillStyle    = "#88AACC";
        ctx.font         = "bold 9px monospace";
        ctx.textAlign    = "right";
        ctx.textBaseline = "top";
        ctx.fillText(`×${n}`, THUMB_W - 3, 3);
    }
}

// ── Gallery setup ────────────────────────────────────────────────────────────
function setup(node) {
    const gw = (name) => node.widgets?.find(w => w.name === name);

    const wrapper = document.createElement("div");
    wrapper.style.cssText = [
        "width:100%", "box-sizing:border-box", "background:#09091A",
        "border:1px solid #1E2A38", "border-radius:3px", "padding:5px 8px 6px 8px",
        "user-select:none",
    ].join(";");

    // ── header ───────────────────────────────────────────────────────────────
    const header = document.createElement("div");
    header.style.cssText = "display:flex;flex-direction:column;gap:3px;margin-bottom:5px;";

    const headerRow1 = document.createElement("div");
    headerRow1.style.cssText = "display:flex;align-items:center;gap:6px;";

    const title = document.createElement("span");
    title.textContent = "Pose Gallery";
    title.style.cssText = "color:#446688;font:11px monospace;flex:1;pointer-events:none;";

    const statusEl = document.createElement("span");
    statusEl.style.cssText = "color:#334455;font:10px monospace;white-space:nowrap;";

    const refreshBtn = document.createElement("button");
    refreshBtn.textContent = "⟳ Refresh";
    refreshBtn.style.cssText = [
        "background:#0A1A0A", "color:#44AA44", "border:1px solid #226622",
        "border-radius:3px", "font:11px monospace", "padding:1px 7px",
        "cursor:pointer", "pointer-events:all", "white-space:nowrap",
    ].join(";");

    headerRow1.appendChild(title);
    headerRow1.appendChild(statusEl);
    headerRow1.appendChild(refreshBtn);

    const headerRow2 = document.createElement("div");
    headerRow2.style.cssText = "display:flex;align-items:center;gap:4px;min-height:18px;";

    const upBtn = document.createElement("button");
    upBtn.textContent = "↑ Up";
    upBtn.title = "Go to parent folder";
    upBtn.style.cssText = [
        "background:#12121E", "color:#446688", "border:1px solid #2A3A4A",
        "border-radius:3px", "font:10px monospace", "padding:1px 6px",
        "cursor:pointer", "pointer-events:all", "white-space:nowrap", "flex-shrink:0",
    ].join(";");

    const pathEl = document.createElement("span");
    pathEl.style.cssText = [
        "color:#2A4A6A", "font:10px monospace", "overflow:hidden",
        "text-overflow:ellipsis", "white-space:nowrap", "flex:1",
    ].join(";");

    headerRow2.appendChild(upBtn);
    headerRow2.appendChild(pathEl);

    header.appendChild(headerRow1);
    header.appendChild(headerRow2);
    wrapper.appendChild(header);

    // ── scrollable thumbnail grid ─────────────────────────────────────────────
    const grid = document.createElement("div");
    grid.style.cssText = [
        "display:flex", "flex-wrap:wrap", `gap:${THUMB_GAP}px`,
        `max-height:${GALLERY_H}px`, "overflow-y:auto",
        "padding:2px", "box-sizing:border-box",
    ].join(";");
    wrapper.appendChild(grid);

    // ── navigation helpers ────────────────────────────────────────────────────
    function currentFolder() {
        return (gw("folder_path")?.value || "poses").trim();
    }
    function navigateTo(folderPath) {
        const folderW = gw("folder_path");
        if (!folderW) return;
        folderW.value = folderPath;
        folderW.callback?.(folderPath);
        setTimeout(refresh, 20);
    }
    function navigateUp() {
        const cur = currentFolder().replace(/\\/g, "/");
        const slash = cur.lastIndexOf("/");
        if (slash > 0) navigateTo(cur.substring(0, slash));
    }

    // ── refresh logic ─────────────────────────────────────────────────────────
    async function refresh() {
        const filenameW = gw("filename");
        const folder    = currentFolder();

        const normalized = folder.replace(/\\/g, "/");
        const hasParent  = normalized.includes("/");
        pathEl.textContent  = folder;
        upBtn.style.opacity = hasParent ? "1" : "0.3";
        upBtn.style.cursor  = hasParent ? "pointer" : "default";

        grid.innerHTML       = "";
        statusEl.textContent = "…";

        let subdirs = [], files = [];
        try {
            const resp = await fetch(
                `/eric_composer_studio/list_poses?folder=${encodeURIComponent(folder)}`
            );
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            const data = await resp.json();
            if (Array.isArray(data)) { files = data; }   // legacy format
            else { subdirs = data.subdirs || []; files = data.files || []; }
        } catch (err) {
            statusEl.textContent = `⚠ ${err.message}`;
            return;
        }

        const total = subdirs.length + files.length;
        statusEl.textContent = total
            ? `${files.length} file${files.length !== 1 ? "s" : ""}` +
              (subdirs.length ? `, ${subdirs.length} folder${subdirs.length !== 1 ? "s" : ""}` : "")
            : "empty";

        // ── folder tiles ─────────────────────────────────────────────────────
        for (const dirname of subdirs) {
            const item = document.createElement("div");
            item.style.cssText = [
                "display:flex", "flex-direction:column", "align-items:center",
                "cursor:pointer", "padding:2px", "border-radius:4px",
                "border:2px solid transparent", "box-sizing:border-box", "flex-shrink:0",
            ].join(";");

            const cv = document.createElement("canvas");
            const dpr = window.devicePixelRatio || 1;
            cv.width = THUMB_W * dpr; cv.height = THUMB_H * dpr;
            cv.style.width = THUMB_W + "px"; cv.style.height = THUMB_H + "px";
            const ctx = cv.getContext("2d");
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
            ctx.fillStyle = "#06060F"; ctx.fillRect(0, 0, THUMB_W, THUMB_H);
            // folder body
            const fx = 12, fy = 22, fw = THUMB_W - 24, fh = THUMB_H - 38;
            ctx.fillStyle = "#1A2A3A"; ctx.strokeStyle = "#2A5A8A"; ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.moveTo(fx, fy + 9); ctx.lineTo(fx + fw * 0.38, fy + 9);
            ctx.lineTo(fx + fw * 0.38 + 7, fy); ctx.lineTo(fx + fw * 0.68, fy);
            ctx.lineTo(fx + fw * 0.68, fy + 9); ctx.lineTo(fx + fw, fy + 9);
            ctx.lineTo(fx + fw, fy + fh); ctx.lineTo(fx, fy + fh);
            ctx.closePath(); ctx.fill(); ctx.stroke();
            ctx.font = "22px sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
            ctx.fillText("📁", THUMB_W / 2, THUMB_H / 2 + 6);

            const lbl = document.createElement("div");
            lbl.style.cssText = [
                `width:${THUMB_W}px`, "font:9px monospace", "color:#3A6A9A",
                "text-align:center", "overflow:hidden", "text-overflow:ellipsis",
                "white-space:nowrap", "margin-top:2px", "line-height:1.3", "pointer-events:none",
            ].join(";");
            lbl.title = dirname; lbl.textContent = dirname;
            item.appendChild(cv); item.appendChild(lbl);

            item.addEventListener("mouseenter", () => { item.style.borderColor = "#2A5A8A"; lbl.style.color = "#5599CC"; });
            item.addEventListener("mouseleave", () => { item.style.borderColor = "transparent"; lbl.style.color = "#3A6A9A"; });
            item.addEventListener("click", () => {
                const base = currentFolder().replace(/\\/g, "/");
                navigateTo(base.endsWith("/") ? base + dirname : base + "/" + dirname);
            });
            grid.appendChild(item);
        }

        // ── file tiles ────────────────────────────────────────────────────────
        const curFile = filenameW?.value || "";
        for (const entry of files) {
            const item = document.createElement("div");
            item.dataset.filename = entry.filename;
            item.style.cssText = [
                "display:flex", "flex-direction:column", "align-items:center",
                "cursor:pointer", "padding:2px", "border-radius:4px",
                "border:2px solid transparent", "box-sizing:border-box", "flex-shrink:0",
            ].join(";");

            const cv = document.createElement("canvas");
            renderThumb(cv, entry);

            const lbl = document.createElement("div");
            lbl.style.cssText = [
                `width:${THUMB_W}px`, "font:9px monospace", "color:#4A7A9A",
                "text-align:center", "overflow:hidden", "text-overflow:ellipsis",
                "white-space:nowrap", "margin-top:2px", "line-height:1.3", "pointer-events:none",
            ].join(";");
            lbl.title = entry.filename;
            lbl.textContent = entry.name
                || entry.filename
                    .replace(/^\d{8}_\d{6}_/, "")
                    .replace(/\.json$/i, "")
                    .replace(/_/g, " ");

            item.appendChild(cv); item.appendChild(lbl);

            item.addEventListener("mouseenter", () => {
                if ((filenameW?.value || "") !== entry.filename) item.style.borderColor = "#2A4A6A";
            });
            item.addEventListener("mouseleave", () => {
                if ((filenameW?.value || "") !== entry.filename) item.style.borderColor = "transparent";
            });
            item.addEventListener("click", () => {
                if (filenameW) { filenameW.value = entry.filename; filenameW.callback?.(entry.filename); }
                grid.querySelectorAll("[data-filename]").forEach(el => {
                    const on = el.dataset.filename === entry.filename;
                    el.style.borderColor = on ? "#5599FF" : "transparent";
                    const l = el.querySelector("div"); if (l) l.style.color = on ? "#88BBFF" : "#4A7A9A";
                });
            });
            if (entry.filename === curFile) { item.style.borderColor = "#5599FF"; lbl.style.color = "#88BBFF"; }
            grid.appendChild(item);
        }
    }

    upBtn.addEventListener("click", () => navigateUp());
    refreshBtn.addEventListener("click", () => {
        refreshBtn.textContent = "⟳ …";
        refresh().finally(() => { refreshBtn.textContent = "⟳ Refresh"; });
    });

    // Auto-refresh when folder_path widget is edited
    const folderW = gw("folder_path");
    if (folderW) {
        const origCB = folderW.callback;
        folderW.callback = function (...args) {
            origCB?.(...args);
            setTimeout(refresh, 80);
        };
    }

    node._poseGalleryRefresh = refresh;

    const widget = node.addDOMWidget("_pose_gallery", "custom", wrapper, {
        serialize:   false,
        hideOnZoom:  false,
        getValue:    () => "",
        setValue:    () => {},
    });
    widget.computeSize = (w) => [w, HEADER_H + GALLERY_H + 10];

    setTimeout(refresh, 350);
    return widget;
}

app.registerExtension({
    name: "Eric.LoadPoseKeypoint",
    async nodeCreated(node) {
        if (node.comfyClass !== NODE_TYPE) return;
        if (!node.size || node.size[0] < 300) node.size = [360, 490];
        setup(node);
    },
});
