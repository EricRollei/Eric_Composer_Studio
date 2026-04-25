/*
 * text_layer.js  v1 - Eric_Composer_Studio
 *
 * Augments the EricTextLayer node with native color-picker swatches for the
 * color / gradient_color / stroke_color string widgets. The original text
 * inputs remain so users can still paste hex values directly.
 */

import { app } from "../../scripts/app.js";

const COLOR_WIDGETS = ["color", "gradient_color", "stroke_color"];

// Normalise to "#RRGGBB" so <input type=color> accepts the value.
// Accepts: "#RGB", "#RRGGBB", "rgb(...)", named colors. Unknown strings → "#ffffff".
function normalizeHex(value) {
    if (typeof value !== "string") return "#ffffff";
    const v = value.trim();
    if (/^#[0-9a-f]{6}$/i.test(v)) return v.toLowerCase();
    if (/^#[0-9a-f]{3}$/i.test(v)) {
        const r = v[1], g = v[2], b = v[3];
        return ("#" + r + r + g + g + b + b).toLowerCase();
    }
    // Use the browser to resolve named colors / rgb() etc. by setting a
    // hidden element's style and reading back the computed RGB.
    try {
        const probe = document.createElement("div");
        probe.style.color = v;
        document.body.appendChild(probe);
        const computed = getComputedStyle(probe).color;
        document.body.removeChild(probe);
        const m = computed.match(/^rgba?\((\d+),\s*(\d+),\s*(\d+)/);
        if (m) {
            const r = parseInt(m[1]).toString(16).padStart(2, "0");
            const g = parseInt(m[2]).toString(16).padStart(2, "0");
            const b = parseInt(m[3]).toString(16).padStart(2, "0");
            return ("#" + r + g + b).toLowerCase();
        }
    } catch (_) { /* ignore */ }
    return "#ffffff";
}

function buildPickerRow(node, widgetName) {
    const widget = node.widgets?.find(w => w.name === widgetName);
    if (!widget) return null;

    const wrap = document.createElement("div");
    wrap.style.cssText =
        "display:flex;align-items:center;gap:6px;width:100%;padding:2px 4px;" +
        "box-sizing:border-box;font:11px sans-serif;color:#ddd;";

    const label = document.createElement("span");
    label.textContent = widgetName;
    label.style.cssText = "flex:0 0 90px;opacity:0.85;";
    wrap.appendChild(label);

    const picker = document.createElement("input");
    picker.type = "color";
    picker.value = normalizeHex(widget.value);
    picker.style.cssText =
        "flex:0 0 28px;height:20px;padding:0;border:1px solid #444;" +
        "background:transparent;cursor:pointer;border-radius:3px;";
    wrap.appendChild(picker);

    const hex = document.createElement("input");
    hex.type = "text";
    hex.value = widget.value || "#ffffff";
    hex.spellcheck = false;
    hex.style.cssText =
        "flex:1 1 auto;height:20px;min-width:0;padding:0 6px;" +
        "background:#1d1d1d;color:#eee;border:1px solid #444;border-radius:3px;" +
        "font:11px monospace;";
    wrap.appendChild(hex);

    // picker -> writes hex back to the widget + the text input
    picker.addEventListener("input", () => {
        const v = picker.value.toLowerCase();
        widget.value = v;
        hex.value = v;
        widget.callback?.(v);
        node.graph?.setDirtyCanvas(true, true);
    });

    // text -> updates widget; if the typed text resolves to a colour the
    // swatch follows, otherwise we still let the user keep editing.
    const commitHex = () => {
        const raw = hex.value.trim();
        widget.value = raw;
        picker.value = normalizeHex(raw);
        widget.callback?.(raw);
        node.graph?.setDirtyCanvas(true, true);
    };
    hex.addEventListener("change", commitHex);
    hex.addEventListener("blur", commitHex);
    hex.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
            e.preventDefault();
            commitHex();
            hex.blur();
        }
    });

    // Keep picker / text in sync if something else changes the widget value
    // (e.g. workflow load).
    const origCallback = widget.callback;
    widget.callback = function (v) {
        if (typeof v === "string") {
            hex.value = v;
            picker.value = normalizeHex(v);
        }
        return origCallback?.apply(this, arguments);
    };

    return { wrap, picker, hex, widget };
}

app.registerExtension({
    name: "EricComposerStudio.TextLayerColorPicker",

    nodeCreated(node) {
        if (node.comfyClass !== "EricTextLayer") return;

        // Defer to the next frame so all default widgets exist.
        requestAnimationFrame(() => {
            for (const name of COLOR_WIDGETS) {
                const built = buildPickerRow(node, name);
                if (!built) continue;

                // Add a DOM widget. ComfyUI's addDOMWidget will manage layout.
                node.addDOMWidget(`${name}__picker`, "ericColorPicker", built.wrap, {
                    serialize: false,
                    hideOnZoom: false,
                });

                // Hide the underlying text widget visually while keeping its
                // value (so the workflow round-trips the same way it did
                // before). We do this by setting its computed type so
                // LiteGraph skips drawing it.
                const w = built.widget;
                if (w && !w.__ericHidden) {
                    w.__ericHidden = true;
                    w.computeSize = () => [0, -4]; // collapses the row
                    w.type = "hidden";              // LiteGraph skips drawing
                }
            }
            node.setSize?.(node.computeSize());
            node.graph?.setDirtyCanvas(true, true);
        });
    },
});
