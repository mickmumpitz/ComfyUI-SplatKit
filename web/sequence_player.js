// Puts the splat player inside the SplatKit Player node.
//
// The node renders nothing: it registers the trained folder with a small route and
// returns a token. This drops an <iframe> on the node pointing the vendored player at that
// folder. The player is a normal web page, so it keeps its own controls, transport and
// WebGL context.
//
// The extension base is taken from this file's own URL rather than a fixed folder name, so
// the pack works whatever the custom_nodes folder is called.

import { app } from "../../scripts/app.js";

const BASE = new URL("./", import.meta.url).pathname;
const PLAYER = `${BASE}player/index.html`;
const NODE = "SplatKit_SequencePlayer";
const MIN_HEIGHT = 320;

function playerWidget(node, token) {
    let widget = node.widgets?.find((w) => w.name === "splatkit_player");
    if (widget) {
        if (widget.__token !== token) {
            widget.__token = token;
            widget.element.src = `${PLAYER}?dir=/splatkit/seq/${token}/`;
        }
        return widget;
    }
    const frame = document.createElement("iframe");
    frame.src = `${PLAYER}?dir=/splatkit/seq/${token}/`;
    Object.assign(frame.style, {
        border: "0", borderRadius: "6px", width: "100%", height: "100%", background: "#000",
    });
    widget = node.addDOMWidget("splatkit_player", "splatkit_player", frame, {
        serialize: false,      // a viewer, not an input
        hideOnZoom: false,
    });
    widget.__token = token;
    if (node.size[1] < MIN_HEIGHT) {
        node.setSize([Math.max(node.size[0], 360), MIN_HEIGHT]);
    }
    return widget;
}

app.registerExtension({
    name: "SplatKit.SequencePlayer",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE) return;
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            const payload = message?.splatkit?.[0];
            if (!payload?.token) return;
            playerWidget(this, payload.token);
            this.setDirtyCanvas(true, true);
        };
    },
});
