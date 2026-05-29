"""Interactive workflow graph components for the Streamlit UI."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from typing import Any

from ui.message_utils import normalize_message_content


TEXT_LIMIT = 12000
NODE_WIDTH = 164
NODE_HEIGHT = 66
NODE_GAP = 188
CANVAS_PADDING_X = 28
CANVAS_HEIGHT = 360


def build_workflow_graph(query: str, messages: list[Any]) -> dict[str, Any]:
    """Build a compact graph model from one ChemGraph exchange.

    The UI stores each agent run as a list of LangChain-style messages. This
    model turns the user query, assistant turns, tool-call requests, and tool
    results into clickable nodes with explicit input and output text.
    """
    query_text = normalize_message_content(query).strip()
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, str]] = []
    nodes_by_id: dict[str, dict[str, Any]] = {}
    pending_tool_nodes: dict[str, str] = {}

    def add_node(
        *,
        kind: str,
        label: str,
        subtitle: str,
        input_text: Any,
        output_text: Any,
        meta: Mapping[str, Any] | None = None,
    ) -> str:
        node_id = f"node-{len(nodes)}"
        node = {
            "id": node_id,
            "kind": kind,
            "label": _compact_label(label),
            "title": str(label or kind.title()),
            "subtitle": _compact_label(subtitle, limit=34),
            "input": _format_detail(input_text),
            "output": _format_detail(output_text),
            "meta": _jsonable(dict(meta or {})),
            "x": CANVAS_PADDING_X + len(nodes) * NODE_GAP,
            "y": _node_y(kind),
        }
        nodes.append(node)
        nodes_by_id[node_id] = node
        return node_id

    def add_edges(sources: Sequence[str], target: str, label: str = "") -> None:
        for source in sources:
            if source and source != target:
                edges.append({"source": source, "target": target, "label": label})

    query_id = add_node(
        kind="query",
        label="Query",
        subtitle="user input",
        input_text="",
        output_text=query_text,
        meta={"role": "human"},
    )
    ready_sources: list[str] = [query_id]
    skipped_initial_query = False

    for message_index, message in enumerate(messages):
        msg_type = _message_type(message)
        name = _message_name(message)
        content = normalize_message_content(_message_content(message)).strip()
        tool_calls = _extract_tool_calls(message)

        if (
            msg_type == "human"
            and not skipped_initial_query
            and _same_text(content, query_text)
        ):
            skipped_initial_query = True
            continue

        if msg_type == "tool":
            tool_call_id = _message_attr(message, "tool_call_id")
            tool_node_id = (
                pending_tool_nodes.get(str(tool_call_id)) if tool_call_id else None
            )
            if tool_node_id and tool_node_id in nodes_by_id:
                node = nodes_by_id[tool_node_id]
                node["output"] = _format_detail(content or "(empty tool result)")
                node["meta"].update(
                    _jsonable(
                        {
                            "message_index": message_index,
                            "role": msg_type,
                            "tool_call_id": tool_call_id,
                            "tool_name": name or node.get("title"),
                        }
                    )
                )
            else:
                label = name or "Tool result"
                tool_node_id = add_node(
                    kind="tool",
                    label=label,
                    subtitle="tool result",
                    input_text={"tool_call_id": tool_call_id, "tool_name": name},
                    output_text=content or "(empty tool result)",
                    meta={
                        "message_index": message_index,
                        "role": msg_type,
                        "tool_call_id": tool_call_id,
                        "tool_name": name,
                    },
                )
                add_edges(ready_sources, tool_node_id, "result")
                ready_sources = [tool_node_id]
            continue

        if msg_type == "human":
            human_id = add_node(
                kind="query",
                label="Human input",
                subtitle="follow-up",
                input_text=_source_outputs(nodes_by_id, ready_sources),
                output_text=content,
                meta={"message_index": message_index, "role": msg_type},
            )
            add_edges(ready_sources, human_id, "input")
            ready_sources = [human_id]
            continue

        if msg_type == "ai" or tool_calls:
            output_text = _assistant_output(content, tool_calls)
            assistant_id = add_node(
                kind="assistant",
                label="Assistant",
                subtitle=(
                    _pluralize_tool_calls(len(tool_calls))
                    if tool_calls
                    else "response"
                ),
                input_text=_source_outputs(nodes_by_id, ready_sources),
                output_text=output_text,
                meta={
                    "message_index": message_index,
                    "role": msg_type,
                    "tool_call_count": len(tool_calls),
                },
            )
            add_edges(ready_sources, assistant_id, "message")

            if not tool_calls:
                ready_sources = [assistant_id]
                continue

            tool_node_ids: list[str] = []
            for call_index, call in enumerate(tool_calls, start=1):
                tool_name = str(call.get("name") or f"tool {call_index}")
                tool_call_id = str(
                    call.get("id") or f"message-{message_index}-tool-{call_index}"
                )
                tool_node_id = add_node(
                    kind="tool",
                    label=tool_name,
                    subtitle="tool call",
                    input_text=call.get("args"),
                    output_text="Waiting for tool result.",
                    meta={
                        "message_index": message_index,
                        "role": "tool_call",
                        "tool_call_id": tool_call_id,
                        "tool_name": tool_name,
                    },
                )
                add_edges([assistant_id], tool_node_id, "calls")
                pending_tool_nodes[tool_call_id] = tool_node_id
                tool_node_ids.append(tool_node_id)
            ready_sources = tool_node_ids
            continue

        if content:
            message_id = add_node(
                kind="message",
                label=name or msg_type or "Message",
                subtitle=msg_type or type(message).__name__,
                input_text=_source_outputs(nodes_by_id, ready_sources),
                output_text=content,
                meta={"message_index": message_index, "role": msg_type, "name": name},
            )
            add_edges(ready_sources, message_id, "message")
            ready_sources = [message_id]

    width = max(760, CANVAS_PADDING_X * 2 + max(len(nodes), 1) * NODE_GAP)
    return {
        "query": query_text,
        "nodes": nodes,
        "edges": edges,
        "canvas": {"width": width, "height": CANVAS_HEIGHT},
        "metrics": {
            "node_count": len(nodes),
            "tool_count": sum(1 for node in nodes if node["kind"] == "tool"),
            "assistant_count": sum(
                1 for node in nodes if node["kind"] == "assistant"
            ),
        },
    }


def render_workflow_graph(query: str, messages: list[Any], *, key: str) -> None:
    """Render the interactive graph in Streamlit."""
    import streamlit as st

    graph = build_workflow_graph(query, messages)
    html = _workflow_graph_html(graph, key=key)
    st.components.v1.html(html, height=520, scrolling=False)


def _workflow_graph_html(graph: dict[str, Any], *, key: str) -> str:
    root_key = re.sub(r"[^a-zA-Z0-9_-]+", "_", key).strip("_") or "workflow"
    root_id = f"cg_workflow_{root_key}"
    payload = json.dumps(graph, default=str).replace("</", "<\\/")

    template = r"""
<div id="__ROOT_ID__" class="cg-workflow">
  <style>
    #__ROOT_ID__ {
      --cg-bg: #f7f9fb;
      --cg-panel: #ffffff;
      --cg-border: #d8e1ec;
      --cg-ink: #182433;
      --cg-muted: #68778a;
      --cg-query: #207868;
      --cg-query-bg: #e7f5f0;
      --cg-assistant: #315f9b;
      --cg-assistant-bg: #e8f0fb;
      --cg-tool: #9b620f;
      --cg-tool-bg: #fff1d8;
      --cg-message: #6f5a90;
      --cg-message-bg: #f0ecf7;
      color: var(--cg-ink);
      font: 13px/1.42 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    #__ROOT_ID__ * { box-sizing: border-box; }
    #__ROOT_ID__ .cg-shell {
      display: grid;
      grid-template-columns: minmax(0, 1.6fr) minmax(300px, 0.9fr);
      gap: 10px;
      height: 500px;
    }
    #__ROOT_ID__ .cg-map,
    #__ROOT_ID__ .cg-inspector {
      border: 1px solid var(--cg-border);
      border-radius: 8px;
      background: var(--cg-panel);
      min-width: 0;
    }
    #__ROOT_ID__ .cg-map {
      position: relative;
      overflow: hidden;
      background:
        linear-gradient(#edf2f7 1px, transparent 1px),
        linear-gradient(90deg, #edf2f7 1px, transparent 1px),
        var(--cg-bg);
      background-size: 28px 28px;
      cursor: grab;
      touch-action: none;
      user-select: none;
    }
    #__ROOT_ID__ .cg-map.panning {
      cursor: grabbing;
    }
    #__ROOT_ID__ .cg-zoom {
      position: absolute;
      top: 8px;
      right: 8px;
      z-index: 4;
      display: flex;
      gap: 4px;
      padding: 4px;
      border: 1px solid var(--cg-border);
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.92);
      box-shadow: 0 6px 18px rgba(24, 36, 51, 0.08);
    }
    #__ROOT_ID__ .cg-zoom button {
      width: 28px;
      height: 26px;
      border: 1px solid #cfd8e5;
      border-radius: 6px;
      background: #ffffff;
      color: var(--cg-ink);
      font: 700 12px/1 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      cursor: pointer;
    }
    #__ROOT_ID__ .cg-zoom button:hover {
      background: #f2f6fb;
    }
    #__ROOT_ID__ .cg-canvas {
      position: relative;
      width: __WIDTH__px;
      height: __HEIGHT__px;
      transform-origin: 0 0;
      will-change: transform;
    }
    #__ROOT_ID__ svg {
      position: absolute;
      inset: 0;
      width: __WIDTH__px;
      height: __HEIGHT__px;
      pointer-events: none;
    }
    #__ROOT_ID__ .cg-node {
      position: absolute;
      width: __NODE_WIDTH__px;
      height: __NODE_HEIGHT__px;
      border: 1.5px solid var(--cg-border);
      border-radius: 8px;
      padding: 8px 9px;
      background: #ffffff;
      color: var(--cg-ink);
      text-align: left;
      cursor: pointer;
      box-shadow: 0 7px 18px rgba(24, 36, 51, 0.08);
      overflow: hidden;
    }
    #__ROOT_ID__ .cg-node:hover { filter: brightness(0.985); }
    #__ROOT_ID__ .cg-node.selected {
      outline: 3px solid rgba(49, 95, 155, 0.24);
      border-width: 2px;
    }
    #__ROOT_ID__ .cg-node.query {
      border-color: var(--cg-query);
      background: var(--cg-query-bg);
    }
    #__ROOT_ID__ .cg-node.assistant {
      border-color: var(--cg-assistant);
      background: var(--cg-assistant-bg);
    }
    #__ROOT_ID__ .cg-node.tool {
      border-color: var(--cg-tool);
      background: var(--cg-tool-bg);
    }
    #__ROOT_ID__ .cg-node.message {
      border-color: var(--cg-message);
      background: var(--cg-message-bg);
    }
    #__ROOT_ID__ .cg-label {
      display: block;
      font-weight: 700;
      font-size: 12.5px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    #__ROOT_ID__ .cg-subtitle {
      display: block;
      margin-top: 4px;
      color: var(--cg-muted);
      font-size: 11.5px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    #__ROOT_ID__ .cg-inspector {
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
      overflow: hidden;
    }
    #__ROOT_ID__ .cg-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      border-bottom: 1px solid var(--cg-border);
      padding: 9px 11px;
      background: #fbfcfe;
      min-width: 0;
    }
    #__ROOT_ID__ .cg-head strong {
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    #__ROOT_ID__ .cg-pill {
      flex: 0 0 auto;
      border: 1px solid var(--cg-border);
      border-radius: 999px;
      padding: 2px 7px;
      color: var(--cg-muted);
      font-size: 11px;
      background: #ffffff;
    }
    #__ROOT_ID__ .cg-details {
      overflow: auto;
      padding: 10px 11px 12px;
    }
    #__ROOT_ID__ .cg-section {
      margin: 0 0 12px;
    }
    #__ROOT_ID__ .cg-section-title {
      margin: 0 0 5px;
      color: var(--cg-muted);
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0;
    }
    #__ROOT_ID__ pre {
      margin: 0;
      padding: 8px;
      border: 1px solid #e2e8f0;
      border-radius: 6px;
      background: #f8fafc;
      color: #17212f;
      white-space: pre-wrap;
      word-break: break-word;
      max-height: 170px;
      overflow: auto;
      font: 12px/1.42 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }
    #__ROOT_ID__ .cg-kv {
      display: grid;
      grid-template-columns: minmax(90px, 0.45fr) minmax(0, 1fr);
      gap: 5px 8px;
      margin: 0;
      font-size: 12px;
    }
    #__ROOT_ID__ .cg-kv dt {
      color: var(--cg-muted);
      overflow: hidden;
      text-overflow: ellipsis;
    }
    #__ROOT_ID__ .cg-kv dd {
      margin: 0;
      min-width: 0;
      overflow-wrap: anywhere;
    }
    @media (max-width: 760px) {
      #__ROOT_ID__ .cg-shell {
        grid-template-columns: 1fr;
        height: auto;
      }
      #__ROOT_ID__ .cg-map {
        height: 360px;
      }
      #__ROOT_ID__ .cg-inspector {
        min-height: 330px;
      }
    }
  </style>
  <div class="cg-shell">
    <div class="cg-map">
      <div class="cg-zoom" aria-label="Graph zoom controls">
        <button type="button" data-zoom="out" title="Zoom out">-</button>
        <button type="button" data-zoom="fit" title="Fit graph">Fit</button>
        <button type="button" data-zoom="in" title="Zoom in">+</button>
        <button type="button" data-zoom="reset" title="Reset view">1:1</button>
      </div>
      <div class="cg-canvas">
        <svg class="cg-edges" aria-hidden="true"></svg>
        <div class="cg-nodes"></div>
      </div>
    </div>
    <section class="cg-inspector" aria-label="Workflow node inspector">
      <div class="cg-head">
        <strong class="cg-title">Inspector</strong>
        <span class="cg-pill"></span>
      </div>
      <div class="cg-details"></div>
    </section>
  </div>
  <script>
    (() => {
      const root = document.getElementById("__ROOT_ID__");
      const data = __DATA__;
      const nodeWidth = __NODE_WIDTH__;
      const nodeHeight = __NODE_HEIGHT__;
      const nodesById = new Map(data.nodes.map((node) => [node.id, node]));
      const map = root.querySelector(".cg-map");
      const canvas = root.querySelector(".cg-canvas");
      const edgeLayer = root.querySelector(".cg-edges");
      const nodeLayer = root.querySelector(".cg-nodes");
      const title = root.querySelector(".cg-title");
      const pill = root.querySelector(".cg-pill");
      const details = root.querySelector(".cg-details");
      let selectedId = null;
      const storageKey = "__ROOT_ID___view";
      const view = loadView();
      let dragging = null;

      function esc(value) {
        return String(value ?? "").replace(/[&<>"']/g, (ch) => ({
          "&": "&amp;",
          "<": "&lt;",
          ">": "&gt;",
          '"': "&quot;",
          "'": "&#39;"
        }[ch]));
      }

      function pretty(value) {
        if (value === null || value === undefined || value === "") return "";
        if (typeof value === "string") return value;
        try {
          return JSON.stringify(value, null, 2);
        } catch (_err) {
          return String(value);
        }
      }

      function section(name, value) {
        const text = pretty(value);
        if (!text) return "";
        return `
          <div class="cg-section">
            <div class="cg-section-title">${esc(name)}</div>
            <pre>${esc(text)}</pre>
          </div>`;
      }

      function metadata(node) {
        const meta = node.meta || {};
        const entries = [
          ["type", node.kind],
          ["role", meta.role || node.subtitle],
          ["tool", meta.tool_name || ""],
          ["tool_call_id", meta.tool_call_id || ""],
          ["message_index", meta.message_index ?? ""]
        ].filter((row) => row[1] !== "");
        return `
          <div class="cg-section">
            <div class="cg-section-title">Node</div>
            <dl class="cg-kv">
              ${entries.map(([key, value]) => `<dt>${esc(key)}</dt><dd>${esc(value)}</dd>`).join("")}
            </dl>
          </div>`;
      }

      function renderEdges() {
        edgeLayer.innerHTML = `
          <defs>
            <marker id="__ROOT_ID___arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
              <path d="M0,0 L8,4 L0,8 Z" fill="#8b98a8"></path>
            </marker>
          </defs>`;
        for (const edge of data.edges) {
          const source = nodesById.get(edge.source);
          const target = nodesById.get(edge.target);
          if (!source || !target) continue;
          const x1 = source.x + nodeWidth;
          const y1 = source.y + nodeHeight / 2;
          const x2 = target.x;
          const y2 = target.y + nodeHeight / 2;
          const mid = Math.max(26, (x2 - x1) / 2);
          const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
          path.setAttribute("d", `M ${x1} ${y1} C ${x1 + mid} ${y1}, ${x2 - mid} ${y2}, ${x2} ${y2}`);
          path.setAttribute("fill", "none");
          path.setAttribute("stroke", "#8b98a8");
          path.setAttribute("stroke-width", "1.5");
          path.setAttribute("marker-end", "url(#__ROOT_ID___arrow)");
          edgeLayer.appendChild(path);
        }
      }

      function selectNode(id) {
        selectedId = id;
        const node = nodesById.get(id) || data.nodes[0];
        root.querySelectorAll(".cg-node").forEach((el) => {
          el.classList.toggle("selected", el.dataset.nodeId === node.id);
        });
        title.textContent = node.title || node.label || "Inspector";
        pill.textContent = node.kind || "";
        details.innerHTML = [
          metadata(node),
          section("Input", node.input),
          section("Output", node.output),
          section("Metadata", node.meta)
        ].join("");
      }

      function renderNodes() {
        nodeLayer.innerHTML = "";
        for (const node of data.nodes) {
          const button = document.createElement("button");
          button.type = "button";
          button.className = `cg-node ${node.kind || "message"}`;
          button.dataset.nodeId = node.id;
          button.style.left = `${node.x}px`;
          button.style.top = `${node.y}px`;
          button.innerHTML = `
            <span class="cg-label">${esc(node.label)}</span>
            <span class="cg-subtitle">${esc(node.subtitle)}</span>`;
          button.addEventListener("click", () => selectNode(node.id));
          nodeLayer.appendChild(button);
        }
      }

      function clamp(value, min, max) {
        return Math.max(min, Math.min(max, value));
      }

      function loadView() {
        try {
          const saved = JSON.parse(window.localStorage.getItem("__ROOT_ID___view") || "null");
          if (saved && Number.isFinite(saved.scale) && Number.isFinite(saved.x) && Number.isFinite(saved.y)) {
            return { scale: saved.scale, x: saved.x, y: saved.y };
          }
        } catch (_err) {
          // Ignore malformed localStorage state.
        }
        return { scale: 1, x: 8, y: 18 };
      }

      function saveView() {
        try {
          window.localStorage.setItem(storageKey, JSON.stringify(view));
        } catch (_err) {
          // localStorage can be unavailable in some embedded contexts.
        }
      }

      function applyView() {
        view.scale = clamp(view.scale, 0.35, 2.4);
        canvas.style.transform = `translate(${view.x}px, ${view.y}px) scale(${view.scale})`;
        saveView();
      }

      function zoomAt(clientX, clientY, factor) {
        const rect = map.getBoundingClientRect();
        const px = clientX - rect.left;
        const py = clientY - rect.top;
        const nextScale = clamp(view.scale * factor, 0.35, 2.4);
        view.x = px - ((px - view.x) * nextScale) / view.scale;
        view.y = py - ((py - view.y) * nextScale) / view.scale;
        view.scale = nextScale;
        applyView();
      }

      function fitGraph() {
        const scaleX = (map.clientWidth - 28) / Math.max(1, data.canvas.width);
        const scaleY = (map.clientHeight - 28) / Math.max(1, data.canvas.height);
        view.scale = clamp(Math.min(scaleX, scaleY), 0.35, 1.2);
        view.x = Math.max(10, (map.clientWidth - data.canvas.width * view.scale) / 2);
        view.y = Math.max(10, (map.clientHeight - data.canvas.height * view.scale) / 2);
        applyView();
      }

      function resetGraph() {
        view.scale = 1;
        view.x = 8;
        view.y = 18;
        applyView();
      }

      root.querySelectorAll(".cg-zoom button").forEach((button) => {
        button.addEventListener("click", () => {
          const action = button.dataset.zoom;
          const rect = map.getBoundingClientRect();
          const cx = rect.left + rect.width / 2;
          const cy = rect.top + rect.height / 2;
          if (action === "in") zoomAt(cx, cy, 1.18);
          if (action === "out") zoomAt(cx, cy, 1 / 1.18);
          if (action === "fit") fitGraph();
          if (action === "reset") resetGraph();
        });
      });

      map.addEventListener("wheel", (event) => {
        event.preventDefault();
        zoomAt(event.clientX, event.clientY, event.deltaY < 0 ? 1.1 : 1 / 1.1);
      }, { passive: false });

      map.addEventListener("pointerdown", (event) => {
        if (event.target.closest(".cg-node, .cg-zoom")) return;
        dragging = { pointerId: event.pointerId, x: event.clientX, y: event.clientY, vx: view.x, vy: view.y };
        map.classList.add("panning");
        map.setPointerCapture(event.pointerId);
      });

      map.addEventListener("pointermove", (event) => {
        if (!dragging || dragging.pointerId !== event.pointerId) return;
        view.x = dragging.vx + event.clientX - dragging.x;
        view.y = dragging.vy + event.clientY - dragging.y;
        applyView();
      });

      function endPan(event) {
        if (!dragging || dragging.pointerId !== event.pointerId) return;
        dragging = null;
        map.classList.remove("panning");
      }

      map.addEventListener("pointerup", endPan);
      map.addEventListener("pointercancel", endPan);

      renderEdges();
      renderNodes();
      applyView();
      selectNode(data.nodes[Math.max(0, data.nodes.length - 1)]?.id);
    })();
  </script>
</div>
"""

    return (
        template.replace("__ROOT_ID__", root_id)
        .replace("__DATA__", payload)
        .replace("__WIDTH__", str(graph["canvas"]["width"]))
        .replace("__HEIGHT__", str(graph["canvas"]["height"]))
        .replace("__NODE_WIDTH__", str(NODE_WIDTH))
        .replace("__NODE_HEIGHT__", str(NODE_HEIGHT))
    )


def _assistant_output(content: str, tool_calls: list[dict[str, Any]]) -> str:
    parts = []
    if content:
        parts.append(content)
    if tool_calls:
        rendered_calls = [
            {
                "id": call.get("id"),
                "name": call.get("name"),
                "args": call.get("args"),
            }
            for call in tool_calls
        ]
        parts.append("Tool calls:\n" + _json_text(rendered_calls))
    return "\n\n".join(parts) if parts else "(empty assistant message)"


def _source_outputs(
    nodes_by_id: Mapping[str, dict[str, Any]], source_ids: Sequence[str]
) -> str:
    parts: list[str] = []
    for source_id in source_ids:
        node = nodes_by_id.get(source_id)
        if not node:
            continue
        title = node.get("title") or node.get("label") or source_id
        output = str(node.get("output") or "")
        parts.append(f"{title} output:\n{output}")
    return "\n\n".join(parts)


def _extract_tool_calls(message: Any) -> list[dict[str, Any]]:
    raw_calls = _message_attr(message, "tool_calls")
    if not raw_calls:
        additional_kwargs = _message_attr(message, "additional_kwargs")
        if isinstance(additional_kwargs, Mapping):
            raw_calls = additional_kwargs.get("tool_calls")
    if not raw_calls:
        return []
    if not isinstance(raw_calls, Sequence) or isinstance(raw_calls, (str, bytes)):
        return []
    return [_normalize_tool_call(call) for call in raw_calls]


def _normalize_tool_call(call: Any) -> dict[str, Any]:
    if isinstance(call, Mapping):
        call_id = call.get("id")
        name = call.get("name")
        args = call.get("args")
        function = call.get("function")
        if isinstance(function, Mapping):
            name = name or function.get("name")
            args = args if args is not None else function.get("arguments")
        return {
            "id": str(call_id) if call_id is not None else None,
            "name": name,
            "args": _parse_json_args(args),
        }

    return {
        "id": str(getattr(call, "id", "")) or None,
        "name": getattr(call, "name", None),
        "args": _parse_json_args(getattr(call, "args", None)),
    }


def _parse_json_args(args: Any) -> Any:
    if not isinstance(args, str):
        return _jsonable(args)
    stripped = args.strip()
    if not stripped:
        return ""
    try:
        return _jsonable(json.loads(stripped))
    except json.JSONDecodeError:
        return args


def _message_type(message: Any) -> str:
    value = _message_attr(message, "type") or _message_attr(message, "role")
    if value:
        normalized = str(value).lower()
        return {
            "assistant": "ai",
            "user": "human",
            "function": "tool",
        }.get(normalized, normalized)
    return type(message).__name__.replace("Message", "").lower()


def _message_name(message: Any) -> str | None:
    value = _message_attr(message, "name")
    return str(value) if value else None


def _message_content(message: Any) -> Any:
    return _message_attr(message, "content")


def _message_attr(message: Any, attr: str) -> Any:
    if isinstance(message, Mapping):
        return message.get(attr)
    return getattr(message, attr, None)


def _format_detail(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return _compact_text(value)
    return _compact_text(_json_text(value))


def _json_text(value: Any) -> str:
    try:
        return json.dumps(_jsonable(value), indent=2, ensure_ascii=False)
    except TypeError:
        return str(value)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if hasattr(value, "model_dump"):
        try:
            return _jsonable(value.model_dump())
        except Exception:
            pass
    if isinstance(value, Mapping):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(item) for item in value]
    return str(value)


def _node_y(kind: str) -> int:
    if kind == "tool":
        return 208
    if kind == "message":
        return 132
    return 70


def _pluralize_tool_calls(count: int) -> str:
    return f"{count} tool call" if count == 1 else f"{count} tool calls"


def _compact_label(value: Any, *, limit: int = 28) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "..."


def _compact_text(value: str, *, limit: int = TEXT_LIMIT) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return f"{text[:limit].rstrip()}\n\n... truncated {len(text) - limit} characters"


def _same_text(left: str, right: str) -> bool:
    return " ".join(left.split()) == " ".join(right.split())
