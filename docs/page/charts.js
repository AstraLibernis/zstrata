// Chart helpers shared with an earlier project's page: line, bar and heatmap charts in SVG,
// coloured from the theme tokens, with tooltips that use textContent only.
const NS = "http://www.w3.org/2000/svg";
function el(tag, attrs, parent) { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; }
function txt(parent, x, y, s, attrs = {}) { const t = el("text", { x, y, ...attrs }, parent); t.textContent = s; return t; }

/* Tooltip shared by all charts. Labels go in with textContent. */
function tipFor(box) {
  const t = document.createElement("div"); t.className = "tip"; t.hidden = true; box.appendChild(t);
  return {
    show(x, y, head, lines) {
      t.replaceChildren();
      const h = document.createElement("div"); h.className = "h"; h.textContent = head; t.appendChild(h);
      for (const [color, val, name] of lines) {
        const r = document.createElement("div"); r.className = "r";
        const i = document.createElement("i"); i.style.background = color; r.appendChild(i);
        const b = document.createElement("b"); b.textContent = val; r.appendChild(b);
        const s = document.createElement("span"); s.textContent = name; r.appendChild(s);
        t.appendChild(r);
      }
      t.hidden = false;
      const bw = box.clientWidth, tw = t.offsetWidth;
      t.style.left = Math.min(Math.max(4, x + 14), bw - tw - 4) + "px";
      t.style.top = Math.max(4, y - 10) + "px";
    },
    hide() { t.hidden = true; }
  };
}

function frame(box, title, sub, legend) {
  const c = document.createElement("div"); c.className = "chart";
  const h = document.createElement("div"); h.className = "ttl"; h.textContent = title; c.appendChild(h);
  if (sub) { const s = document.createElement("div"); s.className = "sub"; s.textContent = sub; c.appendChild(s); }
  if (legend && legend.length > 1) {
    const lg = document.createElement("div"); lg.className = "legend";
    for (const [color, name, kind] of legend) { const sp = document.createElement("span"); const i = document.createElement("i"); if (kind) i.className = kind; i.style.background = color; if (kind === "dash") i.style.color = color; sp.appendChild(i); sp.appendChild(document.createTextNode(name)); lg.appendChild(sp); }
    c.appendChild(lg);
  }
  box.appendChild(c);
  return c;
}

function niceTicks(lo, hi, n = 5) {
  const span = hi - lo, step0 = span / n, mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => span / s <= n) || mag * 10;
  const out = []; for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10)); return out;
}

/* Line chart. series: [{name, color, pts:[[x,y,extra]], dash}] ; opts: xlog, ylog, xTicks, xFmt, yLabel, xLabel, marks */
function lineChart(box, title, sub, series, o = {}) {
  const c = frame(box, title, sub, series.map(s => [s.color, s.name, s.dash ? "dash" : ""]));
  const W = Math.max(300, c.clientWidth - 24), H = o.h || 300, m = { l: 52, r: o.mr || 18, t: 14, b: 40 };
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": title }, c);
  const xs = series.flatMap(s => s.pts.map(p => p[0])), ys = series.flatMap(s => s.pts.map(p => p[1]));
  const fx = o.xlog ? Math.log2 : (v => v), fy = o.ylog ? Math.log10 : (v => v);
  const x0 = fx(o.xmin ?? Math.min(...xs)), x1 = fx(o.xmax ?? Math.max(...xs));
  const y0 = o.ylog ? fy(o.ymin ?? Math.min(...ys)) : (o.ymin ?? 0), y1 = o.ylog ? fy(o.ymax ?? Math.max(...ys)) : (o.ymax ?? Math.max(...ys) * 1.08);
  const X = v => m.l + (fx(v) - x0) / (x1 - x0) * (W - m.l - m.r);
  const Y = v => H - m.b - (fy(v) - y0) / (y1 - y0) * (H - m.t - m.b);
  const yt = o.ylog ? (o.yTicks || []) : niceTicks(y0, y1);
  for (const v of yt) { el("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), stroke: css("--grid"), "stroke-width": 1 }, svg); txt(svg, m.l - 6, Y(v) + 3.5, o.yFmt ? o.yFmt(v) : String(v), { "text-anchor": "end" }); }
  el("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, stroke: css("--axis") }, svg);
  for (const v of (o.xTicks || niceTicks(Math.min(...xs), Math.max(...xs)))) txt(svg, X(v), H - m.b + 15, o.xFmt ? o.xFmt(v) : String(v), { "text-anchor": "middle" });
  if (o.xLabel) txt(svg, (m.l + W - m.r) / 2, H - 6, o.xLabel, { "text-anchor": "middle", class: "lbl" });
  if (o.yLabel) txt(svg, 12, m.t + 2, o.yLabel, { class: "lbl" });
  for (const mk of (o.marks || [])) {
    if (mk.x != null) { el("line", { x1: X(mk.x), x2: X(mk.x), y1: m.t, y2: H - m.b, stroke: css(mk.color || "--hot"), "stroke-width": 1, "stroke-dasharray": "3 3" }, svg); txt(svg, X(mk.x) + (mk.left ? -5 : 5), m.t + 10 + (mk.dy || 0), mk.label, { class: "note", "text-anchor": mk.left ? "end" : "start" }); }
    if (mk.y != null) { el("line", { x1: m.l, x2: W - m.r, y1: Y(mk.y), y2: Y(mk.y), stroke: css(mk.color || "--muted"), "stroke-width": 1, "stroke-dasharray": "3 3" }, svg); txt(svg, m.l + 6, Y(mk.y) - 5, mk.label, { class: "note" }); }
  }
  for (const s of series) {
    const d = s.pts.map((p, i) => (i ? "L" : "M") + X(p[0]).toFixed(1) + "," + Y(p[1]).toFixed(1)).join("");
    el("path", { d, fill: "none", stroke: s.color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round", ...(s.dash ? { "stroke-dasharray": "5 4" } : {}) }, svg);
    for (const p of s.pts) el("circle", { cx: X(p[0]), cy: Y(p[1]), r: 2.6, fill: s.color }, svg);
    if (o.endLabels && s.pts.length) { const p = s.pts[s.pts.length - 1]; txt(svg, X(p[0]) + 6, Y(p[1]) + 3.5, s.short || s.name, { class: "lbl" }); }
  }
  const hair = el("line", { y1: m.t, y2: H - m.b, stroke: css("--axis"), "stroke-width": 1, visibility: "hidden" }, svg);
  const tip = tipFor(c);
  const allX = [...new Set(xs)].sort((a, b) => a - b);
  const hit = el("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent", tabindex: 0 }, svg);
  function at(xv, cx, cy) {
    hair.setAttribute("x1", X(xv)); hair.setAttribute("x2", X(xv)); hair.setAttribute("visibility", "visible");
    const lines = series.map(s => { const p = s.pts.find(q => q[0] === xv); return p ? [s.color, (o.vFmt ? o.vFmt(p[1]) : fmt(p[1])) + (o.unit ? " " + o.unit : ""), s.name + (p[2] ? " · " + p[2] : "")] : null; }).filter(Boolean);
    tip.show(cx, cy, (o.xName || "x") + " " + (o.xFmt ? o.xFmt(xv) : xv), lines);
  }
  hit.addEventListener("pointermove", e => {
    const r = svg.getBoundingClientRect(), px = (e.clientX - r.left) / r.width * W;
    let best = allX[0]; for (const v of allX) if (Math.abs(X(v) - px) < Math.abs(X(best) - px)) best = v;
    const cr = c.getBoundingClientRect(); at(best, e.clientX - cr.left, e.clientY - cr.top);
  });
  let ki = 0;
  hit.addEventListener("focus", () => at(allX[ki], 60, 40));
  hit.addEventListener("keydown", e => { if (e.key === "ArrowRight") ki = Math.min(allX.length - 1, ki + 1); else if (e.key === "ArrowLeft") ki = Math.max(0, ki - 1); else return; e.preventDefault(); at(allX[ki], 60, 40); });
  const off = () => { hair.setAttribute("visibility", "hidden"); tip.hide(); };
  hit.addEventListener("pointerleave", off); hit.addEventListener("blur", off);
}

/* Horizontal bar chart: items [{name, v, color, note}] */
function barChart(box, title, sub, items, o = {}) {
  const c = frame(box, title, sub);
  const W = Math.max(300, c.clientWidth - 24), rowH = 26, m = { l: o.ml || 150, r: 74, t: 6, b: 24 }, H = m.t + m.b + items.length * rowH;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": title }, c);
  const max = o.max || Math.max(...items.map(i => i.v)) * 1.05;
  const lo = o.log ? (o.min || Math.min(...items.map(i => i.v)) / 2) : 0;
  const X = o.log ? (v => m.l + Math.log10(Math.max(v, lo) / lo) / Math.log10(max / lo) * (W - m.l - m.r)) : (v => m.l + v / max * (W - m.l - m.r));
  for (const v of (o.ticks || niceTicks(0, max, 4))) { el("line", { x1: X(v), x2: X(v), y1: m.t, y2: H - m.b, stroke: css("--grid") }, svg); txt(svg, X(v), H - 8, o.tFmt ? o.tFmt(v) : String(v), { "text-anchor": "middle" }); }
  const tip = tipFor(c);
  items.forEach((it, i) => {
    const y = m.t + i * rowH + 7, h = Math.min(16, rowH - 10), w = Math.max(1, X(it.v) - m.l);
    txt(svg, m.l - 8, y + h / 2 + 4, it.name, { "text-anchor": "end", class: "lbl" });
    const r = Math.min(4, w / 2);
    el("path", { d: `M${m.l},${y}h${w - r}a${r},${r} 0 0 1 ${r},${r}v${h - 2 * r}a${r},${r} 0 0 1 -${r},${r}h-${w - r}z`, fill: it.color || css("--s1") }, svg);
    txt(svg, m.l + w + 6, y + h / 2 + 4, (o.vFmt ? o.vFmt(it.v) : fmt(it.v)) + (o.unit ? " " + o.unit : ""), { class: "lbl" });
    const hit = el("rect", { x: 0, y: y - 5, width: W, height: rowH, fill: "transparent", tabindex: 0 }, svg);
    const show = (cx, cy) => tip.show(cx, cy, it.name, [[it.color || css("--s1"), (o.vFmt ? o.vFmt(it.v) : fmt(it.v)) + (o.unit ? " " + o.unit : ""), it.note || ""]]);
    hit.addEventListener("pointermove", e => { const cr = c.getBoundingClientRect(); show(e.clientX - cr.left, e.clientY - cr.top); });
    hit.addEventListener("focus", () => show(80, 40 + i * 20));
    hit.addEventListener("pointerleave", () => tip.hide()); hit.addEventListener("blur", () => tip.hide());
  });
}

/* Heatmap: xs (columns), ys (rows), get(x,y) -> {v, extra} */
function heatmap(box, title, sub, xs, ys, get, o = {}) {
  const c = frame(box, title, sub);
  const W = Math.max(300, c.clientWidth - 24), m = { l: o.ml || 64, r: 10, t: 8, b: 40 }, cw = (W - m.l - m.r) / xs.length, ch = 30, H = m.t + m.b + ys.length * ch;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": title }, c);
  const steps = ["--seq0", "--seq1", "--seq2", "--seq3", "--seq4", "--seq5"].map(css);
  const vals = []; xs.forEach(x => ys.forEach(y => { const g = get(x, y); if (g) vals.push(g.v); }));
  const max = o.max || Math.max(...vals);
  const tip = tipFor(c);
  ys.forEach((y, j) => {
    txt(svg, m.l - 8, m.t + j * ch + ch / 2 + 4, o.yFmt ? o.yFmt(y) : y, { "text-anchor": "end", class: "lbl" });
    xs.forEach((x, i) => {
      const g = get(x, y); if (!g) return;
      const k = Math.min(steps.length - 1, Math.floor(g.v / max * steps.length));
      const rx = m.l + i * cw + 1, ry = m.t + j * ch + 1;
      const cell = el("rect", { x: rx, y: ry, width: cw - 2, height: ch - 2, rx: 2, fill: steps[k], tabindex: 0 }, svg);
      const t = txt(svg, rx + (cw - 2) / 2, ry + ch / 2 + 2, fmt(g.v, 0), { "text-anchor": "middle" });
      t.style.pointerEvents = "none";
      t.style.fill = k >= 3 ? css("--on-strong") : css("--ink-2");
      const show = (cx, cy) => tip.show(cx, cy, (o.xName || "") + " " + (o.xFmt ? o.xFmt(x) : x) + " · " + (o.yName || "") + " " + (o.yFmt ? o.yFmt(y) : y), [[steps[k], fmt(g.v) + (o.unit ? " " + o.unit : ""), g.extra || ""]]);
      cell.addEventListener("pointermove", e => { const cr = c.getBoundingClientRect(); show(e.clientX - cr.left, e.clientY - cr.top); });
      cell.addEventListener("focus", () => show(rx, ry + 30));
      cell.addEventListener("pointerleave", () => tip.hide()); cell.addEventListener("blur", () => tip.hide());
    });
  });
  xs.forEach((x, i) => txt(svg, m.l + i * cw + cw / 2, H - m.b + 15, o.xFmt ? o.xFmt(x) : x, { "text-anchor": "middle" }));
  if (o.xLabel) txt(svg, (m.l + W - m.r) / 2, H - 6, o.xLabel, { "text-anchor": "middle", class: "lbl" });
}

