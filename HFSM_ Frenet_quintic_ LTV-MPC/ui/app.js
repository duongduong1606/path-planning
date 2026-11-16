/* Static scenario viewer for the L2 lateral path-planning pipeline.
 *
 * Reads two generated globals:
 *   window.SCENARIO_DATA    - one exported ui_data.json payload
 *   window.SCENARIO_LIBRARY - metadata for every scenario in the library
 *
 * No build step, no external dependencies.  All drawing is Canvas 2D.
 */
(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  const isNum = (v) => typeof v === "number" && Number.isFinite(v);

  const COLORS = {
    road: "#1b2634",
    roadEdge: "#33475c",
    lane: "#e8eef5",
    laneYellow: "#f2c14e",
    laneDim: "rgba(215, 226, 238, 0.55)",
    center: "rgba(255, 255, 255, 0.16)",
    barrier: "#f0a02a",
    barrierHalo: "rgba(240, 160, 42, 0.30)",
    ego: "#7ee0a8",
    egoEdge: "#0d2a1e",
    traj: "#ff3b30",
    trajOld: "rgba(255, 59, 48, 0.16)",
    truck: "#f2a04b",
    car: "#6aa8ff",
    other: "#9aa8b8",
    accent: "#39c6f2",
    warn: "#f2b84b",
    good: "#46d68a",
    grid: "rgba(255, 255, 255, 0.07)",
    text: "#8fa2b8",
  };

  const S = {
    data: null,
    ts: null,
    ego: null,
    duration: 1,
    t: 0,
    playing: false,
    speed: 1,
    follow: true,
    zoom: 1,
    loadedId: null,
    loadToken: 0,
    raf: 0,
    lastFrame: 0,
  };

  /* ------------------------------------------------------------ utilities */

  function times() {
    return S.ts && S.ts.time_s ? S.ts.time_s : [0];
  }

  function indexAt(t) {
    const arr = times();
    let lo = 0, hi = arr.length - 1;
    if (t <= arr[0]) return 0;
    if (t >= arr[hi]) return hi;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (arr[mid] < t) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  }

  function sample(key, t) {
    const arr = S.ts && S.ts[key];
    if (!arr || !arr.length) return NaN;
    return arr[indexAt(t)];
  }

  function fmt(value, digits) {
    if (!isNum(value)) return "—";
    if (Math.abs(value) >= 1000) return value.toFixed(0);
    return value.toFixed(digits == null ? 2 : digits);
  }

  function signed(value, digits) {
    if (!isNum(value)) return "—";
    const s = value.toFixed(digits == null ? 2 : digits);
    return value > 0 ? "+" + s : s;
  }

  function setupCanvas(canvas) {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const w = Math.max(1, Math.round((rect.width || canvas.clientWidth || 300) * dpr));
    const h = Math.max(1, Math.round((rect.height || canvas.clientHeight || 150) * dpr));
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w: w / dpr, h: h / dpr };
  }

  /* -------------------------------------------------------------- booting */

  function buildSelector() {
    const select = $("scenario-select");
    const library = Array.isArray(window.SCENARIO_LIBRARY) ? window.SCENARIO_LIBRARY : [];
    select.innerHTML = "";
    if (!library.length) {
      const opt = document.createElement("option");
      opt.textContent = S.data && S.data.scenario ? S.data.scenario.id : "no library";
      select.appendChild(opt);
      return;
    }
    const byFamily = new Map();
    for (const item of library) {
      const fam = item.family || "?";
      if (!byFamily.has(fam)) byFamily.set(fam, []);
      byFamily.get(fam).push(item);
    }
    const families = Array.from(byFamily.keys()).sort((a, b) => {
      const na = parseInt(String(a).replace(/\D/g, ""), 10);
      const nb = parseInt(String(b).replace(/\D/g, ""), 10);
      if (isNaN(na) || isNaN(nb)) return String(a).localeCompare(String(b));
      if (na !== nb) return na - nb;
      return String(a).localeCompare(String(b));
    });
    for (const fam of families) {
      const group = document.createElement("optgroup");
      group.label = fam;
      for (const item of byFamily.get(fam)) {
        const opt = document.createElement("option");
        opt.value = item.id;
        opt.textContent = item.id + " · " + item.title;
        group.appendChild(opt);
      }
      select.appendChild(group);
    }
    select.value = S.loadedId || "";
  }

  function setStatus(text, bad) {
    const el = $("load-status");
    el.textContent = text || "";
    el.classList.toggle("bad", !!bad);
  }

  function tryLoadScenario(id) {
    if (!id || id === S.loadedId) return;
    setStatus("loading " + id + "…", false);
    // Optional per-scenario bundle: ui/data/<id>.js assigns window.SCENARIO_DATA.
    // A token keeps a slow, superseded load from overwriting a newer selection.
    const token = (S.loadToken = (S.loadToken || 0) + 1);
    const script = document.createElement("script");
    script.src = "data/" + encodeURIComponent(id) + ".js";
    script.onload = () => {
      script.remove();
      if (token !== S.loadToken) return;
      if (window.SCENARIO_DATA && window.SCENARIO_DATA.scenario && window.SCENARIO_DATA.scenario.id === id) {
        loadData(window.SCENARIO_DATA);
        setStatus("", false);
      } else {
        setStatus("bundle for " + id + " is empty", true);
      }
    };
    script.onerror = () => {
      script.remove();
      if (token !== S.loadToken) return;
      setStatus("not exported — run: python -m l2_path_planner ui --scenario " + id, true);
      $("scenario-select").value = S.loadedId;
    };
    document.head.appendChild(script);
  }

  function loadData(data) {
    if (!data || !data.scenario || !data.timeseries || !data.ego || !data.ego.length) {
      showEmpty();
      return;
    }
    S.data = data;
    S.ts = data.timeseries;
    S.ego = data.ego;
    S.loadedId = data.scenario.id;
    const last = times()[times().length - 1];
    S.duration = Math.max(0.1, data.scenario.duration_s || last || 1);
    S.t = 0;
    S.playing = false;
    S.zoom = 1;
    S.follow = true;

    $("empty").hidden = true;
    $("app").hidden = false;
    $("verdict").hidden = false;

    renderHeader();
    buildKpis();
    renderChecks();
    renderMetrics();
    renderLegend();
    buildTimeline();
    buildSelector();
    setStatus("", false);
    resizeAll();
    renderAll();
  }

  function showEmpty() {
    $("app").hidden = true;
    $("empty").hidden = false;
    $("verdict").hidden = true;
  }

  function renderHeader() {
    const sc = S.data.scenario;
    $("scenario-title").textContent = sc.title || sc.id;
    const bits = [
      sc.id,
      sc.category ? sc.category + " family" : "",
      sc.requirements && sc.requirements.length ? sc.requirements.join(" · ") : "",
      "seed " + (sc.seed != null ? sc.seed : "—"),
      fmt(S.duration, 1) + " s",
    ].filter(Boolean);
    $("scenario-meta").textContent = bits.join("   ·   ");
    const verdict = $("verdict");
    const passed = !!(S.data.result && S.data.result.passed);
    verdict.className = "verdict " + (passed ? "pass" : "fail");
    verdict.textContent = passed ? "PASS" : "FAIL";
    verdict.hidden = false;
  }

  /* ------------------------------------------------------------------ KPIs */

  const KPI_DEFS = [
    { id: "kpi-mode", key: "Active function", wide: true },
    { id: "kpi-speed", key: "Speed" },
    { id: "kpi-target", key: "Target speed" },
    { id: "kpi-d", key: "Lateral error" },
    { id: "kpi-heading", key: "Heading error" },
    { id: "kpi-tlc", key: "Min TLC" },
    { id: "kpi-ay", key: "Lateral accel" },
    { id: "kpi-wind", key: "Crosswind" },
  ];

  function buildKpis() {
    const grid = $("kpis");
    grid.innerHTML = "";
    for (const def of KPI_DEFS) {
      const card = document.createElement("div");
      card.className = "kpi" + (def.wide ? " wide" : "");
      const k = document.createElement("div");
      k.className = "k";
      k.textContent = def.key;
      const v = document.createElement("div");
      v.className = "v";
      v.id = def.id;
      card.appendChild(k);
      card.appendChild(v);
      grid.appendChild(card);
    }
  }

  function updateKpis(i) {
    const mode = modeName(S.ts.active_function ? S.ts.active_function[i] : 0);
    const lateral = S.ego[i] ? S.ego[i].lateral_state : "";
    $("kpi-mode").innerHTML =
      '<span class="mode-badge mode-' + mode + '">' + mode + "</span>" +
      (lateral ? ' <small style="color:var(--muted);font-size:11px">' + lateral + "</small>" : "");

    const speed = S.ts.speed_mps ? S.ts.speed_mps[i] : NaN;
    $("kpi-speed").innerHTML = fmt(speed, 1) + " <small>m/s</small>";

    const target = S.ts.target_speed_mps ? S.ts.target_speed_mps[i] : NaN;
    $("kpi-target").innerHTML = fmt(target, 1) + " <small>m/s</small>";

    const d = S.ts.road_d_m ? S.ts.road_d_m[i] : NaN;
    $("kpi-d").innerHTML = signed(d, 2) + " <small>m</small>";

    const heading = S.ts.heading_error_rad ? S.ts.heading_error_rad[i] : NaN;
    $("kpi-heading").innerHTML = isNum(heading) ? signed(heading * 180 / Math.PI, 2) + " <small>deg</small>" : "—";

    const tl = S.ts.tlc_left_s ? S.ts.tlc_left_s[i] : 99;
    const tr = S.ts.tlc_right_s ? S.ts.tlc_right_s[i] : 99;
    const tlc = Math.min(tl, tr);
    $("kpi-tlc").innerHTML = tlc >= 90 ? "—" : fmt(tlc, 2) + " <small>s</small>";

    const ay = S.ts.lateral_acceleration_mps2 ? S.ts.lateral_acceleration_mps2[i] : NaN;
    $("kpi-ay").innerHTML = signed(ay, 2) + " <small>m/s²</small>";

    const wind = S.ts.crosswind_mps ? S.ts.crosswind_mps[i] : NaN;
    $("kpi-wind").innerHTML = signed(wind, 1) + " <small>m/s</small>";
  }

  function modeName(code) {
    return ({ 0: "NONE", 1: "HWA", 2: "LKA", 3: "ELK" })[code] || "NONE";
  }

  /* --------------------------------------------------------------- panels */

  function renderChecks() {
    const host = $("checks");
    host.innerHTML = "";
    const checks = (S.data.result && S.data.result.checks) || {};
    const names = Object.keys(checks);
    if (!names.length) {
      host.innerHTML = '<div class="check pass"><span class="icon">✓</span><div><div class="name">No checks</div></div></div>';
      return;
    }
    for (const name of names) {
      const c = checks[name];
      const el = document.createElement("div");
      el.className = "check " + (c.passed ? "pass" : "fail");
      const icon = document.createElement("span");
      icon.className = "icon";
      icon.textContent = c.passed ? "✓" : "✕";
      const body = document.createElement("div");
      const title = document.createElement("div");
      title.className = "name";
      title.textContent = name.replace(/_/g, " ");
      const detail = document.createElement("div");
      detail.className = "detail";
      detail.innerHTML = checkDetail(c);
      body.appendChild(title);
      body.appendChild(detail);
      el.appendChild(icon);
      el.appendChild(body);
      host.appendChild(el);
    }
  }

  function checkDetail(c) {
    if (c.expected !== undefined) {
      const obs = Array.isArray(c.observed) ? (c.observed.length ? c.observed.join(", ") : "none") : String(c.observed);
      return "expected " + [].concat(c.expected).join(", ") + " · observed " + obs;
    }
    if (c.forbidden !== undefined) {
      const obs = Array.isArray(c.observed) ? (c.observed.length ? c.observed.join(", ") : "none") : String(c.observed);
      return "forbidden " + [].concat(c.forbidden).join(", ") + " · observed " + obs;
    }
    if (c.window !== undefined) {
      return "window " + c.window[0] + "–" + c.window[1] + " s · first " + (c.observed == null ? "—" : fmt(c.observed, 2) + " s");
    }
    if (c.limit !== undefined) {
      return "observed " + fmt(c.observed, 3) + " · limit " + fmt(c.limit, 3);
    }
    if (c.observed !== undefined) return "observed " + fmt(c.observed, 3);
    return "";
  }

  function renderMetrics() {
    const host = $("metrics");
    host.innerHTML = "";
    const metrics = (S.data.result && S.data.result.metrics) || {};
    const order = [
      "max_abs_lateral_error_m",
      "min_lane_margin_m",
      "max_abs_lateral_acceleration_mps2",
      "max_abs_lateral_jerk_mps3",
      "active_functions",
    ];
    const keys = order.filter((k) => k in metrics).concat(Object.keys(metrics).filter((k) => !order.includes(k)));
    for (const key of keys) {
      const value = metrics[key];
      const k = document.createElement("div");
      k.className = "mk";
      k.textContent = key.replace(/_/g, " ");
      const v = document.createElement("div");
      v.className = "mv";
      v.textContent = Array.isArray(value) ? (value.length ? value.join(", ") : "—") : fmt(value, 3);
      host.appendChild(k);
      host.appendChild(v);
    }
  }

  function renderLegend() {
    const items = [
      { label: "ego", color: COLORS.ego, box: true },
      { label: "plan", color: COLORS.traj, line: true },
      { label: "truck", color: COLORS.truck, box: true },
      { label: "car", color: COLORS.car, box: true },
      { label: "barrier / edge", color: COLORS.barrier, line: true },
      { label: "lane line", color: COLORS.lane, line: true, dash: true },
      { label: "centre line", color: COLORS.laneYellow, line: true },
    ];
    $("legend").innerHTML = items
      .map((it) => {
        const cls = "swatch" + (it.box ? " box" : "");
        const style = it.box
          ? "background:" + it.color
          : "background:" + it.color + ";" + (it.dash ? "opacity:.6" : "");
        return '<span class="item"><span class="' + cls + '" style="' + style + '"></span>' + it.label + "</span>";
      })
      .join("");
  }

  function buildTimeline() {
    const host = $("timeline");
    host.innerHTML = "";
    const cursor = document.createElement("div");
    cursor.className = "cursor";
    cursor.id = "timeline-cursor";
    host.appendChild(cursor);
    const events = S.data.events || [];
    for (const ev of events) {
      const marker = document.createElement("div");
      const kind = ev.event === "MODE" || ev.event === "DISTURBANCE" || ev.event === "SFM_LONGITUDINAL" ? ev.event : "other";
      marker.className = "marker " + kind;
      marker.style.left = (clamp(ev.t / S.duration, 0, 1) * 100) + "%";
      marker.title = fmt(ev.t, 2) + " s · " + ev.event + (ev.detail ? " · " + ev.detail : "");
      host.appendChild(marker);
    }
  }

  /* ---------------------------------------------------------------- scene */

  // Vertical half-window in metres at zoom 1.  22 m shows ~10 lanes-worth of
  // road ahead while keeping a 3.5 m lane readable at ~35 px.
  const BASE_HALF_H_M = 22;
  const ZOOM_MIN = 0.5;
  const ZOOM_MAX = 4;

  function viewTransform(w, h) {
    const road = S.data.road;
    if (S.follow) {
      const i = indexAt(S.t);
      const cx = S.ts.x_m ? S.ts.x_m[i] : 0;
      const cy = S.ts.y_m ? S.ts.y_m[i] : 0;
      const halfH = BASE_HALF_H_M / S.zoom;
      return { cx, cy, scale: h / (2 * halfH) };
    }
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    const eat = (pt) => {
      if (!pt) return;
      minX = Math.min(minX, pt[0]); maxX = Math.max(maxX, pt[0]);
      minY = Math.min(minY, pt[1]); maxY = Math.max(maxY, pt[1]);
    };
    if (road) {
      const carriage = road.lanes && road.lanes.carriageway;
      if (carriage) {
        (carriage.left || []).forEach(eat);
        (carriage.right || []).forEach(eat);
      } else {
        (road.left || []).forEach(eat);
        (road.right || []).forEach(eat);
      }
    }
    if (!isFinite(minX)) { minX = -10; maxX = 10; minY = -10; maxY = 10; }
    const bw = Math.max(maxX - minX, 20);
    const bh = Math.max(maxY - minY, 20);
    const scale = Math.min(w / bw, h / bh) * 0.9;
    return { cx: (minX + maxX) / 2, cy: (minY + maxY) / 2, scale };
  }

  // World frame is x-forward / y-right (the same convention as the scenario
  // data and `frenet_to_global`), so screen y grows with world y: an eastbound
  // ego keeps its right-hand side below it. Do not flip this sign without also
  // flipping the `ctx.rotate(yaw)` calls in drawActors/drawEgo.
  function w2s(x, y, view, w, h) {
    return [w / 2 + (x - view.cx) * view.scale, h / 2 + (y - view.cy) * view.scale];
  }

  function screenPolyline(points, i0, i1, view, w, h) {
    const out = [];
    if (!points) return out;
    const start = clamp(i0, 0, points.length - 1);
    const end = clamp(i1, 0, points.length - 1);
    for (let i = start; i <= end; i++) {
      const pt = points[i];
      out.push(w2s(pt[0], pt[1], view, w, h));
    }
    return out;
  }

  function strokePolyline(ctx, pts, offsetPx) {
    if (!pts || pts.length < 2) return;
    const off = offsetPx || 0;
    ctx.beginPath();
    for (let i = 0; i < pts.length; i++) {
      let x = pts[i][0], y = pts[i][1];
      if (off) {
        const a = pts[Math.max(0, i - 1)];
        const b = pts[Math.min(pts.length - 1, i + 1)];
        const dx = b[0] - a[0], dy = b[1] - a[1];
        const len = Math.hypot(dx, dy) || 1;
        x += (-dy / len) * off;
        y += (dx / len) * off;
      }
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  function sIndex(s, upper) {
    const arr = (S.data.road && S.data.road.s) || [];
    let lo = 0, hi = arr.length - 1;
    if (upper) {
      while (lo < hi) {
        const mid = (lo + hi + 1) >> 1;
        if (arr[mid] <= s) lo = mid;
        else hi = mid - 1;
      }
      return lo;
    }
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (arr[mid] < s) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  }

  function drawRoad(ctx, view, w, h) {
    const road = S.data.road;
    if (!road) return;
    const lanes = road.lanes || null;
    const carriage = lanes && lanes.carriageway ? lanes.carriageway : null;
    const left = (carriage ? carriage.left : road.left) || [];
    const right = (carriage ? carriage.right : road.right) || [];

    if (left.length > 1 && right.length > 1) {
      ctx.beginPath();
      for (let i = 0; i < left.length; i++) {
        const p = w2s(left[i][0], left[i][1], view, w, h);
        i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]);
      }
      for (let i = right.length - 1; i >= 0; i--) {
        const p = w2s(right[i][0], right[i][1], view, w, h);
        ctx.lineTo(p[0], p[1]);
      }
      ctx.closePath();
      ctx.fillStyle = COLORS.road;
      ctx.fill();
      ctx.strokeStyle = COLORS.roadEdge;
      ctx.lineWidth = 1;
      ctx.setLineDash([]);
      ctx.stroke();
    }

    if (lanes && lanes.lines && lanes.lines.length) {
      // Lane boundaries, styled per Vietnamese road-marking classes.
      const styles = lanes.styles || [];
      for (const line of lanes.lines) {
        for (const style of styles) {
          if (style.index !== line.index) continue;
          const i0 = sIndex(style.s_start, false);
          const i1 = sIndex(style.s_end, true);
          if (i1 <= i0) continue;
          drawLaneLine(
            ctx,
            screenPolyline(line.points, i0, i1, view, w, h),
            style.kind,
            view.scale,
          );
        }
      }
    } else {
      // Lane centre reference (faint dashes) for the legacy single-lane road.
      if (road.x && road.y) {
        const pts = [];
        for (let i = 0; i < road.x.length; i += 2) pts.push(w2s(road.x[i], road.y[i], view, w, h));
        ctx.setLineDash([2, 8]);
        ctx.lineWidth = 1;
        ctx.strokeStyle = COLORS.center;
        strokePolyline(ctx, pts, 0);
        ctx.setLineDash([]);
      }

      // Lane boundaries styled per marking range.
      const markings = road.markings || [];
      if (!markings.length) {
        drawLaneLine(ctx, screenPolyline(left, 0, left.length - 1, view, w, h), "WHITE_DASHED", view.scale);
        drawLaneLine(ctx, screenPolyline(right, 0, right.length - 1, view, w, h), "WHITE_DASHED", view.scale);
      }
      for (const entry of markings) {
        const i0 = sIndex(entry.s_start_m, false);
        const i1 = sIndex(entry.s_end_m, true);
        if (i1 <= i0) continue;
        drawLaneLine(ctx, screenPolyline(left, i0, i1, view, w, h), entry.left, view.scale);
        drawLaneLine(ctx, screenPolyline(right, i0, i1, view, w, h), entry.right, view.scale);
      }
    }

    // Barriers / road edges.
    for (const edge of road.edges || []) {
      const pts = (edge.points || []).map((pt) => w2s(pt[0], pt[1], view, w, h));
      ctx.lineWidth = 3.5;
      ctx.strokeStyle = COLORS.barrier;
      ctx.setLineDash([]);
      strokePolyline(ctx, pts, 0);
      ctx.lineWidth = 1;
      ctx.strokeStyle = COLORS.barrierHalo;
      strokePolyline(ctx, pts, 3.5);
    }
  }

  // QCVN 41: lane lines are 15 cm wide; white 2.1 dashes are 3 m on / 6 m off,
  // so the dash pattern is expressed in metres and scaled to pixels.
  function drawLaneLine(ctx, pts, kind, scale) {
    const k = String(kind || "NONE").toUpperCase();
    if (k === "NONE" || k === "LOST" || !pts || pts.length < 2) return;
    const width = Math.max(1, 0.15 * scale);
    const gap = Math.max(1.2, 0.15 * scale);
    ctx.setLineDash([]);
    ctx.lineWidth = width;
    ctx.strokeStyle = COLORS.lane;
    if (k === "WHITE_DASHED" || k === "DASHED") {
      ctx.setLineDash([3 * scale, 6 * scale]);
      strokePolyline(ctx, pts, 0);
    } else if (k === "DOUBLE_YELLOW" || k === "YELLOW_DOUBLE") {
      ctx.strokeStyle = COLORS.laneYellow;
      strokePolyline(ctx, pts, -gap);
      strokePolyline(ctx, pts, gap);
    } else if (k === "SINGLE_YELLOW" || k === "YELLOW_SOLID") {
      ctx.strokeStyle = COLORS.laneYellow;
      strokePolyline(ctx, pts, 0);
    } else if (k === "WHITE_DOUBLE" || k === "DOUBLE") {
      strokePolyline(ctx, pts, -gap);
      strokePolyline(ctx, pts, gap);
    } else {
      // WHITE_SOLID / SOLID / ROAD_EDGE / BARRIER
      strokePolyline(ctx, pts, 0);
    }
    ctx.setLineDash([]);
  }

  function currentTrajectories() {
    const list = S.data.trajectories || [];
    if (!list.length) return [];
    let lo = 0, hi = list.length - 1, best = 0;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (list[mid].stamp <= S.t + 1e-6) { best = mid; lo = mid + 1; }
      else hi = mid - 1;
    }
    const out = [];
    for (let k = Math.max(0, best - 3); k <= best; k++) out.push({ item: list[k], age: best - k });
    return out;
  }

  function drawTrajectories(ctx, view, w, h) {
    const items = currentTrajectories();
    for (const entry of items) {
      const pts = (entry.item.points || []).map((pt) => w2s(pt.x, pt.y, view, w, h));
      if (pts.length < 2) continue;
      ctx.setLineDash([]);
      ctx.lineWidth = entry.age === 0 ? 1.2 : 1;
      ctx.strokeStyle = entry.age === 0 ? COLORS.traj : COLORS.trajOld;
      strokePolyline(ctx, pts, 0);
      if (entry.age === 0) {
        const head = pts[pts.length - 1];
        ctx.beginPath();
        ctx.arc(head[0], head[1], 2, 0, Math.PI * 2);
        ctx.fillStyle = COLORS.traj;
        ctx.fill();
      }
    }
  }

  function actorAt(actor, t) {
    const pts = actor.points || [];
    if (!pts.length) return null;
    let lo = 0, hi = pts.length - 1;
    if (t <= pts[0].t) return { pt: pts[0], next: pts[Math.min(1, pts.length - 1)] };
    if (t >= pts[hi].t) return { pt: pts[hi], next: pts[hi] };
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (pts[mid].t < t) lo = mid + 1;
      else hi = mid;
    }
    return { pt: pts[lo], next: pts[Math.min(lo + 1, pts.length - 1)] };
  }

  function drawActors(ctx, view, w, h) {
    for (const actor of S.data.actors || []) {
      const hit = actorAt(actor, S.t);
      if (!hit) continue;
      const p = hit.pt;
      const [sx, sy] = w2s(p.x, p.y, view, w, h);
      const yaw = Math.atan2(hit.next.y - p.y, hit.next.x - p.x) || 0;
      const isTruck = String(actor.class).toUpperCase().indexOf("TRUCK") >= 0;
      const len = (isTruck ? 11.5 : 4.5) * view.scale;
      const wid = (isTruck ? 2.5 : 1.85) * view.scale;
      ctx.save();
      ctx.translate(sx, sy);
      ctx.rotate(yaw);
      ctx.fillStyle = isTruck ? COLORS.truck : COLORS.car;
      ctx.globalAlpha = 0.9;
      roundRect(ctx, -len / 2, -wid / 2, len, wid, Math.min(3, wid * 0.3));
      ctx.fill();
      ctx.globalAlpha = 1;
      ctx.strokeStyle = "rgba(0,0,0,0.35)";
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.restore();
      if (view.scale > 2.2) {
        ctx.font = "10px " + getComputedStyle(document.body).fontFamily;
        ctx.fillStyle = COLORS.text;
        ctx.fillText(actor.id, sx + 6, sy - 6);
      }
    }
  }

  function drawEgo(ctx, view, w, h) {
    const i = indexAt(S.t);
    const x = S.ts.x_m ? S.ts.x_m[i] : 0;
    const y = S.ts.y_m ? S.ts.y_m[i] : 0;
    const yaw = S.ts.yaw_rad ? S.ts.yaw_rad[i] : 0;
    const [sx, sy] = w2s(x, y, view, w, h);
    const len = 4.5 * view.scale;
    const wid = 1.85 * view.scale;
    const mode = modeName(S.ts.active_function ? S.ts.active_function[i] : 0);
    const halo = { NONE: "rgba(143,162,184,0.35)", HWA: "rgba(57,198,242,0.55)", LKA: "rgba(242,184,75,0.6)", ELK: "rgba(255,107,107,0.65)" }[mode];

    ctx.save();
    ctx.translate(sx, sy);
    ctx.rotate(yaw);
    // Halo showing the active function.
    ctx.beginPath();
    ctx.arc(0, 0, Math.max(10, len * 0.85), 0, Math.PI * 2);
    ctx.strokeStyle = halo;
    ctx.lineWidth = 2;
    ctx.stroke();
    // Body.
    ctx.fillStyle = COLORS.ego;
    roundRect(ctx, -len / 2, -wid / 2, len, wid, Math.min(3, wid * 0.28));
    ctx.fill();
    ctx.strokeStyle = COLORS.egoEdge;
    ctx.lineWidth = 1.2;
    ctx.stroke();
    // Windshield.
    ctx.fillStyle = "rgba(6, 30, 22, 0.75)";
    roundRect(ctx, len * 0.02, -wid * 0.36, len * 0.24, wid * 0.72, 1.5);
    ctx.fill();
    // Nose.
    ctx.beginPath();
    ctx.moveTo(len / 2, 0);
    ctx.lineTo(len / 2 - wid * 0.5, -wid * 0.42);
    ctx.lineTo(len / 2 - wid * 0.5, wid * 0.42);
    ctx.closePath();
    ctx.fillStyle = "#dff6ff";
    ctx.fill();
    ctx.restore();
  }

  function roundRect(ctx, x, y, w, h, r) {
    const rr = Math.min(r, Math.abs(w) / 2, Math.abs(h) / 2);
    ctx.beginPath();
    ctx.moveTo(x + rr, y);
    ctx.arcTo(x + w, y, x + w, y + h, rr);
    ctx.arcTo(x + w, y + h, x, y + h, rr);
    ctx.arcTo(x, y + h, x, y, rr);
    ctx.arcTo(x, y, x + w, y, rr);
    ctx.closePath();
  }

  function laneHudText() {
    const lanes = S.data && S.data.road && S.data.road.lanes;
    if (!lanes || !lanes.widths || !lanes.widths.length) return "";
    const total = lanes.carriageway && isNum(lanes.carriageway.width_m)
      ? lanes.carriageway.width_m
      : lanes.widths.reduce((a, b) => a + b, 0);
    const widths = lanes.widths.map((v) => v.toFixed(2)).join("/");
    let text = "lanes " + lanes.widths.length + " × " + widths + " m";
    if (lanes.opposing) text += " + " + lanes.opposing.count + " opposite";
    return text + " · " + total.toFixed(2) + " m";
  }

  function drawScene() {
    const canvas = $("scene");
    const { ctx, w, h } = setupCanvas(canvas);
    ctx.clearRect(0, 0, w, h);
    // Subtle grid.
    ctx.strokeStyle = COLORS.grid;
    ctx.lineWidth = 1;
    for (let x = 0; x < w; x += 48) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
    }
    for (let y = 0; y < h; y += 48) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    }

    const view = viewTransform(w, h);
    drawRoad(ctx, view, w, h);
    drawTrajectories(ctx, view, w, h);
    drawActors(ctx, view, w, h);
    drawEgo(ctx, view, w, h);

    const i = indexAt(S.t);
    const mode = modeName(S.ts.active_function ? S.ts.active_function[i] : 0);
    const zoomText = S.follow ? "zoom " + S.zoom.toFixed(2) + "×" : "fit";
    const laneText = laneHudText();
    $("scene-hud").innerHTML =
      "<span>" + (S.follow ? "follow" : "fit") + "</span>" +
      "<span>" + zoomText + "</span>" +
      "<span>" + view.scale.toFixed(2) + " px/m</span>" +
      (laneText ? "<span>" + laneText + "</span>" : "") +
      "<span>mode " + mode + "</span>";
  }

  /* ---------------------------------------------------------------- plots */

  function drawPlot(canvas, series, opts) {
    opts = opts || {};
    const { ctx, w, h } = setupCanvas(canvas);
    ctx.clearRect(0, 0, w, h);
    const hasRight = series.some((s) => s.axis === "right");
    const L = 38, R = hasRight ? 40 : 10, T = 14, B = 18;
    const pw = Math.max(10, w - L - R);
    const ph = Math.max(10, h - T - B);
    const dur = S.duration || 1;
    const tArr = times();

    let lmin = Infinity, lmax = -Infinity, rmin = Infinity, rmax = -Infinity;
    for (const s of series) {
      const vals = S.ts[s.key];
      if (!vals) continue;
      for (let i = 0; i < vals.length; i++) {
        let v = vals[i];
        if (!isNum(v)) continue;
        if (s.negate) v = -v;
        if (s.axis === "right") { rmin = Math.min(rmin, v); rmax = Math.max(rmax, v); }
        else { lmin = Math.min(lmin, v); lmax = Math.max(lmax, v); }
      }
    }
    if (!isFinite(lmin)) { lmin = -1; lmax = 1; }
    if (opts.includeZero !== false) { lmin = Math.min(lmin, 0); lmax = Math.max(lmax, 0); }
    if (lmax - lmin < 1e-6) { lmin -= 1; lmax += 1; }
    const padL = (lmax - lmin) * 0.08;
    lmin -= padL; lmax += padL;
    if (hasRight) {
      if (!isFinite(rmin)) { rmin = -1; rmax = 1; }
      if (opts.includeZero !== false) { rmin = Math.min(rmin, 0); rmax = Math.max(rmax, 0); }
      if (rmax - rmin < 1e-6) { rmin -= 1; rmax += 1; }
      const padR = (rmax - rmin) * 0.08;
      rmin -= padR; rmax += padR;
    }

    const yL = (v) => T + ph - ((v - lmin) / (lmax - lmin)) * ph;
    const yR = (v) => T + ph - ((v - rmin) / (rmax - rmin)) * ph;
    const xT = (t) => L + (clamp(t, 0, dur) / dur) * pw;

    // Grid + left labels.
    ctx.font = "10px " + getComputedStyle(document.body).fontFamily;
    ctx.textBaseline = "middle";
    ctx.strokeStyle = COLORS.grid;
    ctx.fillStyle = COLORS.text;
    for (let k = 0; k <= 4; k++) {
      const y = T + (ph * k) / 4;
      ctx.beginPath(); ctx.moveTo(L, y); ctx.lineTo(L + pw, y); ctx.stroke();
      const value = lmax - ((lmax - lmin) * k) / 4;
      ctx.textAlign = "right";
      ctx.fillText(fmt(value, Math.abs(value) < 1 ? 2 : 1), L - 6, y);
    }
    if (hasRight) {
      ctx.textAlign = "left";
      for (let k = 0; k <= 4; k++) {
        const y = T + (ph * k) / 4;
        const value = rmax - ((rmax - rmin) * k) / 4;
        ctx.fillText(fmt(value, Math.abs(value) < 1 ? 2 : 1), L + pw + 6, y);
      }
    }

    // Time ticks.
    const stepT = niceStep(dur / 5);
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (let t = 0; t <= dur + 1e-9; t += stepT) {
      const x = xT(t);
      ctx.beginPath(); ctx.moveTo(x, T); ctx.lineTo(x, T + ph); ctx.stroke();
      ctx.fillText(fmt(t, t < 10 ? 1 : 0), x, T + ph + 4);
    }

    // Series.
    for (const s of series) {
      const vals = S.ts[s.key];
      if (!vals) continue;
      ctx.beginPath();
      ctx.lineWidth = s.width || 1.6;
      ctx.strokeStyle = s.color;
      ctx.setLineDash(s.dash ? [5, 5] : []);
      const stride = Math.max(1, Math.floor(vals.length / (pw * 1.5)));
      let started = false;
      for (let i = 0; i < vals.length; i += stride) {
        let v = vals[i];
        if (!isNum(v)) continue;
        if (s.negate) v = -v;
        const x = xT(tArr[i]);
        const y = s.axis === "right" ? yR(v) : yL(v);
        if (!started) { ctx.moveTo(x, y); started = true; }
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Cursor.
    const cx = xT(S.t);
    ctx.strokeStyle = "rgba(57,198,242,0.85)";
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(cx, T); ctx.lineTo(cx, T + ph); ctx.stroke();

    // Legend.
    let lx = L + 4;
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    for (const s of series) {
      if (!s.label) continue;
      ctx.strokeStyle = s.color;
      ctx.lineWidth = 2;
      ctx.setLineDash(s.dash ? [4, 4] : []);
      ctx.beginPath(); ctx.moveTo(lx, T + 4); ctx.lineTo(lx + 12, T + 4); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = s.color;
      ctx.fillText(s.label, lx + 16, T - 1);
      lx += 20 + ctx.measureText(s.label).width + 8;
    }
  }

  function niceStep(raw) {
    const pow = Math.pow(10, Math.floor(Math.log10(Math.max(raw, 1e-6))));
    const norm = raw / pow;
    const step = norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 5 ? 5 : 10;
    return step * pow;
  }

  /* -------------------------------------------------------------- playback */

  function renderAll() {
    if (!S.data) return;
    const i = indexAt(S.t);
    drawScene();
    drawPlot($("plot-speed"), [
      { key: "speed_mps", color: COLORS.accent, label: "speed" },
      { key: "target_speed_mps", color: COLORS.warn, label: "target", dash: true },
    ], { includeZero: true });
    drawPlot($("plot-lateral"), [
      { key: "road_d_m", color: COLORS.accent, label: "d" },
      { key: "lane_margin_m", color: COLORS.good, label: "+margin", dash: true },
      { key: "lane_margin_m", color: COLORS.good, label: "−margin", negate: true, dash: true },
    ], { includeZero: true });
    drawPlot($("plot-steer"), [
      { key: "steering_command_rad", color: COLORS.accent, label: "steer" },
      { key: "lateral_acceleration_mps2", color: COLORS.warn, label: "ay", axis: "right" },
    ], { includeZero: true });
    updateKpis(i);
    updateTransport();
  }

  function updateTransport() {
    $("clock").textContent = fmt(S.t, 2) + " s";
    $("scrub").value = String(clamp(S.t / S.duration, 0, 1));
    const cursor = $("timeline-cursor");
    if (cursor) cursor.style.left = clamp(S.t / S.duration, 0, 1) * 100 + "%";
    $("btn-play").textContent = S.playing ? "❚❚ Pause" : "▶ Play";
  }

  function resizeAll() {
    // Canvases are sized from their CSS boxes on each draw; force a redraw.
    if (S.data) renderAll();
  }

  function play() {
    if (S.playing) return;
    if (S.t >= S.duration - 1e-6) S.t = 0;
    S.playing = true;
    S.lastFrame = performance.now();
    S.raf = requestAnimationFrame(frame);
    updateTransport();
  }

  function pause() {
    S.playing = false;
    if (S.raf) cancelAnimationFrame(S.raf);
    S.raf = 0;
    updateTransport();
  }

  function frame(now) {
    if (!S.playing) return;
    const dt = Math.min(0.1, (now - S.lastFrame) / 1000);
    S.lastFrame = now;
    S.t += dt * S.speed;
    if (S.t >= S.duration) {
      S.t = S.duration;
      S.playing = false;
    }
    renderAll();
    if (S.playing) S.raf = requestAnimationFrame(frame);
  }

  function step(delta) {
    pause();
    S.t = clamp(S.t + delta, 0, S.duration);
    renderAll();
  }

  /* -------------------------------------------------------------- handlers */

  function setFollow(value) {
    S.follow = value;
    $("btn-follow").classList.toggle("active", value);
    $("btn-fit").classList.toggle("active", !value);
    renderAll();
  }

  function setZoom(value) {
    S.zoom = clamp(isFinite(value) ? value : 1, ZOOM_MIN, ZOOM_MAX);
    if (!S.follow) {
      // Zooming is a "look closer at the car" gesture: leave fit mode.
      S.follow = true;
      $("btn-follow").classList.add("active");
      $("btn-fit").classList.remove("active");
    }
    renderAll();
  }

  function bind() {
    $("scenario-select").addEventListener("change", (ev) => tryLoadScenario(ev.target.value));
    $("btn-play").addEventListener("click", () => (S.playing ? pause() : play()));
    $("btn-back").addEventListener("click", () => step(-0.1));
    $("btn-fwd").addEventListener("click", () => step(0.1));
    $("scrub").addEventListener("input", (ev) => {
      // Read the value first: pause() repaints the transport and would
      // overwrite the slider with the current time.
      const fraction = parseFloat(ev.target.value);
      pause();
      S.t = clamp((isNaN(fraction) ? 0 : fraction) * S.duration, 0, S.duration);
      renderAll();
    });
    $("play-speed").addEventListener("input", (ev) => {
      S.speed = parseFloat(ev.target.value) || 1;
      $("play-speed-label").textContent = S.speed + "×";
    });
    $("btn-follow").addEventListener("click", () => setFollow(true));
    $("btn-fit").addEventListener("click", () => setFollow(false));

    $("scene").addEventListener(
      "wheel",
      (ev) => {
        ev.preventDefault();
        const factor = Math.exp(-ev.deltaY * 0.0016);
        setZoom(S.zoom * factor);
      },
      { passive: false },
    );

    document.addEventListener("keydown", (ev) => {
      if (ev.target && /^(INPUT|SELECT|TEXTAREA)$/.test(ev.target.tagName)) return;
      if (ev.code === "Space") { ev.preventDefault(); S.playing ? pause() : play(); }
      else if (ev.code === "ArrowRight") { ev.preventDefault(); step(0.1); }
      else if (ev.code === "ArrowLeft") { ev.preventDefault(); step(-0.1); }
      else if (ev.code === "Home") { pause(); S.t = 0; renderAll(); }
      else if (ev.code === "End") { pause(); S.t = S.duration; renderAll(); }
      else if (ev.key === "+" || ev.key === "=") { ev.preventDefault(); setZoom(S.zoom * 1.25); }
      else if (ev.key === "-" || ev.key === "_") { ev.preventDefault(); setZoom(S.zoom / 1.25); }
      else if (ev.key === "0") { ev.preventDefault(); setZoom(1); }
    });

    let resizeTimer = 0;
    window.addEventListener("resize", () => {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(resizeAll, 80);
    });

    // Drop an exported ui_data.json straight onto the page.
    ["dragover", "drop"].forEach((name) => {
      document.addEventListener(name, (ev) => {
        ev.preventDefault();
        if (name !== "drop") return;
        const file = ev.dataTransfer && ev.dataTransfer.files && ev.dataTransfer.files[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = () => {
          try {
            const payload = JSON.parse(String(reader.result));
            loadData(payload);
          } catch (err) {
            setStatus("could not parse dropped file", true);
          }
        };
        reader.readAsText(file);
      });
    });
  }

  /* ----------------------------------------------------------------- boot */

  function boot() {
    bind();
    if (window.SCENARIO_DATA && window.SCENARIO_DATA.scenario) {
      loadData(window.SCENARIO_DATA);
    } else {
      showEmpty();
      if (Array.isArray(window.SCENARIO_LIBRARY) && window.SCENARIO_LIBRARY.length) {
        buildSelector();
      }
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
