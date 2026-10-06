// Shop-floor page: layout editor, configuration tables, simulation with Gantt charts.
// Chart instances stay outside Alpine's reactive state (its proxies break Chart.js).
const shopCharts = {};
const DAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"];
const WEEK = 7 * 1440;

function clockMinutes(clock) {
  const [h, m] = clock.split(":").map(Number);
  return h * 60 + m;
}

function clockText(minutes) {
  return `${String(Math.floor(minutes / 60)).padStart(2, "0")}:${String(minutes % 60).padStart(2, "0")}`;
}

function startOffset(start) {
  return DAYS.indexOf(start.day) * 1440 + clockMinutes(start.time);
}

function splitWeekTime(minute, start) {
  const absolute = startOffset(start) + minute;
  const week = Math.floor(absolute / WEEK) + 1;
  const inWeek = absolute % WEEK;
  return { week, day: DAYS[Math.floor(inWeek / 1440)], time: clockText(inWeek % 1440) };
}

function joinWeekTime({ week, day, time }, start) {
  return Math.max(0, (week - 1) * WEEK + DAYS.indexOf(day) * 1440 + clockMinutes(time) - startOffset(start));
}

function normalize(config) {
  const c = clone(config);
  c.orders ||= [];
  c.start ||= { day: "mon", time: "06:00" };
  c.objective ||= { profile: "balanced", weights: {} };
  c.objective.weights ||= {};
  c.logistics.partial_shipments ??= false;
  for (const mt of c.machine_types) {
    mt.setup ||= { operators: 1, initial: 0, default: 0, times: [] };
    mt.setup.times ||= [];
  }
  for (const t of c.transformations) {
    t.setup_family ??= null;
    t.interruptible ??= true;
  }
  return c;
}

function floorSvg(config, selected) {
  const items = [
    ...config.layout.stores.map((s, i) => ({ kind: "store", index: i, name: s.name, sub: s.kind, pos: s.position })),
    ...config.machines.map((m, i) => ({ kind: "machine", index: i, name: m.name, sub: `${m.type} · in ${m.input_buffer} / out ${m.output_buffer}`, pos: m.position })),
  ];
  const xs = items.map((i) => i.pos[0]);
  const ys = items.map((i) => i.pos[1]);
  const minX = Math.min(0, ...xs) - 8;
  const minY = Math.min(0, ...ys) - 8;
  const width = Math.max(20, Math.max(...xs) - minX + 8);
  const height = Math.max(20, Math.max(...ys) - minY + 8);
  const font = Math.max(width, height) / 55;
  const box = font * 2.2;
  const shapes = items.map((item) => {
    const active = selected && selected.kind === item.kind && selected.index === item.index;
    const cls = item.kind === "store" ? `store ${item.sub}` : "machine";
    const [x, y] = item.pos;
    const shape = item.kind === "store"
      ? `<rect x="${x - box / 2}" y="${y - box / 2}" width="${box}" height="${box}" class="${cls}${active ? " selected" : ""}" />`
      : `<rect x="${x - box}" y="${y - box / 2}" width="${box * 2}" height="${box}" rx="${box / 5}" class="${cls}${active ? " selected" : ""}" />`;
    return `<g data-kind="${item.kind}" data-index="${item.index}" class="item">${shape}
      <text x="${x}" y="${y + box / 2 + font * 1.1}" font-size="${font}" text-anchor="middle">${item.name}</text>
      <text x="${x}" y="${y + box / 2 + font * 2.2}" font-size="${font * 0.75}" text-anchor="middle" class="sub">${item.sub}</text></g>`;
  });
  const grid = [];
  for (let gx = Math.ceil(minX / 10) * 10; gx < minX + width; gx += 10) grid.push(`<line x1="${gx}" y1="${minY}" x2="${gx}" y2="${minY + height}" />`);
  for (let gy = Math.ceil(minY / 10) * 10; gy < minY + height; gy += 10) grid.push(`<line x1="${minX}" y1="${gy}" x2="${minX + width}" y2="${gy}" />`);
  return `<svg viewBox="${minX} ${minY} ${width} ${height}" preserveAspectRatio="xMidYMid meet">
    <g class="grid">${grid.join("")}</g>${shapes.join("")}</svg>`;
}

document.addEventListener("alpine:init", () => {
  Alpine.data("shop", () => ({
    tabs: [
      { id: "layout", label: "Layout" },
      { id: "operations", label: "Parts & operations" },
      { id: "machines", label: "Machines & setup" },
      { id: "logistics", label: "Transport & staff" },
      { id: "orders", label: "Orders & objective" },
      { id: "simulation", label: "Simulation" },
    ],
    tab: location.hash.slice(1) || "layout",
    days: DAYS,
    toast: "",
    busy: false,
    shops: [],
    shopId: "",
    currentShopId: "",
    config: normalize({ name: "", layout: { stores: [] }, part_types: [], transformations: [], machine_types: [], machines: [], transport: { speed: 1, vehicles: [] }, staff: { shifts: [] }, logistics: { pickups: [] } }),
    saved: "",
    analysis: null,
    issues: [],
    profiles: {},
    weightLabels: {
      revenue: "per unit of revenue",
      material_cost: "per unit of material spend",
      holding_cost: "per part in progress or stored, per hour",
      lateness: "per unit of an order, per hour after its deadline",
      setup: "per setup hour",
      transport: "per vehicle hour",
    },
    selected: null,
    drag: null,
    sim: { policy: "pull", queue_limit: 2, minutes: null },
    result: null,
    kpiKeys: ["objective", "revenue", "material_cost", "orders_on_time", "late_unit_hours", "setup_hours", "transport_hours", "holding_part_hours"],
    machineHeight: 300,
    vehicleHeight: 120,

    async init() {
      window.addEventListener("hashchange", () => (this.tab = location.hash.slice(1) || "layout"));
      try {
        this.profiles = await api("GET", "/shop-profiles");
        this.shops = await api("GET", "/shops");
        if (this.shops.length) this.selectShop(this.shops[0].id);
      } catch (e) {
        this.notify(e);
      }
      this.$watch("config", debounce(() => this.validate(), 400));
      this.$watch("config", () => this.drawFloor());
      this.$watch("selected", () => this.drawFloor());
      this.$watch("tab", (tab) => tab === "layout" && this.$nextTick(() => this.drawFloor()));
      this.bindFloor();
    },

    notify(message) {
      this.toast = message instanceof Error ? message.message : message;
      setTimeout(() => (this.toast = ""), 6000);
    },

    fmt(v) {
      if (v === null || v === undefined) return "–";
      if (typeof v !== "number") return v;
      return Number.isInteger(v) ? v.toLocaleString() : v.toLocaleString(undefined, { maximumFractionDigits: 1 });
    },

    splitList(text) {
      return text.split(",").map((s) => s.trim()).filter(Boolean);
    },

    get dirty() {
      return JSON.stringify(this.config) !== this.saved;
    },

    get errorCount() {
      return this.issues.filter((i) => i.level === "error").length;
    },

    get partNames() {
      return this.config.part_types.map((p) => p.name).filter(Boolean);
    },

    get finalProducts() {
      return this.analysis?.final || this.partNames;
    },

    kindOf(name) {
      if (!this.analysis) return "";
      if (this.analysis.raw.includes(name)) return "raw";
      if (this.analysis.final.includes(name)) return "final";
      return "intermediate";
    },

    weekTime(minute) {
      return splitWeekTime(minute ?? 0, this.config.start);
    },

    setWeekTime(order, field, change) {
      order[field] = joinWeekTime({ ...this.weekTime(order[field]), ...change }, this.config.start);
    },

    formatMinute(minute) {
      const { week, day, time } = this.weekTime(Math.round(minute));
      return `${week > 1 ? `W${week} ` : ""}${day} ${time}`;
    },

    toggleDay(item, day) {
      item.days = item.days.includes(day) ? item.days.filter((d) => d !== day) : DAYS.filter((d) => d === day || item.days.includes(d));
    },

    addSetupTime(mt) {
      const families = this.analysis?.families?.[mt.name] || [];
      mt.setup.times.push({ from: families[0], to: families[1] ?? families[0], minutes: mt.setup.default });
    },

    // ----- shops

    selectShop(id) {
      const shop = this.shops.find((s) => s.id === id);
      if (!shop || (this.dirty && this.saved && !confirm("Discard unsaved changes?"))) {
        this.$nextTick(() => (this.shopId = this.currentShopId));
        return;
      }
      this.shopId = this.currentShopId = id;
      this.setConfig(shop.config);
    },

    setConfig(config) {
      this.config = normalize(config);
      this.saved = JSON.stringify(this.config);
      this.selected = null;
      this.result = null;
      this.validate();
      this.$nextTick(() => this.drawFloor());
    },

    async save() {
      try {
        const shop = this.shopId ? await api("PUT", `/shops/${this.shopId}`, this.config) : await api("POST", "/shops", this.config);
        this.shops = await api("GET", "/shops");
        this.shopId = this.currentShopId = shop.id;
        this.saved = JSON.stringify(this.config);
        this.notify(`Saved ${shop.name}.`);
      } catch (e) {
        this.notify(e);
      }
    },

    duplicate() {
      const copy = clone(this.config);
      copy.name = `${copy.name} copy`;
      this.shopId = this.currentShopId = "";
      this.config = copy;
      this.saved = "";
    },

    async remove() {
      if (!confirm(`Delete shop ${this.config.name}?`)) return;
      try {
        await api("DELETE", `/shops/${this.shopId}`);
        this.saved = "";
        this.shops = await api("GET", "/shops");
        if (this.shops.length) this.selectShop(this.shops[0].id);
      } catch (e) {
        this.notify(e);
      }
    },

    exportJson() {
      download(`${this.config.name || "shop"}.json`, new Blob([JSON.stringify(this.config, null, 2)], { type: "application/json" }));
    },

    async importJson(event) {
      const file = event.target.files[0];
      event.target.value = "";
      if (!file) return;
      try {
        this.setConfig(JSON.parse(await file.text()));
        this.shopId = this.currentShopId = "";
        this.saved = "";
      } catch (e) {
        this.notify(`Import failed: ${e.message}`);
      }
    },

    async validate() {
      try {
        this.analysis = await api("POST", "/shops/validate", this.config);
        this.issues = this.analysis.issues;
      } catch (e) {
        this.issues = e.message.split("\n").map((message) => ({ level: "error", message }));
      }
    },

    // ----- layout

    addMachine() {
      const type = this.config.machine_types[0]?.name || "";
      this.config.machines.push({ name: `machine-${this.config.machines.length + 1}`, type, position: [10, 10], input_buffer: 4, output_buffer: 4 });
      this.selected = { kind: "machine", index: this.config.machines.length - 1 };
    },

    addStore() {
      this.config.layout.stores.push({ name: `store-${this.config.layout.stores.length + 1}`, kind: "intermediate", position: [5, 5], capacity: null });
      this.selected = { kind: "store", index: this.config.layout.stores.length - 1 };
    },

    drawFloor() {
      const el = document.getElementById("floor");
      if (el && this.config.layout) el.innerHTML = floorSvg(this.config, this.selected);
    },

    bindFloor() {
      const el = document.getElementById("floor");
      const point = (event) => {
        const svg = el.querySelector("svg");
        const p = svg.createSVGPoint();
        p.x = event.clientX;
        p.y = event.clientY;
        return p.matrixTransform(svg.getScreenCTM().inverse());
      };
      el.addEventListener("pointerdown", (event) => {
        const g = event.target.closest("g.item");
        if (!g) return;
        const kind = g.dataset.kind;
        const index = Number(g.dataset.index);
        this.selected = { kind, index };
        const item = kind === "machine" ? this.config.machines[index] : this.config.layout.stores[index];
        const p = point(event);
        this.drag = { item, dx: item.position[0] - p.x, dy: item.position[1] - p.y };
        el.setPointerCapture(event.pointerId);
      });
      el.addEventListener("pointermove", (event) => {
        if (!this.drag) return;
        const p = point(event);
        this.drag.item.position = [Math.round(p.x + this.drag.dx), Math.round(p.y + this.drag.dy)];
      });
      el.addEventListener("pointerup", () => (this.drag = null));
    },

    // ----- simulation

    async run() {
      this.busy = true;
      try {
        const body = { config: this.config, policy: this.sim.policy, queue_limit: this.sim.queue_limit || 2 };
        if (this.sim.minutes) body.minutes = this.sim.minutes;
        const sim = await api("POST", "/shop-simulations", body);
        this.result = sim.result;
        this.$nextTick(() => this.drawResult());
      } catch (e) {
        this.notify(e);
      } finally {
        this.busy = false;
      }
    },

    drawResult() {
      const r = this.result;
      for (const chart of Object.values(shopCharts)) chart.destroy();
      const tick = (value) => this.formatMinute(value);
      // Show the span with activity (plus an hour), ticks every 2 or 12 hours from the planning start.
      const lastActivity = Math.max(0, ...r.jobs.map((j) => j.finished ?? r.minutes), ...r.trips.map((t) => t.end));
      const span = Math.min(r.minutes, lastActivity + 60);
      const step = span <= 2 * 1440 ? 120 : 720;
      const unstaffed = this.unstaffedBands(span);

      // Machines: one lane per slot, bars for setup, processing and blocked time.
      const lanes = {};
      const rows = [];
      const bars = { setup: [], run: [], blocked: [] };
      for (const job of [...r.jobs].filter((j) => j.run_start !== null || j.setup_start !== null).sort((a, b) => (a.setup_start ?? a.run_start) - (b.setup_start ?? b.run_start))) {
        const begin = job.setup_start ?? job.run_start;
        const end = job.finished ?? r.minutes;
        const machineLanes = (lanes[job.machine] ||= []);
        let lane = machineLanes.findIndex((free) => free <= begin);
        if (lane === -1) lane = machineLanes.push(0) - 1;
        machineLanes[lane] = end;
        const row = `${job.machine} #${lane + 1}`;
        if (!rows.includes(row)) rows.push(row);
        if (job.setup_start !== null) bars.setup.push({ x: [job.setup_start, job.run_start ?? end], y: row, job });
        if (job.run_start !== null) bars.run.push({ x: [job.run_start, job.run_end ?? end], y: row, job });
        if (job.run_end !== null && end > job.run_end) bars.blocked.push({ x: [job.run_end, end], y: row, job });
      }
      rows.sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
      this.machineHeight = 60 + rows.length * 20;
      const gantt = (id, labels, datasets, tooltip) =>
        new Chart(document.getElementById(id), {
          type: "bar",
          data: { labels, datasets },
          options: {
            indexAxis: "y",
            animation: false,
            maintainAspectRatio: false,
            scales: {
              x: {
                type: "linear",
                min: 0,
                max: span,
                ticks: { callback: tick },
                afterBuildTicks: (axis) => {
                  axis.ticks = Array.from({ length: Math.floor(span / step) + 1 }, (_, k) => ({ value: k * step }));
                },
              },
              y: { ticks: { autoSkip: false, font: { size: 10 } } },
            },
            plugins: { legend: { display: false }, tooltip: { callbacks: { label: tooltip } } },
          },
          plugins: [unstaffed],
        });
      this.$nextTick(() => {
        shopCharts.machines = gantt(
          "machine-gantt",
          rows,
          [
            { data: bars.setup, backgroundColor: cssVar("--warning"), grouped: false },
            { data: bars.run, backgroundColor: (c) => colorFor(c.raw?.job.output || ""), grouped: false },
            { data: bars.blocked, backgroundColor: cssVar("--error"), grouped: false },
          ],
          (c) => {
            const j = c.raw.job;
            const what = ["setup", "processing", "blocked"][c.datasetIndex];
            return `${j.transformation} (${what}) ${this.formatMinute(c.raw.x[0])}–${this.formatMinute(c.raw.x[1])}${j.paused ? `, paused ${j.paused} min` : ""}`;
          },
        );
        const vehicleRows = r.vehicles;
        this.vehicleHeight = 60 + vehicleRows.length * 24;
        shopCharts.vehicles = gantt(
          "vehicle-gantt",
          vehicleRows,
          [{ data: r.trips.map((t) => ({ x: [t.start, t.end], y: t.vehicle, trip: t })), backgroundColor: (c) => colorFor(c.raw?.trip.dst || ""), grouped: false }],
          (c) => {
            const t = c.raw.trip;
            const parts = Object.entries(t.parts).map(([p, n]) => `${n} ${p}`).join(", ");
            return `${t.src} → ${t.dst}: ${parts} (${this.formatMinute(t.start)}–${this.formatMinute(t.end)})`;
          },
        );
        const names = Object.keys(r.kpis.utilization);
        shopCharts.utilisation = new Chart(document.getElementById("utilisation"), {
          type: "bar",
          data: { labels: names, datasets: [{ data: names.map((n) => 100 * r.kpis.utilization[n]), backgroundColor: cssVar("--accent") }] },
          options: { indexAxis: "y", animation: false, maintainAspectRatio: false, scales: { x: { min: 0, max: 100, title: { display: true, text: "%" } } }, plugins: { legend: { display: false } } },
        });
      });
    },

    unstaffedBands(minutes) {
      // Chart.js plugin drawing grey bands where no shift has workers.
      const offset = startOffset(this.config.start);
      const workers = new Array(WEEK).fill(0);
      for (const s of this.config.staff.shifts) {
        for (const d of s.days) {
          for (let m = clockMinutes(s.start); m < clockMinutes(s.end); m++) workers[DAYS.indexOf(d) * 1440 + m] += s.workers;
        }
      }
      const bands = [];
      let open = null;
      for (let t = 0; t <= minutes; t++) {
        const idle = t < minutes && workers[(offset + t) % WEEK] === 0;
        if (idle && open === null) open = t;
        if (!idle && open !== null) {
          bands.push([open, t]);
          open = null;
        }
      }
      return {
        id: "unstaffed",
        beforeDatasetsDraw(chart) {
          const { ctx, chartArea, scales } = chart;
          ctx.save();
          ctx.fillStyle = "rgba(128, 128, 128, 0.12)";
          for (const [a, b] of bands) {
            const x1 = scales.x.getPixelForValue(a);
            const x2 = scales.x.getPixelForValue(b);
            ctx.fillRect(x1, chartArea.top, x2 - x1, chartArea.bottom - chartArea.top);
          }
          ctx.restore();
        },
      };
    },
  }));
});
