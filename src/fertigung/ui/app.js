// Chart and graph instances live outside Alpine's reactive state; its proxies break both libraries.
const charts = {};
let graph = null;

const KEY_STORAGE = "fertigung-api-key";

function storedKey() {
  try {
    return localStorage.getItem(KEY_STORAGE) || "";
  } catch {
    return "";
  }
}

function storeKey(key) {
  try {
    localStorage.setItem(KEY_STORAGE, key);
  } catch {
    // Without storage the key is asked for again on the next load.
  }
  sessionKey = key;
}

let sessionKey = storedKey();

async function request(method, path, body, retry = true) {
  const sentKey = sessionKey;
  const headers = sentKey ? { "X-API-Key": sentKey } : {};
  if (body !== undefined) headers["Content-Type"] = "application/json";
  const res = await fetch(path, { method, headers, body: body === undefined ? undefined : JSON.stringify(body) });
  if (res.status === 401 && retry) {
    // Parallel requests fail together; only the first one asks, the others retry with the new key.
    const key = sessionKey !== sentKey ? sessionKey : prompt("This server requires an API key:");
    if (key) {
      storeKey(key.trim());
      return request(method, path, body, false);
    }
  }
  return res;
}

async function api(method, path, body) {
  const res = await request(method, path, body);
  if (res.status === 204) return null;
  const data = await res.json().catch(() => null);
  if (!res.ok) throw new Error(errorMessage(data) || res.statusText);
  return data;
}

function errorMessage(data) {
  const detail = data?.detail;
  if (Array.isArray(detail)) return detail.map((e) => `${e.loc.slice(1).join(".")}: ${e.msg}`).join("\n");
  return detail;
}

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

function debounce(fn, ms) {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), ms);
  };
}

function colorFor(name) {
  let h = 0;
  for (const c of name) h = (h * 31 + c.charCodeAt(0)) % 360;
  return `hsl(${150 + (h % 150)}, 50%, 55%)`;
}

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function download(filename, blob) {
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = filename;
  link.click();
  URL.revokeObjectURL(link.href);
}

document.addEventListener("alpine:init", () => {
  Alpine.data("app", () => ({
    tabs: [
      { id: "plant", label: "Plant" },
      { id: "orders", label: "Orders & reward" },
      { id: "training", label: "Training" },
      { id: "simulation", label: "Simulation" },
      { id: "models", label: "Models" },
    ],
    tab: location.hash.slice(1) || "plant",
    toast: "",
    busy: false,
    schemas: {},
    plants: [],
    plantId: "",
    currentPlantId: "",
    config: { name: "", buffer_capacity: 10, part_types: [], transformations: [], machine_types: [], machines: [], orders: [], reward: {} },
    saved: "",
    analysis: null,
    issues: [],
    jobs: [],
    selectedJobId: null,
    selectedJob: null,
    training: {},
    trainingFields: ["timesteps", "seed", "n_envs", "horizon", "learning_rate", "ent_coef", "pretrain", "pretrain_episodes", "pretrain_epochs"],
    models: [],
    sim: { policy: "pull", model_id: "", ticks: null, seed: 0 },
    simResult: null,
    evaluation: null,
    kpiKeys: ["time", "revenue", "material_cost", "profit", "orders_on_time", "total_lateness", "utilization", "avg_buffer"],
    metricKeys: ["reward", "reward_unshaped", "profit", "shipped", "orders_on_time", "total_lateness", "utilization", "avg_buffer"],
    ganttHeight: 300,

    async init() {
      window.addEventListener("hashchange", () => (this.tab = location.hash.slice(1) || "plant"));
      this.$watch("tab", (tab) => tab === "plant" && this.$nextTick(() => graph?.resize().fit()));
      try {
        const openapi = await api("GET", "/openapi.json");
        this.schemas = openapi.components.schemas;
        this.resetTraining();
        await Promise.all([this.loadPlants(), this.loadModels(), this.loadJobs()]);
        if (this.plants.length) this.selectPlant(this.plants[0].id);
      } catch (e) {
        this.notify(e);
      }
      this.$watch("config", debounce(() => this.validate(), 400));
      setInterval(() => this.pollJobs(), 3000);
    },

    notify(message) {
      this.toast = message instanceof Error ? message.message : message;
      setTimeout(() => (this.toast = ""), 6000);
    },

    fmt(v) {
      if (v === null || v === undefined) return "–";
      if (typeof v !== "number") return v;
      return Number.isInteger(v) ? v.toLocaleString() : v.toLocaleString(undefined, { maximumFractionDigits: 2 });
    },

    splitList(text) {
      return text.split(",").map((s) => s.trim()).filter(Boolean);
    },

    schemaProps(name) {
      return this.schemas[name]?.properties || {};
    },

    get dirty() {
      return JSON.stringify(this.config) !== this.saved;
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
      if (this.analysis.intermediate.includes(name)) return "intermediate";
      return "";
    },

    // Plants
    async loadPlants() {
      this.plants = await api("GET", "/plants");
    },

    selectPlant(id) {
      const plant = this.plants.find((p) => p.id === id);
      if (!plant || (this.dirty && this.saved && !confirm("Discard unsaved changes?"))) {
        this.$nextTick(() => (this.plantId = this.currentPlantId));
        return;
      }
      this.plantId = this.currentPlantId = id;
      this.setConfig(plant.config);
    },

    setConfig(config) {
      const reward = Object.fromEntries(Object.entries(this.schemaProps("RewardConfig")).map(([k, v]) => [k, v.default]));
      this.config = { orders: [], ...clone(config), reward: { ...reward, ...(config.reward || {}) } };
      this.saved = JSON.stringify(this.config);
      this.validate();
    },

    async savePlant() {
      try {
        const plant = this.plantId
          ? await api("PUT", `/plants/${this.plantId}`, this.config)
          : await api("POST", "/plants", this.config);
        await this.loadPlants();
        this.plantId = this.currentPlantId = plant.id;
        this.saved = JSON.stringify(this.config);
        this.notify(`Saved ${plant.name}.`);
      } catch (e) {
        this.notify(e);
      }
    },

    duplicatePlant() {
      const copy = clone(this.config);
      copy.name = `${copy.name} copy`;
      this.plantId = this.currentPlantId = "";
      this.config = copy;
      this.saved = "";
    },

    async deletePlant() {
      if (!confirm(`Delete plant ${this.config.name}?`)) return;
      try {
        await api("DELETE", `/plants/${this.plantId}`);
        this.saved = "";
        await this.loadPlants();
        if (this.plants.length) this.selectPlant(this.plants[0].id);
      } catch (e) {
        this.notify(e);
      }
    },

    exportPlant() {
      download(`${this.config.name || "plant"}.json`, new Blob([JSON.stringify(this.config, null, 2)], { type: "application/json" }));
    },

    async importPlant(event) {
      const file = event.target.files[0];
      event.target.value = "";
      if (!file) return;
      try {
        this.setConfig(JSON.parse(await file.text()));
        this.plantId = this.currentPlantId = "";
        this.saved = "";
      } catch (e) {
        this.notify(`Import failed: ${e.message}`);
      }
    },

    async validate() {
      try {
        this.analysis = await api("POST", "/plants/validate", this.config);
        this.issues = this.analysis.issues;
        this.drawGraph();
      } catch (e) {
        this.issues = e.message.split("\n").map((message) => ({ level: "error", message }));
      }
    },

    drawGraph() {
      const a = this.analysis;
      const kind = (p) => (a.raw.includes(p) ? "raw" : a.final.includes(p) ? "final" : "intermediate");
      const used = new Set(a.edges.flatMap((e) => [e.input, e.output]));
      const label = (p) => {
        const cost = a.material_costs[p];
        return cost === null ? p : `${p}\n${this.fmt(cost)}`;
      };
      const elements = [
        ...[...used].map((p) => ({ data: { id: p, label: label(p) }, classes: kind(p) })),
        ...a.edges.map((e, i) => ({
          data: { id: `e${i}`, source: e.input, target: e.output, label: e.count > 1 ? `${e.transformation} ×${e.count}` : e.transformation },
        })),
      ];
      const style = [
        { selector: "node", style: { label: "data(label)", "text-wrap": "wrap", "text-valign": "center", "font-size": 10, width: 44, height: 44, color: "#fff", "background-color": cssVar("--intermediate") } },
        { selector: ".raw", style: { "background-color": cssVar("--raw") } },
        { selector: ".final", style: { "background-color": cssVar("--final"), shape: "round-rectangle", width: 56 } },
        { selector: "edge", style: { "curve-style": "bezier", "target-arrow-shape": "triangle", width: 1.5, "line-color": cssVar("--muted"), "target-arrow-color": cssVar("--muted"), label: "data(label)", "font-size": 8, color: cssVar("--muted"), "text-rotation": "autorotate" } },
      ];
      const roots = [...used].filter((p) => kind(p) === "raw");
      if (graph) graph.destroy();
      graph = cytoscape({
        container: document.getElementById("graph"),
        elements,
        style,
        layout: { name: "breadthfirst", directed: true, roots, spacingFactor: 1.1 },
        wheelSensitivity: 0.3,
      });
    },

    // Training
    resetTraining() {
      const props = this.schemaProps("TrainingConfig");
      this.training = Object.fromEntries(this.trainingFields.map((k) => [k, props[k]?.default ?? null]));
    },

    async startTraining() {
      const training = { ...this.training, pretrain: this.training.pretrain || null };
      for (const key of Object.keys(training)) if (training[key] === "") training[key] = null;
      try {
        const job = await api("POST", "/training-jobs", { plant_id: this.plantId, training });
        await this.loadJobs();
        this.selectJob(job.id);
      } catch (e) {
        this.notify(e);
      }
    },

    async loadJobs() {
      this.jobs = await api("GET", "/training-jobs");
    },

    async pollJobs() {
      const active = this.jobs.some((j) => ["queued", "running", "cancelling"].includes(j.status));
      if (!active && this.selectedJob?.status !== "running") return;
      await this.loadJobs();
      if (this.selectedJobId) await this.selectJob(this.selectedJobId);
      if (this.jobs.some((j) => j.status === "completed" && !this.models.some((m) => m.id === j.model_id))) {
        await this.loadModels();
      }
    },

    async selectJob(id) {
      this.selectedJobId = id;
      this.selectedJob = await api("GET", `/training-jobs/${id}`);
      this.drawCurve(this.selectedJob.curve);
    },

    async cancelJob(id) {
      try {
        await api("DELETE", `/training-jobs/${id}`);
        await this.loadJobs();
      } catch (e) {
        this.notify(e);
      }
    },

    drawCurve(curve) {
      const series = (key) => curve.filter((p) => key in p).map((p) => ({ x: p.timesteps, y: p[key] }));
      const datasets = [
        { label: "training episodes", data: series("train_reward"), borderColor: cssVar("--accent"), pointRadius: 0 },
        { label: "evaluation", data: series("eval_reward"), borderColor: cssVar("--final"), backgroundColor: cssVar("--final") },
      ];
      if (charts.curve) {
        charts.curve.data.datasets.forEach((d, i) => (d.data = datasets[i].data));
        charts.curve.update("none");
        return;
      }
      charts.curve = new Chart(document.getElementById("curve"), {
        type: "line",
        data: { datasets },
        options: {
          animation: false,
          maintainAspectRatio: false,
          scales: { x: { type: "linear", title: { display: true, text: "timesteps" } }, y: { title: { display: true, text: "episode reward" } } },
        },
      });
    },

    // Simulation
    request() {
      const body = { config: this.config, seed: this.sim.seed || 0 };
      if (this.sim.ticks) body.ticks = this.sim.ticks;
      if (this.sim.policy === "model") body.model_id = this.sim.model_id;
      return body;
    },

    async runSimulation() {
      if (this.sim.policy === "model" && !this.sim.model_id) return this.notify("Select a model.");
      this.busy = true;
      try {
        const sim = await api("POST", "/simulations", { ...this.request(), policy: this.sim.policy });
        this.simResult = sim.result;
        this.$nextTick(() => this.drawGantt(sim.result, sim.request.ticks));
      } catch (e) {
        this.notify(e);
      } finally {
        this.busy = false;
      }
    },

    async compare() {
      this.busy = true;
      try {
        const body = this.request();
        if (this.sim.policy !== "model") delete body.model_id;
        this.evaluation = await api("POST", "/evaluations", body);
      } catch (e) {
        this.notify(e);
      } finally {
        this.busy = false;
      }
    },

    drawGantt(result, ticks) {
      // Assign each job to the first free lane of its machine so parallel slots do not overlap.
      const lanes = {};
      const rows = [];
      const processing = [];
      const blocked = [];
      for (const bar of [...result.gantt].sort((a, b) => a.start - b.start)) {
        const end = bar.end ?? ticks;
        const machineLanes = (lanes[bar.machine] ||= []);
        let lane = machineLanes.findIndex((free) => free <= bar.start);
        if (lane === -1) lane = machineLanes.push(0) - 1;
        machineLanes[lane] = end;
        const row = `${bar.machine} #${lane + 1}`;
        if (!rows.includes(row)) rows.push(row);
        processing.push({ x: [bar.start, bar.blocked_from ?? end], y: row, bar });
        if (bar.blocked_from !== null) blocked.push({ x: [bar.blocked_from, end], y: row, bar });
      }
      rows.sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
      this.ganttHeight = 40 + rows.length * 18;
      charts.gantt?.destroy();
      this.$nextTick(() => {
        charts.gantt = new Chart(document.getElementById("gantt"), {
          type: "bar",
          data: {
            labels: rows,
            datasets: [
              { label: "processing", data: processing, backgroundColor: (c) => colorFor(c.raw?.bar.output || ""), grouped: false },
              { label: "blocked", data: blocked, backgroundColor: cssVar("--error"), grouped: false },
            ],
          },
          options: {
            indexAxis: "y",
            animation: false,
            maintainAspectRatio: false,
            scales: { x: { type: "linear", min: 0, max: ticks, title: { display: true, text: "tick" } }, y: { ticks: { autoSkip: false, font: { size: 10 } } } },
            plugins: {
              legend: { display: false },
              tooltip: {
                callbacks: {
                  label: (c) => {
                    const b = c.raw.bar;
                    return `${b.transformation} → ${b.output}, ticks ${c.raw.x[0]}–${c.raw.x[1]}${c.datasetIndex ? " (blocked)" : ""}`;
                  },
                },
              },
            },
          },
        });
      });
    },

    // Models
    async loadModels() {
      this.models = await api("GET", "/models");
      if (!this.sim.model_id && this.models.length) this.sim.model_id = this.models[0].id;
    },

    async downloadModel(model) {
      const res = await request("GET", `/models/${model.id}/download`);
      if (!res.ok) return this.notify(`Download failed: ${res.statusText}`);
      download(`${model.name.replaceAll(" ", "_")}.zip`, await res.blob());
    },

    simulateModel(model) {
      this.sim.policy = "model";
      this.sim.model_id = model.id;
      location.hash = "simulation";
      this.runSimulation();
    },

    async deleteModel(model) {
      if (!confirm(`Delete model ${model.name}?`)) return;
      try {
        await api("DELETE", `/models/${model.id}`);
        await this.loadModels();
      } catch (e) {
        this.notify(e);
      }
    },
  }));
});
