// Helpers shared by the plant page (app.js) and the shop-floor page (shop.js).
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


// Merges objects with their getters intact (object spread would evaluate them once).
function component(...parts) {
  return parts.reduce((target, part) => Object.defineProperties(target, Object.getOwnPropertyDescriptors(part)), {});
}

// Stored configurations under `path` (e.g. "/plants") with validation, import and export.
// The page provides setConfig(config) and optionally afterValidate().
function configEditor(path, label) {
  return {
    toast: "",
    busy: false,
    records: [],
    recordId: "",
    currentRecordId: "",
    saved: "",
    analysis: null,
    issues: [],

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
      if (this.analysis.intermediate.includes(name)) return "intermediate";
      return "";
    },

    async loadRecords() {
      this.records = await api("GET", path);
    },

    selectRecord(id) {
      const record = this.records.find((r) => r.id === id);
      if (!record || (this.dirty && this.saved && !confirm("Discard unsaved changes?"))) {
        this.$nextTick(() => (this.recordId = this.currentRecordId));
        return;
      }
      this.recordId = this.currentRecordId = id;
      this.setConfig(record.config);
    },

    async save() {
      try {
        const record = this.recordId ? await api("PUT", `${path}/${this.recordId}`, this.config) : await api("POST", path, this.config);
        await this.loadRecords();
        this.recordId = this.currentRecordId = record.id;
        this.saved = JSON.stringify(this.config);
        this.notify(`Saved ${record.name}.`);
      } catch (e) {
        this.notify(e);
      }
    },

    duplicate() {
      const copy = clone(this.config);
      copy.name = `${copy.name} copy`;
      this.recordId = this.currentRecordId = "";
      this.config = copy;
      this.saved = "";
    },

    async remove() {
      if (!confirm(`Delete ${label} ${this.config.name}?`)) return;
      try {
        await api("DELETE", `${path}/${this.recordId}`);
        this.saved = "";
        await this.loadRecords();
        if (this.records.length) this.selectRecord(this.records[0].id);
      } catch (e) {
        this.notify(e);
      }
    },

    exportJson() {
      download(`${this.config.name || label}.json`, new Blob([JSON.stringify(this.config, null, 2)], { type: "application/json" }));
    },

    async importJson(event) {
      const file = event.target.files[0];
      event.target.value = "";
      if (!file) return;
      try {
        this.setConfig(JSON.parse(await file.text()));
        this.recordId = this.currentRecordId = "";
        this.saved = "";
      } catch (e) {
        this.notify(`Import failed: ${e.message}`);
      }
    },

    async validate() {
      try {
        this.analysis = await api("POST", `${path}/validate`, this.config);
        this.issues = this.analysis.issues;
        this.afterValidate?.();
      } catch (e) {
        this.issues = e.message.split("\n").map((message) => ({ level: "error", message }));
      }
    },
  };
}
