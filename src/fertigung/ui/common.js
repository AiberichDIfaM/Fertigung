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

