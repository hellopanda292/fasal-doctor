// Fasal Doctor: runs the crop-disease model in the browser with ONNX Runtime Web.
const SIZE = 224;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];
const UNSURE_BELOW = 0.6; // top probability under this shows the "retake the photo" message
const CROP_EMOJI = { Potato: "🥔", Tomato: "🍅", Maize: "🌽" };

const $ = (id) => document.getElementById(id);
const state = { session: null, labels: [], crop: "", logits: null, pending: null };

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";

// ---------- model loading with a progress bar ----------
async function fetchWithProgress(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`);
  const total = Number(res.headers.get("content-length")) || 0;
  if (!res.body || !total) return new Uint8Array(await res.arrayBuffer());
  const reader = res.body.getReader();
  const out = new Uint8Array(total);
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    out.set(value, got);
    got += value.length;
    onProgress(got / total);
  }
  return out;
}

async function init() {
  try {
    const [labels, metrics] = await Promise.all([
      fetch("model/labels.json").then((r) => r.json()),
      fetch("model/metrics.json").then((r) => r.json()),
    ]);
    state.labels = labels;
    renderInfo(labels, metrics);
    loadSamples();
    const bytes = await fetchWithProgress("model/fasal_doctor.onnx", (f) => {
      $("load-bar").style.width = `${Math.round(f * 100)}%`;
      $("load-pct").textContent = `${Math.round(f * 100)}%`;
    });
    state.session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
    $("loading").hidden = true;
    if (state.pending) classify(state.pending);
    else $("placeholder").hidden = false;
  } catch (err) {
    console.error(err);
    $("loading").hidden = true;
    showError("मॉडल लोड नहीं हो पाया। इंटरनेट जाँचकर पेज दोबारा खोलें। (The model could not be loaded.)");
  }
}

function renderInfo(labels, m) {
  const stats = [
    [`${(m.test_accuracy * 100).toFixed(1)}%`, "accuracy on unseen leaves"],
    [m.test_images.toLocaleString("en-IN"), "test photos"],
    [m.classes, "classes, 3 crops"],
    ["~0.1 s", "per photo, on device"],
  ];
  $("stats").innerHTML = stats.map(([v, l]) => `<div class="stat"><strong>${v}</strong><span>${l}</span></div>`).join("");
  $("classes").innerHTML = labels
    .map((l) => `<span>${CROP_EMOJI[l.crop]} ${l.disease_hi}</span>`)
    .join("");
}

async function loadSamples() {
  try {
    const list = await fetch("samples/samples.json").then((r) => r.json());
    $("samples").innerHTML = list
      .map((s, i) => `<button class="sample" type="button" data-i="${i}"><img src="samples/${s.file}" alt="" loading="lazy" /><span>${s.label}</span></button>`)
      .join("");
    $("samples").querySelectorAll(".sample").forEach((b) =>
      b.addEventListener("click", () => {
        const s = list[Number(b.dataset.i)];
        state.logits = null; // the old result belongs to the old photo
        setCrop(s.crop);
        loadImage(`samples/${s.file}`);
      }),
    );
  } catch {
    $("samples").parentElement.hidden = true;
  }
}

// ---------- image input ----------
function loadImage(src) {
  state.logits = null;
  $("result").hidden = true;
  const img = new Image();
  img.onload = () => {
    $("preview").src = src;
    $("preview").hidden = false;
    $("empty").hidden = true;
    if (state.session) classify(img);
    else state.pending = img;
  };
  img.onerror = () => showError("यह फोटो खुल नहीं पाई। कोई दूसरी फोटो चुनें। (Could not open this image.)");
  img.src = src;
}

function onFile(file) {
  if (!file || !file.type.startsWith("image/")) return;
  loadImage(URL.createObjectURL(file));
}

$("camera").addEventListener("change", (e) => onFile(e.target.files[0]));
$("gallery").addEventListener("change", (e) => onFile(e.target.files[0]));
const drop = $("drop");
drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (e) => {
  e.preventDefault();
  drop.classList.remove("over");
  onFile(e.dataTransfer.files[0]);
});

// ---------- crop filter ----------
function setCrop(crop) {
  state.crop = crop;
  document.querySelectorAll(".chip").forEach((c) => c.setAttribute("aria-pressed", String(c.dataset.crop === crop)));
  if (state.logits) render();
}
document.querySelectorAll(".chip").forEach((c) => c.addEventListener("click", () => setCrop(c.dataset.crop)));

// ---------- preprocessing: centre square, resized to 224x224, normalised like torchvision ----------
function toTensor(img, flip) {
  const side = Math.min(img.naturalWidth, img.naturalHeight);
  let canvas = document.createElement("canvas");
  canvas.width = canvas.height = side;
  canvas.getContext("2d").drawImage(img, (img.naturalWidth - side) / 2, (img.naturalHeight - side) / 2, side, side, 0, 0, side, side);
  // Halve step by step so large phone photos are averaged, not aliased.
  while (canvas.width / 2 >= SIZE) {
    const half = document.createElement("canvas");
    half.width = half.height = Math.floor(canvas.width / 2);
    const hc = half.getContext("2d");
    hc.imageSmoothingQuality = "high";
    hc.drawImage(canvas, 0, 0, half.width, half.height);
    canvas = half;
  }
  const out = document.createElement("canvas");
  out.width = out.height = SIZE;
  const ctx = out.getContext("2d");
  ctx.imageSmoothingQuality = "high";
  if (flip) { ctx.translate(SIZE, 0); ctx.scale(-1, 1); }
  ctx.drawImage(canvas, 0, 0, SIZE, SIZE);
  const px = ctx.getImageData(0, 0, SIZE, SIZE).data;
  const data = new Float32Array(3 * SIZE * SIZE);
  for (let i = 0; i < SIZE * SIZE; i++) {
    for (let c = 0; c < 3; c++) data[c * SIZE * SIZE + i] = (px[i * 4 + c] / 255 - MEAN[c]) / STD[c];
  }
  return data;
}

async function classify(img) {
  state.pending = null;
  $("error").hidden = true;
  try {
    // The photo and its mirror image, averaged: a cheap way to make the answer steadier.
    const batch = new Float32Array(2 * 3 * SIZE * SIZE);
    batch.set(toTensor(img, false), 0);
    batch.set(toTensor(img, true), 3 * SIZE * SIZE);
    const out = await state.session.run({ image: new ort.Tensor("float32", batch, [2, 3, SIZE, SIZE]) });
    const raw = out.logits.data;
    const n = state.labels.length;
    state.logits = Array.from({ length: n }, (_, i) => (raw[i] + raw[n + i]) / 2);
    render();
  } catch (err) {
    console.error(err);
    showError("जाँच नहीं हो पाई। कोई दूसरी फोटो आज़माएँ। (Something went wrong.)");
  }
}

// ---------- result ----------
function softmax(xs) {
  const m = Math.max(...xs);
  const e = xs.map((x) => Math.exp(x - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((x) => x / s);
}

function render() {
  const allowed = state.labels.map((l, i) => ({ ...l, i })).filter((l) => !state.crop || l.crop === state.crop);
  const probs = softmax(allowed.map((l) => state.logits[l.i]));
  const ranked = allowed.map((l, k) => ({ ...l, p: probs[k] })).sort((a, b) => b.p - a.p);
  const top = ranked[0];
  const advice = window.ADVICE[top.folder];

  $("placeholder").hidden = true;
  $("result").hidden = false;
  $("r-badge").textContent = top.healthy ? "🌿" : "🩺";
  $("r-badge").classList.toggle("sick", !top.healthy);
  $("r-crop").textContent = `${CROP_EMOJI[top.crop]} ${top.crop_hi} · ${top.crop}`;
  $("r-name").textContent = top.disease_hi;
  $("r-en").textContent = top.disease;
  $("r-conf").textContent = `भरोसा (confidence): ${Math.round(top.p * 100)}%`;
  $("r-unsure").hidden = top.p >= UNSURE_BELOW;
  $("r-urgent").hidden = !(advice.urgent && top.p >= UNSURE_BELOW);
  $("r-urgent").textContent = "⏱️ यह रोग तेज़ी से फैलता है या इसका इलाज नहीं है। जल्दी कदम उठाएँ और विशेषज्ञ से सलाह लें।";
  $("r-top3").innerHTML = ranked
    .slice(0, 3)
    .map((l) => {
      const pct = Math.round(l.p * 100);
      return `<li><span class="t">${CROP_EMOJI[l.crop]} ${l.crop_hi}: ${l.disease_hi} <span style="color:var(--muted);font-weight:500">(${l.disease})</span></span><span class="p">${pct}%</span><span class="bar-track"><span class="bar-fill" style="width:${Math.max(pct, 1)}%"></span></span></li>`;
    })
    .join("");
  $("r-signs").textContent = advice.signs;
  $("r-steps").innerHTML = advice.steps.map((s) => `<li>${s}</li>`).join("");
}

function showError(msg) {
  $("error").textContent = msg;
  $("error").hidden = false;
}

init();
