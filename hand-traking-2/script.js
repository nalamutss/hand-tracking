import {
  HandLandmarker,
  FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

const video = document.getElementById("webcam");
const overlay = document.getElementById("overlay");
const ctx = overlay.getContext("2d");
const startOverlay = document.getElementById("startOverlay");
const startBtn = document.getElementById("startBtn");

const statusText = document.getElementById("statusText");
const statusBlip = document.getElementById("statusBlip");
const frameState = document.getElementById("frameState");
const frameStatus = document.getElementById("frameStatus");
const handCount = document.getElementById("handCount");
const confidenceValue = document.getElementById("confidenceValue");
const modeNameEl = document.getElementById("modeName");
const modeHintEl = document.getElementById("modeHint");

overlay.style.transform = "scaleX(-1)";

let handLandmarker = null;
let running = false;
let lastVideoTime = -1;

const tinyCanvas = document.createElement("canvas");
const tinyCtx = tinyCanvas.getContext("2d");

const INDEX_TIP = 8;
const WRIST = 0;
const TIPS = [8, 12, 16, 20];
const PIPS = [6, 10, 14, 18];

const MODES = {
  1: { name: "SCAN",     color: "#5de2ff", shape: "rect",     effect: "scan" },
  2: { name: "HALFTONE", color: "#ff5de2", shape: "triangle", effect: "halftone" },
  3: { name: "PIXEL",    color: "#3ddc97", shape: "circle",   effect: "pixel" },
  4: { name: "THERMAL",  color: "#ffb84d", shape: "diamond",  effect: "thermal" }
};

let currentMode = MODES[1];
let gestureBuffer = [];
const GESTURE_HOLD_FRAMES = 10; 

async function setup() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );

  const baseModelOptions = {
    modelAssetPath:
      "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
  };

  try {
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: { ...baseModelOptions, delegate: "GPU" },
      runningMode: "VIDEO",
      numHands: 2,
      minHandDetectionConfidence: 0.5,
      minHandPresenceConfidence: 0.5,
      minTrackingConfidence: 0.5
    });
  } catch (err) {
    console.warn("GPU delegate failed, falling back to CPU:", err);
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: { ...baseModelOptions, delegate: "CPU" },
      runningMode: "VIDEO",
      numHands: 2,
      minHandDetectionConfidence: 0.5,
      minHandPresenceConfidence: 0.5,
      minTrackingConfidence: 0.5
    });
  }
}

async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
      audio: false
    });
    video.srcObject = stream;

    await new Promise((resolve) => {
      video.onloadedmetadata = () => resolve();
    });

    video.play();
    resizeCanvas();
    window.addEventListener("resize", resizeCanvas);

    startOverlay.style.display = "none";
    statusBlip.classList.add("active");
    setStatus("MENCARI TANGAN...");
    applyModeToHud();

    running = true;
    requestAnimationFrame(renderLoop);
  } catch (err) {
    setStatus("AKSES KAMERA DITOLAK ATAU TIDAK TERSEDIA.");
    console.error(err);
  }
}

function resizeCanvas() {
  const rect = video.getBoundingClientRect();
  overlay.width = rect.width;
  overlay.height = rect.height;
}

function setStatus(msg) {
  statusText.textContent = msg;
}

function applyModeToHud() {
  modeNameEl.textContent = currentMode.name;
  modeNameEl.style.color = currentMode.color;
}
function renderLoop() {
  if (!running) return;

  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const results = handLandmarker.detectForVideo(video, performance.now());
    draw(results);
  }

  requestAnimationFrame(renderLoop);
}

function draw(results) {
  const w = overlay.width;
  const h = overlay.height;
  ctx.clearRect(0, 0, w, h);

  const hands = results.landmarks || [];
  handCount.textContent = `${hands.length}/2`;

  if (results.handednesses && results.handednesses.length) {
    const avgConf =
      results.handednesses.reduce((sum, h) => sum + (h[0]?.score || 0), 0) /
      results.handednesses.length;
    confidenceValue.textContent = `${Math.round(avgConf * 100)}%`;
  } else {
    confidenceValue.textContent = "--";
  }

  if (hands.length === 0) {
    gestureBuffer = [];
    frameState.textContent = "STANDBY";
    frameStatus.textContent = "INACTIVE";
    modeHintEl.textContent = "";
    setStatus("TIDAK ADA TANGAN TERDETEKSI.");
    return;
  }

  if (hands.length === 1) {
    frameState.textContent = "STANDBY";
    frameStatus.textContent = "INACTIVE";
    drawFingerMarker(toPoint(hands[0][INDEX_TIP], w, h), currentMode.color);
    handleGesture(hands[0], w, h);
    return;
  }

  gestureBuffer = [];
  modeHintEl.textContent = "";
  const p1 = toPoint(hands[0][INDEX_TIP], w, h);
  const p2 = toPoint(hands[1][INDEX_TIP], w, h);

  frameState.textContent = "ACTIVE";
  frameStatus.textContent = "ACTIVE";
  setStatus(`CYBER FRAME AKTIF — MODE ${currentMode.name}.`);

  drawFingerMarker(p1, currentMode.color);
  drawFingerMarker(p2, currentMode.color);
  drawCyberFrame(p1, p2);
}

function handleGesture(landmarks, w, h) {
  const count = countExtendedFingers(landmarks);

  if (count >= 1 && count <= 4) {
    gestureBuffer.push(count);
    if (gestureBuffer.length > GESTURE_HOLD_FRAMES) gestureBuffer.shift();
  } else {
    gestureBuffer = [];
  }

  const held =
    gestureBuffer.length === GESTURE_HOLD_FRAMES &&
    gestureBuffer.every((v) => v === gestureBuffer[0]);

  const targetMode = MODES[count];
  const progress = Math.min(gestureBuffer.length / GESTURE_HOLD_FRAMES, 1);

  if (targetMode && targetMode !== currentMode) {
    setStatus(
      `ANGKAT ${count} JARI → MODE ${targetMode.name} (${Math.round(progress * 100)}%)`
    );
    modeHintEl.textContent = `${count} JARI: ${targetMode.name}`;
    drawHoldRing(toPoint(landmarks[WRIST], w, h), progress, targetMode.color);
  } else if (targetMode) {
    setStatus(`MODE ${targetMode.name} AKTIF. TUNJUKKAN 2 TANGAN UNTUK BUKA FRAME.`);
    modeHintEl.textContent = `MODE: ${targetMode.name}`;
  } else {
    setStatus("ANGKAT 1-4 JARI (SATU TANGAN) UNTUK GANTI MODE.");
    modeHintEl.textContent = "";
  }

  if (held && targetMode && targetMode !== currentMode) {
    currentMode = targetMode;
    applyModeToHud();
    gestureBuffer = [];
  }
}

function countExtendedFingers(landmarks) {
  let count = 0;
  for (let i = 0; i < TIPS.length; i++) {
    const tip = landmarks[TIPS[i]];
    const pip = landmarks[PIPS[i]];
    if (tip.y < pip.y - 0.02) count++;
  }
  return count;
}

function toPoint(landmark, w, h) {
  return { x: landmark.x * w, y: landmark.y * h };
}

function getEffectRects(cx, cy, boxW, boxH) {
  const sx = Math.max(0, cx - boxW);
  const sy = Math.max(0, cy - boxH);
  const ex = Math.min(overlay.width, cx + boxW);
  const ey = Math.min(overlay.height, cy + boxH);
  const sw = Math.max(1, ex - sx);
  const sh = Math.max(1, ey - sy);

  const scaleX = (video.videoWidth || overlay.width) / overlay.width;
  const scaleY = (video.videoHeight || overlay.height) / overlay.height;

  return {
    dest: [sx, sy, sw, sh],
    src: [sx * scaleX, sy * scaleY, sw * scaleX, sh * scaleY]
  };
}

function drawFingerMarker(p, color) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(p.x, p.y, 6, 0, Math.PI * 2);
  ctx.stroke();
  ctx.restore();
}

function drawHoldRing(center, progress, color) {
  ctx.save();
  ctx.strokeStyle = "rgba(255,255,255,0.2)";
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.arc(center.x, center.y, 22, 0, Math.PI * 2);
  ctx.stroke();

  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.arc(center.x, center.y, 22, -Math.PI / 2, -Math.PI / 2 + progress * Math.PI * 2);
  ctx.stroke();
  ctx.restore();
}

function drawCyberFrame(p1, p2) {
  const cx = (p1.x + p2.x) / 2;
  const cy = (p1.y + p2.y) / 2;
  const dx = p2.x - p1.x;
  const dy = p2.y - p1.y;
  const dist = Math.hypot(dx, dy);
  const angle = Math.atan2(dy, dx);

  const boxW = dist;
  const boxH = dist * 0.6;
  const halfW = boxW / 2;
  const halfH = boxH / 2;
  const color = currentMode.color;
  const shape = currentMode.shape;

  ctx.save();
  ctx.translate(cx, cy);
  ctx.rotate(angle);

  ctx.save();
  buildShapePath(shape, halfW, halfH);
  ctx.clip();
  ctx.rotate(-angle);
  ctx.translate(-cx, -cy);
  drawEffect(currentMode.effect, cx, cy, boxW, boxH, color);
  ctx.restore();

  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.setLineDash([6, 5]);
  buildShapePath(shape, halfW, halfH);
  ctx.stroke();
  ctx.setLineDash([]);

  drawVertexTicks(shape, halfW, halfH, color);

  ctx.restore();

  ctx.save();
  ctx.fillStyle = color;
  ctx.font = "10px monospace";
  ctx.textAlign = "center";
  const labelY = cy - Math.abs(Math.sin(angle)) * halfW - halfH - 10;
  ctx.fillText(`CYBER FRAME // ${currentMode.name}`, cx, labelY);
  ctx.restore();
}

function buildShapePath(shape, halfW, halfH) {
  ctx.beginPath();
  if (shape === "rect") {
    ctx.rect(-halfW, -halfH, halfW * 2, halfH * 2);
  } else if (shape === "triangle") {
    ctx.moveTo(0, -halfH);
    ctx.lineTo(halfW, halfH);
    ctx.lineTo(-halfW, halfH);
    ctx.closePath();
  } else if (shape === "circle") {
    ctx.ellipse(0, 0, halfW, halfH, 0, 0, Math.PI * 2);
  } else if (shape === "diamond") {
    ctx.moveTo(0, -halfH);
    ctx.lineTo(halfW, 0);
    ctx.lineTo(0, halfH);
    ctx.lineTo(-halfW, 0);
    ctx.closePath();
  }
}

function drawVertexTicks(shape, halfW, halfH, color) {
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  const tick = 9;
  let verts = [];

  if (shape === "rect") {
    verts = [
      [-halfW, -halfH, 1, 1],
      [halfW, -halfH, -1, 1],
      [-halfW, halfH, 1, -1],
      [halfW, halfH, -1, -1]
    ];
    verts.forEach(([x, y, sx, sy]) => {
      ctx.beginPath();
      ctx.moveTo(x, y + tick * sy);
      ctx.lineTo(x, y);
      ctx.lineTo(x + tick * sx, y);
      ctx.stroke();
    });
    return;
  }

  if (shape === "triangle") verts = [[0, -halfH], [halfW, halfH], [-halfW, halfH]];
  else if (shape === "diamond") verts = [[0, -halfH], [halfW, 0], [0, halfH], [-halfW, 0]];
  else if (shape === "circle")
    verts = [[0, -halfH], [halfW, 0], [0, halfH], [-halfW, 0]];

  verts.forEach(([x, y]) => {
    ctx.beginPath();
    ctx.arc(x, y, 3, 0, Math.PI * 2);
    ctx.stroke();
  });
}

function drawEffect(effect, cx, cy, boxW, boxH, color) {
  if (effect === "scan") return effectScan(cx, cy, boxW, boxH, color);
  if (effect === "halftone") return effectHalftone(cx, cy, boxW, boxH, color);
  if (effect === "pixel") return effectPixel(cx, cy, boxW, boxH, color);
  if (effect === "thermal") return effectThermal(cx, cy, boxW, boxH, color);
}

function effectScan(cx, cy, boxW, boxH, color) {
  const t = performance.now() / 1000;
  const jitter = Math.sin(t * 14) * 2;
  const { dest, src } = getEffectRects(cx, cy, boxW, boxH);
  const [dx, dy, dw, dh] = dest;
  const [sx, sy, sw, sh] = src;

  ctx.globalCompositeOperation = "lighter";
  ctx.globalAlpha = 0.8;
  ctx.filter = "grayscale(1) brightness(1.4) contrast(1.2)";
  ctx.drawImage(video, sx, sy, sw, sh, dx + jitter, dy, dw, dh);

  ctx.globalAlpha = 0.35;
  ctx.filter = "none";
  ctx.drawImage(video, sx, sy, sw, sh, dx - jitter, dy, dw, dh);

  ctx.globalCompositeOperation = "source-over";
  ctx.globalAlpha = 1;
  ctx.filter = "none";

  drawScanlines(cx, cy, boxW, boxH, "rgba(93, 226, 255, 0.25)");
}

function effectHalftone(cx, cy, boxW, boxH, color) {
  const { dest, src } = getEffectRects(cx, cy, boxW, boxH);
  const [dx, dy, dw, dh] = dest;
  const [sx, sy, sw, sh] = src;
  const cellsX = 36;
  const cellsY = Math.max(1, Math.round(cellsX * (dh / dw)));

  tinyCanvas.width = cellsX;
  tinyCanvas.height = cellsY;
  tinyCtx.drawImage(video, sx, sy, sw, sh, 0, 0, cellsX, cellsY);

  ctx.imageSmoothingEnabled = false;
  ctx.filter = "grayscale(1) contrast(1.3)";
  ctx.drawImage(tinyCanvas, dx, dy, dw, dh);
  ctx.filter = "none";
  ctx.imageSmoothingEnabled = true;

  ctx.globalCompositeOperation = "multiply";
  ctx.fillStyle = color;
  ctx.globalAlpha = 0.55;
  ctx.fillRect(dx, dy, dw, dh);
  ctx.globalCompositeOperation = "source-over";
  ctx.globalAlpha = 1;
}

function effectPixel(cx, cy, boxW, boxH, color) {
  const { dest, src } = getEffectRects(cx, cy, boxW, boxH);
  const [dx, dy, dw, dh] = dest;
  const [sx, sy, sw, sh] = src;
  const cellsX = 22;
  const cellsY = Math.max(1, Math.round(cellsX * (dh / dw)));

  tinyCanvas.width = cellsX;
  tinyCanvas.height = cellsY;
  tinyCtx.drawImage(video, sx, sy, sw, sh, 0, 0, cellsX, cellsY);

  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(tinyCanvas, dx, dy, dw, dh);
  ctx.imageSmoothingEnabled = true;

  ctx.globalCompositeOperation = "color";
  ctx.fillStyle = color;
  ctx.globalAlpha = 0.5;
  ctx.fillRect(dx, dy, dw, dh);
  ctx.globalCompositeOperation = "source-over";
  ctx.globalAlpha = 1;
}

function effectThermal(cx, cy, boxW, boxH, color) {
  const { dest, src } = getEffectRects(cx, cy, boxW, boxH);
  const [dx, dy, dw, dh] = dest;
  const [sx, sy, sw, sh] = src;

  ctx.filter = "grayscale(1) invert(1) sepia(1) saturate(4) hue-rotate(-10deg) contrast(1.2)";
  ctx.drawImage(video, sx, sy, sw, sh, dx, dy, dw, dh);
  ctx.filter = "none";

  drawScanlines(cx, cy, boxW, boxH, "rgba(255, 184, 77, 0.2)");
}

function drawScanlines(cx, cy, boxW, boxH, strokeStyle) {
  const sx = cx - boxW;
  const sy = cy - boxH;
  const sw = boxW * 2;
  const sh = boxH * 2;
  ctx.strokeStyle = strokeStyle;
  ctx.lineWidth = 1;
  for (let y = sy; y < sy + sh; y += 4) {
    ctx.beginPath();
    ctx.moveTo(sx, y);
    ctx.lineTo(sx + sw, y);
    ctx.stroke();
  }
}

startBtn.addEventListener("click", async () => {
  startBtn.disabled = true;
  startBtn.textContent = "MEMUAT MODEL...";
  await setup();
  await startCamera();
});