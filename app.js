import { FaceLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/vision_bundle.mjs";

const MODEL_PATH = "./face_landmarker.task";

const RIGHT_EYE = [33, 160, 158, 133, 153, 144];
const LEFT_EYE = [362, 385, 387, 263, 373, 380];
const MOUTH_VERTICAL = [[81, 178], [13, 14], [311, 402]];
const MOUTH_HORIZONTAL = [78, 308];
const FACE_OVAL = [
  10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
  397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
  172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10,
];

const EAR_THRESHOLD = 0.22;
const MAR_THRESHOLD = 0.75;
const HEAD_TILT_THRESHOLD = 18;
const CONSEC_FRAMES_WARNING = 10;
const CONSEC_FRAMES_DROWSY = 20;
const YAWN_CONSEC_FRAMES = 15;

const els = {
  video: document.querySelector("#video"),
  canvas: document.querySelector("#overlay"),
  videoStage: document.querySelector("#videoStage"),
  stageStatus: document.querySelector("#stageStatus"),
  statusCard: document.querySelector("#statusCard"),
  statusLabel: document.querySelector("#statusLabel"),
  statusDetail: document.querySelector("#statusDetail"),
  earMetric: document.querySelector("#earMetric"),
  marMetric: document.querySelector("#marMetric"),
  tiltMetric: document.querySelector("#tiltMetric"),
  fpsMetric: document.querySelector("#fpsMetric"),
  eyeSignal: document.querySelector("#eyeSignal"),
  mouthSignal: document.querySelector("#mouthSignal"),
  headSignal: document.querySelector("#headSignal"),
  startButton: document.querySelector("#startButton"),
  stopButton: document.querySelector("#stopButton"),
};

const ctx = els.canvas.getContext("2d");
let faceLandmarker;
let stream;
let animationId;
let lastVideoTime = -1;
let lastFrameTime = performance.now();
let smoothedFps = 0;
let eyeClosedCounter = 0;
let yawnCounter = 0;
let headTiltCounter = 0;

els.startButton.addEventListener("click", startCamera);
els.stopButton.addEventListener("click", stopCamera);

async function startCamera() {
  setLoading("Loading model");
  els.startButton.disabled = true;

  try {
    if (!faceLandmarker) {
      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm",
      );
      faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: MODEL_PATH },
        runningMode: "VIDEO",
        numFaces: 1,
        minFaceDetectionConfidence: 0.5,
        minFacePresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });
    }

    setLoading("Opening camera");
    stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: "user",
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
    });

    els.video.srcObject = stream;
    await els.video.play();
    resetCounters();
    els.stopButton.disabled = false;
    els.stageStatus.classList.add("hidden");
    renderLoop();
  } catch (error) {
    console.error(error);
    setIdle("Camera unavailable", "Allow camera access and reload the page if the browser blocked the request.");
    els.startButton.disabled = false;
    els.stopButton.disabled = true;
  }
}

function stopCamera() {
  if (animationId) {
    cancelAnimationFrame(animationId);
    animationId = null;
  }

  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    stream = null;
  }

  ctx.clearRect(0, 0, els.canvas.width, els.canvas.height);
  els.video.srcObject = null;
  resetCounters();
  setIdle("Camera idle", "Start the camera to run detection locally in your browser.");
  els.stageStatus.textContent = "Camera idle";
  els.stageStatus.classList.remove("hidden");
  els.startButton.disabled = false;
  els.stopButton.disabled = true;
}

function renderLoop() {
  resizeCanvas();

  if (els.video.currentTime !== lastVideoTime) {
    const now = performance.now();
    const delta = now - lastFrameTime;
    if (delta > 0) {
      const fps = 1000 / delta;
      smoothedFps = smoothedFps ? smoothedFps * 0.9 + fps * 0.1 : fps;
    }
    lastFrameTime = now;

    const result = faceLandmarker.detectForVideo(els.video, now);
    processResult(result);
    lastVideoTime = els.video.currentTime;
  }

  animationId = requestAnimationFrame(renderLoop);
}

function resizeCanvas() {
  const rect = els.videoStage.getBoundingClientRect();
  const nextWidth = Math.max(1, Math.round(rect.width));
  const nextHeight = Math.max(1, Math.round(rect.height));

  if (els.canvas.width !== nextWidth || els.canvas.height !== nextHeight) {
    els.canvas.width = nextWidth;
    els.canvas.height = nextHeight;
  }
}

function processResult(result) {
  ctx.clearRect(0, 0, els.canvas.width, els.canvas.height);

  const face = result.faceLandmarks?.[0];
  if (!face) {
    resetCounters();
    updateUi({
      faceDetected: false,
      ear: 0,
      mar: 0,
      tilt: 0,
      eyesClosed: false,
      yawning: false,
      headDrooping: false,
      alertLevel: 1,
      drowsy: false,
    });
    return;
  }

  const points = face.map((lm) => ({
    x: lm.x * els.canvas.width,
    y: lm.y * els.canvas.height,
    z: lm.z,
  }));

  const ear = (computeEar(points, RIGHT_EYE) + computeEar(points, LEFT_EYE)) / 2;
  const mar = computeMar(points);
  const tilt = computeHeadTilt(points);

  const eyesClosed = ear < EAR_THRESHOLD;
  const yawning = mar > MAR_THRESHOLD;
  const headDrooping = Math.abs(tilt) > HEAD_TILT_THRESHOLD;

  eyeClosedCounter = eyesClosed ? eyeClosedCounter + 1 : 0;
  yawnCounter = yawning ? yawnCounter + 1 : 0;
  headTiltCounter = headDrooping ? headTiltCounter + 1 : 0;

  let score = 0;
  if (eyeClosedCounter >= CONSEC_FRAMES_DROWSY) score += 2;
  else if (eyeClosedCounter >= CONSEC_FRAMES_WARNING) score += 1;
  if (yawnCounter >= YAWN_CONSEC_FRAMES) score += 1;
  if (headTiltCounter >= CONSEC_FRAMES_WARNING) score += 1;

  const drowsy = score >= 2 || eyeClosedCounter >= CONSEC_FRAMES_DROWSY;
  const alertLevel = drowsy ? 2 : score >= 1 ? 1 : 0;

  drawFace(points, { eyesClosed, yawning, alertLevel });
  updateUi({
    faceDetected: true,
    ear,
    mar,
    tilt,
    eyesClosed,
    yawning,
    headDrooping,
    alertLevel,
    drowsy,
  });
}

function computeEar(points, indices) {
  const p = indices.map((index) => points[index]);
  const vertical1 = dist(p[1], p[5]);
  const vertical2 = dist(p[2], p[4]);
  const horizontal = dist(p[0], p[3]);
  return horizontal === 0 ? 0 : (vertical1 + vertical2) / (2 * horizontal);
}

function computeMar(points) {
  const vertical = MOUTH_VERTICAL
    .map(([top, bottom]) => dist(points[top], points[bottom]))
    .reduce((sum, value) => sum + value, 0) / MOUTH_VERTICAL.length;
  const horizontal = dist(points[MOUTH_HORIZONTAL[0]], points[MOUTH_HORIZONTAL[1]]);
  return horizontal === 0 ? 0 : vertical / horizontal;
}

function computeHeadTilt(points) {
  const leftEye = points[LEFT_EYE[3]];
  const rightEye = points[RIGHT_EYE[3]];
  return Math.atan2(leftEye.y - rightEye.y, leftEye.x - rightEye.x) * (180 / Math.PI);
}

function drawFace(points, state) {
  ctx.lineJoin = "round";
  ctx.lineCap = "round";

  drawPolyline(FACE_OVAL.map((index) => points[index]), "#6dd7a2", 1.4);
  drawPolyline(RIGHT_EYE.map((index) => points[index]), state.eyesClosed ? "#ef476f" : "#49c78e", 3, true);
  drawPolyline(LEFT_EYE.map((index) => points[index]), state.eyesClosed ? "#ef476f" : "#49c78e", 3, true);

  const mouthPoints = [
    points[78],
    points[81],
    points[13],
    points[311],
    points[308],
    points[402],
    points[14],
    points[178],
  ];
  drawPolyline(mouthPoints, state.yawning ? "#ffd166" : "#49c78e", 3, true);

  if (state.alertLevel === 2) {
    ctx.save();
    ctx.transform(-1, 0, 0, 1, els.canvas.width, 0);
    ctx.fillStyle = "rgba(239, 71, 111, 0.92)";
    const boxWidth = Math.min(520, els.canvas.width - 40);
    const boxHeight = 70;
    const x = (els.canvas.width - boxWidth) / 2;
    const y = els.canvas.height - boxHeight - 34;
    ctx.fillRect(x, y, boxWidth, boxHeight);
    ctx.fillStyle = "#fff";
    ctx.font = "800 30px system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("DROWSY - WAKE UP", els.canvas.width / 2, y + boxHeight / 2);
    ctx.restore();
  }
}

function drawPolyline(points, color, width, closePath = false) {
  if (points.length < 2) return;

  ctx.beginPath();
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.moveTo(points[0].x, points[0].y);
  points.slice(1).forEach((point) => ctx.lineTo(point.x, point.y));
  if (closePath) ctx.closePath();
  ctx.stroke();
}

function updateUi(state) {
  els.earMetric.textContent = state.ear.toFixed(2);
  els.marMetric.textContent = state.mar.toFixed(2);
  els.tiltMetric.textContent = `${Math.round(state.tilt)} deg`;
  els.fpsMetric.textContent = `${Math.round(smoothedFps)}`;

  els.eyeSignal.textContent = state.eyesClosed ? "Eyes closed" : "Eyes open";
  els.mouthSignal.textContent = state.yawning ? "Yawn detected" : "No yawn";
  els.headSignal.textContent = state.headDrooping ? "Head tilted" : "Head stable";
  setSignalState(els.eyeSignal, state.eyesClosed, state.drowsy && state.eyesClosed);
  setSignalState(els.mouthSignal, state.yawning, false);
  setSignalState(els.headSignal, state.headDrooping, false);

  clearLevelClasses();
  if (!state.faceDetected) {
    els.videoStage.classList.add("warning");
    els.statusCard.classList.add("warning");
    els.statusLabel.textContent = "No face detected";
    els.statusDetail.textContent = "Position your face in the frame.";
    return;
  }

  if (state.alertLevel === 2) {
    els.videoStage.classList.add("danger");
    els.statusCard.classList.add("danger");
    els.statusLabel.textContent = "Drowsy";
    els.statusDetail.textContent = "Prolonged eye closure or multiple fatigue signals detected.";
  } else if (state.alertLevel === 1) {
    els.videoStage.classList.add("warning");
    els.statusCard.classList.add("warning");
    els.statusLabel.textContent = "Warning";
    els.statusDetail.textContent = "A fatigue indicator is building across consecutive frames.";
  } else {
    els.statusLabel.textContent = "Awake";
    els.statusDetail.textContent = "Face detected and fatigue indicators are below thresholds.";
  }
}

function setSignalState(element, active, danger) {
  element.classList.toggle("active", active);
  element.classList.toggle("danger", danger);
}

function setLoading(message) {
  els.stageStatus.textContent = message;
  els.stageStatus.classList.remove("hidden");
  els.statusLabel.textContent = message;
  els.statusDetail.textContent = "Preparing the live detector.";
}

function setIdle(label, detail) {
  clearLevelClasses();
  els.statusLabel.textContent = label;
  els.statusDetail.textContent = detail;
  els.earMetric.textContent = "0.00";
  els.marMetric.textContent = "0.00";
  els.tiltMetric.textContent = "0 deg";
  els.fpsMetric.textContent = "0";
  els.eyeSignal.textContent = "Eyes open";
  els.mouthSignal.textContent = "No yawn";
  els.headSignal.textContent = "Head stable";
  setSignalState(els.eyeSignal, false, false);
  setSignalState(els.mouthSignal, false, false);
  setSignalState(els.headSignal, false, false);
}

function clearLevelClasses() {
  els.videoStage.classList.remove("warning", "danger");
  els.statusCard.classList.remove("warning", "danger");
}

function resetCounters() {
  lastVideoTime = -1;
  smoothedFps = 0;
  eyeClosedCounter = 0;
  yawnCounter = 0;
  headTiltCounter = 0;
}

function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}
