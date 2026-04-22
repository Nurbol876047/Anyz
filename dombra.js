import {
  FilesetResolver,
  HandLandmarker
} from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.34/vision_bundle.mjs';

const WASM_ROOT = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.34/wasm';
const MODEL_ASSET_PATH = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';

const MOTION_TRIGGER = Object.freeze({
  seconds: 2.4,
  minDelta: 0.01,
  decayPerSecond: 1.7
});

const FINGER_CONNECTIONS = [
  [1, 2], [2, 3], [3, 4],
  [5, 6], [6, 7], [7, 8],
  [9, 10], [10, 11], [11, 12],
  [13, 14], [14, 15], [15, 16],
  [17, 18], [18, 19], [19, 20]
];
const FINGER_TIPS = new Set([4, 8, 12, 16, 20]);

const state = {
  cameraReady: false,
  cameraStatus: 'Камера мен MediaPipe жүктеліп жатыр...',
  cameraLabel: 'Қол тану дайындалуда...',
  videoReady: true,
  videoActive: false,
  cursorVisible: false,
  x: 0.5,
  y: 0.22,
  progress: 0,
  motionSeconds: 0,
  moving: false,
  lastTrackedAt: 0,
  lastTrackedX: null,
  lastTrackedY: null,
  statusTitle: 'Қимылды бастауға дайын',
  statusText: 'Камера алдында қолыңызды 2–3 секунд қимылдатыңыз. Сонда видео осы блоктың ішінде ашылады.'
};

const instrumentStage = document.getElementById('instrumentStage');
const stageStatusTitle = document.getElementById('stageStatusTitle');
const stageStatusText = document.getElementById('stageStatusText');
const guideFill = document.getElementById('guideFill');
const handCursor = document.getElementById('handCursor');
const stageVideo = document.getElementById('stageVideo');
const resetStageBtn = document.getElementById('resetStageBtn');

const cameraVideo = document.getElementById('cameraVideo');
const cameraCanvas = document.getElementById('cameraCanvas');
const cameraFallback = document.getElementById('cameraFallback');
const cameraOverlayLabel = document.getElementById('cameraOverlayLabel');
const cameraContext = cameraCanvas.getContext('2d');

const progressFill = document.getElementById('progressFill');
const progressLabel = document.getElementById('progressLabel');

let handLandmarker;
let lastVideoTime = -1;
let rafId = 0;
let stream;
let audioUnlocked = false;
let audioUnlockPromise = null;

function readAudioIntent() {
  try {
    return sessionStorage.getItem('dombraAudioIntent');
  } catch {
    return null;
  }
}

function clearAudioIntent() {
  try {
    sessionStorage.removeItem('dombraAudioIntent');
  } catch {
    // Ignore sessionStorage errors in restrictive browsers.
  }
}

function clamp(value, min = 0, max = 1) {
  return Math.min(max, Math.max(min, value));
}

function render() {
  stageStatusTitle.textContent = state.statusTitle;
  stageStatusText.textContent = state.statusText;

  guideFill.style.transform = `scaleY(${clamp(state.progress).toFixed(3)})`;
  progressFill.style.width = `${Math.round(clamp(state.progress) * 100)}%`;
  progressLabel.textContent = `${Math.round(clamp(state.progress) * 100)}%`;

  handCursor.style.left = `${(state.x * 100).toFixed(2)}%`;
  handCursor.style.top = `${(state.y * 100).toFixed(2)}%`;
  handCursor.classList.toggle('is-visible', state.cursorVisible && !state.videoActive);

  instrumentStage.classList.toggle('is-armed', state.moving || state.progress > 0.02);
  instrumentStage.classList.toggle('is-video-live', state.videoActive);

  cameraFallback.textContent = state.cameraStatus;
  cameraFallback.classList.toggle('is-hidden', state.cameraReady);
  cameraOverlayLabel.textContent = state.cameraLabel;
}

function resetInteraction(title, text) {
  state.moving = false;
  state.progress = 0;
  state.motionSeconds = 0;
  state.lastTrackedAt = 0;
  state.lastTrackedX = null;
  state.lastTrackedY = null;

  if (title) {
    state.statusTitle = title;
  }

  if (text) {
    state.statusText = text;
  }
}

function resetStageView() {
  stageVideo.pause();
  stageVideo.currentTime = 0;
  stageVideo.muted = false;
  stageVideo.volume = 1;
  state.videoActive = false;
  state.cursorVisible = false;
  state.cameraLabel = state.cameraReady
    ? 'Қолды қайтадан домбыраға бағыттаңыз'
    : 'Камера қосылып жатыр...';
  resetInteraction(
    'Қимылды бастауға дайын',
    'Камера алдында қолыңызды 2–3 секунд қимылдатыңыз. Сонда видео осы блоктың ішінде ашылады.'
  );
  render();
}

async function unlockVideoAudio() {
  if (audioUnlocked || !state.videoReady) {
    return audioUnlocked;
  }

  if (audioUnlockPromise) {
    return audioUnlockPromise;
  }

  audioUnlockPromise = (async () => {
    const previousTime = stageVideo.currentTime;
    stageVideo.muted = false;
    stageVideo.volume = 1;

    try {
      await stageVideo.play();
      stageVideo.pause();
      stageVideo.currentTime = previousTime || 0;
      audioUnlocked = true;
      clearAudioIntent();
      return true;
    } catch (error) {
      stageVideo.pause();
      stageVideo.currentTime = previousTime || 0;
      return false;
    } finally {
      audioUnlockPromise = null;
    }
  })();

  return audioUnlockPromise;
}

async function triggerVideo() {
  state.videoActive = true;
  state.moving = false;
  state.progress = 1;
  state.motionSeconds = MOTION_TRIGGER.seconds;
  state.cursorVisible = false;
  state.statusTitle = 'Домбыра жасырылып, видео ашылды';
  state.statusText = 'Камера қол қимылын жеткілікті уақыт көрді. Енді видео дәл осы блоктың ішінде ойнап тұр.';
  state.cameraLabel = 'Видео көрсетіліп тұр';
  render();

  try {
    await unlockVideoAudio();
    stageVideo.currentTime = 0;
    stageVideo.muted = false;
    stageVideo.volume = 1;
    await stageVideo.play();
  } catch (error) {
    try {
      stageVideo.muted = true;
      await stageVideo.play();
      state.statusTitle = 'Видео қосылды, бірақ дыбыс бұғатталды';
      state.statusText = 'Браузер дыбысты бірден ашпады. Бетті бір рет түртіп көрсеңіз, келесі қосылуда дыбыс бірге шығады.';
      render();
    } catch (fallbackError) {
      state.statusTitle = 'Видео автоматты түрде ашылмады';
      state.statusText = 'Плеердегі ойнату батырмасын басып көріңіз.';
      render();
    }
  }
}

function updateStrum(result, now) {
  if (!state.cameraReady || !state.videoReady || state.videoActive) {
    state.cursorVisible = false;
    return;
  }

  const hand = result.landmarks[0];
  if (!hand) {
    state.cursorVisible = false;
    state.cameraLabel = 'Қол көрінбей тұр';
    state.motionSeconds = Math.max(0, state.motionSeconds - 0.08);
    state.progress = clamp(state.motionSeconds / MOTION_TRIGGER.seconds);
    state.moving = false;
    state.lastTrackedAt = 0;
    state.lastTrackedX = null;
    state.lastTrackedY = null;
    state.statusTitle = 'Қолды көрсетіңіз';
    state.statusText = 'Алақаныңызды камера алдына апарып, оны 2–3 секунд қозғалтыңыз.';
    return;
  }

  const indexTip = hand[8];
  state.x = clamp(1 - indexTip.x);
  state.y = clamp(indexTip.y);
  state.cursorVisible = true;

  let movementDelta = 0;
  let deltaTime = 0;

  if (state.lastTrackedX !== null && state.lastTrackedY !== null && state.lastTrackedAt > 0) {
    movementDelta = Math.hypot(state.x - state.lastTrackedX, state.y - state.lastTrackedY);
    deltaTime = Math.min(0.12, Math.max(0, now - state.lastTrackedAt));
  }

  state.lastTrackedX = state.x;
  state.lastTrackedY = state.y;
  state.lastTrackedAt = now;

  if (movementDelta >= MOTION_TRIGGER.minDelta && deltaTime > 0) {
    state.motionSeconds = Math.min(MOTION_TRIGGER.seconds, state.motionSeconds + deltaTime);
    state.moving = true;
    state.cameraLabel = 'Қол қозғалысы табылды';
    state.statusTitle = 'Қозғалыс жиналып жатыр';
    state.statusText = 'Қимылды тағы аздап жалғастырыңыз. 2–3 секунд жеткенде видео ашылады.';
  } else {
    state.motionSeconds = Math.max(0, state.motionSeconds - (deltaTime || 0.04) * MOTION_TRIGGER.decayPerSecond);
    state.moving = false;
    state.cameraLabel = 'Қол көрінді, бірақ қозғалыс аз';

    if (state.motionSeconds > 0.05) {
      state.statusTitle = 'Қимыл баяулап қалды';
      state.statusText = 'Қолды тағы біраз қозғалтыңыз, сонда прогресс толық бітеді.';
    } else {
      state.statusTitle = 'Қимылды бастауға дайын';
      state.statusText = 'Камера алдында қолыңызды 2–3 секунд қимылдатыңыз. Сонда видео осы блоктың ішінде ашылады.';
    }
  }

  state.progress = clamp(state.motionSeconds / MOTION_TRIGGER.seconds);

  if (state.motionSeconds >= MOTION_TRIGGER.seconds) {
    triggerVideo();
  }
}

function resizeCameraCanvas() {
  const width = cameraVideo.videoWidth || 960;
  const height = cameraVideo.videoHeight || 540;
  if (cameraCanvas.width !== width || cameraCanvas.height !== height) {
    cameraCanvas.width = width;
    cameraCanvas.height = height;
  }
}

function clearCanvas() {
  cameraContext.clearRect(0, 0, cameraCanvas.width, cameraCanvas.height);
}

function drawResults(result) {
  resizeCameraCanvas();
  clearCanvas();

  const gradient = cameraContext.createLinearGradient(0, 0, 0, cameraCanvas.height);
  gradient.addColorStop(0, '#030708');
  gradient.addColorStop(1, '#0a1416');
  cameraContext.fillStyle = gradient;
  cameraContext.fillRect(0, 0, cameraCanvas.width, cameraCanvas.height);

  const glow = cameraContext.createRadialGradient(
    cameraCanvas.width * 0.5,
    cameraCanvas.height * 0.45,
    8,
    cameraCanvas.width * 0.5,
    cameraCanvas.height * 0.45,
    cameraCanvas.width * 0.36
  );
  glow.addColorStop(0, 'rgba(112, 255, 152, 0.14)');
  glow.addColorStop(1, 'rgba(112, 255, 152, 0)');
  cameraContext.fillStyle = glow;
  cameraContext.fillRect(0, 0, cameraCanvas.width, cameraCanvas.height);

  if (!result.landmarks.length) return;

  cameraContext.save();
  cameraContext.lineCap = 'round';
  cameraContext.lineJoin = 'round';
  cameraContext.lineWidth = Math.max(3, cameraCanvas.width / 260);
  cameraContext.strokeStyle = 'rgba(112, 255, 152, 0.92)';
  cameraContext.shadowBlur = 18;
  cameraContext.shadowColor = 'rgba(112, 255, 152, 0.28)';

  result.landmarks.forEach((handLandmarks) => {
    FINGER_CONNECTIONS.forEach(([startIndex, endIndex]) => {
      const start = handLandmarks[startIndex];
      const end = handLandmarks[endIndex];
      cameraContext.beginPath();
      cameraContext.moveTo(start.x * cameraCanvas.width, start.y * cameraCanvas.height);
      cameraContext.lineTo(end.x * cameraCanvas.width, end.y * cameraCanvas.height);
      cameraContext.stroke();
    });

    handLandmarks.forEach((landmark, index) => {
      if (index === 0) {
        return;
      }

      cameraContext.beginPath();
      cameraContext.fillStyle = FINGER_TIPS.has(index)
        ? 'rgba(230, 255, 215, 0.98)'
        : 'rgba(112, 255, 152, 0.94)';
      cameraContext.arc(
        landmark.x * cameraCanvas.width,
        landmark.y * cameraCanvas.height,
        FINGER_TIPS.has(index)
          ? Math.max(6, cameraCanvas.width / 110)
          : Math.max(4, cameraCanvas.width / 150),
        0,
        Math.PI * 2
      );
      cameraContext.fill();
    });
  });

  cameraContext.restore();
}

function processFrame() {
  if (!handLandmarker || cameraVideo.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
    rafId = requestAnimationFrame(processFrame);
    return;
  }

  if (cameraVideo.currentTime !== lastVideoTime) {
    const now = performance.now();
    lastVideoTime = cameraVideo.currentTime;
    const result = handLandmarker.detectForVideo(cameraVideo, now);
    drawResults(result);
    updateStrum(result, now / 1000);
    render();
  }

  rafId = requestAnimationFrame(processFrame);
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error('Бұл браузер камера API-сын қолдамайды.');
  }

  stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: {
      facingMode: 'user',
      width: { ideal: 960 },
      height: { ideal: 540 }
    }
  });

  cameraVideo.srcObject = stream;

  await new Promise((resolve) => {
    cameraVideo.onloadedmetadata = () => resolve();
  });

  await cameraVideo.play();
  resizeCameraCanvas();
  state.cameraReady = true;
  state.cameraStatus = 'Камера дайын';
  state.cameraLabel = 'Қолды домбыраның жоғары жағына апарыңыз';
  await unlockVideoAudio();
  render();
}

async function createHandLandmarker() {
  const vision = await FilesetResolver.forVisionTasks(WASM_ROOT);
  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: MODEL_ASSET_PATH
    },
    runningMode: 'VIDEO',
    numHands: 1,
    minHandDetectionConfidence: 0.65,
    minHandPresenceConfidence: 0.6,
    minTrackingConfidence: 0.6
  });
}

function attachListeners() {
  resetStageBtn.addEventListener('click', resetStageView);

  const tryUnlockAudio = () => {
    unlockVideoAudio();
  };

  window.addEventListener('pointerdown', tryUnlockAudio, { passive: true });
  window.addEventListener('keydown', tryUnlockAudio);

  if (readAudioIntent()) {
    unlockVideoAudio();
  }

  stageVideo.addEventListener('playing', () => {
    stageVideo.muted = false;
    stageVideo.volume = 1;
    state.statusTitle = 'Видео ойнап тұр';
    state.statusText = 'Домбыра орнын видео алды. Қайта көру үшін домбыраны қайта көрсету батырмасын қолдануға болады.';
    render();
  });

  stageVideo.addEventListener('volumechange', () => {
    if (!stageVideo.muted && stageVideo.volume > 0) {
      audioUnlocked = true;
      clearAudioIntent();
    }
  });

  stageVideo.addEventListener('ended', () => {
    state.statusTitle = 'Видео аяқталды';
    state.statusText = 'Домбыраны қайта көрсету батырмасымен бастапқы күйге оралыңыз.';
    render();
  });

  stageVideo.addEventListener('error', () => {
    state.videoReady = false;
    state.videoActive = false;
    state.statusTitle = 'Видео файлы табылмады';
    state.statusText = 'media/dombra_story.mp4 файлын тексеріңіз.';
    render();
  });
}

async function init() {
  attachListeners();
  render();

  try {
    await Promise.all([createHandLandmarker(), startCamera()]);
    processFrame();
  } catch (error) {
    state.cameraReady = false;
    state.cameraStatus = `Камера не MediaPipe іске қосылмады: ${error.message}`;
    state.cameraLabel = 'Қол тану қолжетімсіз';
    state.statusTitle = 'Камера іске қосылмады';
    state.statusText = error.message;
    render();
  }
}

window.addEventListener('beforeunload', () => {
  if (rafId) {
    cancelAnimationFrame(rafId);
  }

  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
  }

  handLandmarker?.close();
});

init();
