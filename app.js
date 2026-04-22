import {
  FilesetResolver,
  HandLandmarker
} from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.34/vision_bundle.mjs';

const WASM_ROOT = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.34/wasm';
const MODEL_ASSET_PATH = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';

const VALID_GESTURES = new Set([1, 2]);
const HOLD_SECONDS = 0.85;
const REARM_SECONDS = 0.65;
const MILK_DELAY_SECONDS = 3.0;

const SCENES = {
  idle: {
    label: 'Күту режімі',
    text: 'Камера алдында 1 не 2 саусақ көрсетіңіз. Бір саусақ басқа әуенді, екі саусақ дұрыс күйді іске қосады.'
  },
  one: {
    label: '1 саусақ — бірінші ән',
    text: 'Бірінші әуен ойнап тұр. Естемес алға шықты, бірақ бұл кезде нар сүт бермейді.'
  },
  two_wait: {
    label: '2 саусақ — дұрыс күй',
    text: 'Дұрыс күй танылды. Жаңыл нар жанына келді. 3 секундтан кейін ғана сүт шығуы тиіс.'
  },
  milk: {
    label: 'Сүттің ағуы',
    text: 'Дұрыс күй жүрекке жетті. Жаңыл доып отыр, нар иіп, сүт бере бастады.'
  }
};

const state = {
  camera_ready: false,
  current_gesture: 0,
  gesture_label: 'Қол көрінбей тұр',
  candidate_gesture: 0,
  hold_progress: 0,
  event_id: 0,
  active_track: null,
  scene: 'idle',
  milk_visible: false,
  milk_countdown: 0,
  status: 'JS MediaPipe жүктеліп жатыр...',
  tracks: {
    one: { title: '1 саусақ: басқа әуен', available: true },
    two: { title: '2 саусақ: Төлеген Момбеков - Бозінген (күй)', available: true }
  }
};

const stage = document.getElementById('stage');
const milkFill = document.getElementById('milkFill');
const milkingScene = document.getElementById('milkingScene');
const charEstemes = document.getElementById('char-estemes');
const charOraz = document.getElementById('char-oraz');
const charZhanyl = document.getElementById('char-zhanyl');

const stageStatus = document.getElementById('stageStatus');
const gesturePill = document.getElementById('gesturePill');
const milkPill = document.getElementById('milkPill');
const sceneLabel = document.getElementById('sceneLabel');
const sceneText = document.getElementById('sceneText');

const cameraState = document.getElementById('cameraState');
const gestureState = document.getElementById('gestureState');
const sceneState = document.getElementById('sceneState');
const systemState = document.getElementById('systemState');
const trackOneText = document.getElementById('trackOneText');
const trackTwoText = document.getElementById('trackTwoText');
const countdownBox = document.getElementById('countdownBox');
const cardOne = document.getElementById('cardOne');
const cardTwo = document.getElementById('cardTwo');
const unlockBtn = document.getElementById('unlockBtn');

const cameraFrame = document.getElementById('cameraFrame');
const cameraVideo = document.getElementById('cameraVideo');
const cameraCanvas = document.getElementById('cameraCanvas');
const cameraFallback = document.getElementById('cameraFallback');
const cameraOverlayLabel = document.getElementById('cameraOverlayLabel');
const holdFill = document.getElementById('holdFill');
const cameraContext = cameraCanvas.getContext('2d');

const audioOne = document.getElementById('audioOne');
const audioTwo = document.getElementById('audioTwo');
const tracks = { one: audioOne, two: audioTwo };

let audioUnlocked = false;
let handLandmarker;
let lastVideoTime = -1;
let rafId = 0;
let stream;

let candidateGesture = 0;
let candidateSince = performance.now() / 1000;
let lastTriggerGesture = null;
let neutralSince = performance.now() / 1000;
let pendingMilkAt = null;

function createStars() {
  const stars = document.getElementById('stars');
  const fragment = document.createDocumentFragment();
  for (let i = 0; i < 72; i += 1) {
    const star = document.createElement('span');
    star.className = 'star';
    const size = (Math.random() * 2.2 + 1).toFixed(2);
    star.style.width = `${size}px`;
    star.style.height = `${size}px`;
    star.style.left = `${(Math.random() * 100).toFixed(2)}%`;
    star.style.top = `${(Math.random() * 60).toFixed(2)}%`;
    star.style.animationDelay = `${(Math.random() * 3.8).toFixed(2)}s`;
    fragment.appendChild(star);
  }
  stars.appendChild(fragment);
}

async function primeAudio(audio) {
  audio.muted = true;
  audio.currentTime = 0;
  try {
    await audio.play();
    audio.pause();
  } catch (error) {
    // Some browsers keep autoplay locked until a stronger user gesture.
  }
  audio.currentTime = 0;
  audio.muted = false;
}

async function unlockAudio() {
  await primeAudio(audioOne);
  await primeAudio(audioTwo);
  audioUnlocked = true;
  unlockBtn.textContent = 'Дыбыс дайын';
  unlockBtn.classList.add('is-ready');
  if (state.camera_ready) {
    state.status = 'Дыбыс ашылды. Енді камераға 1 немесе 2 саусақ көрсетіңіз.';
  } else {
    state.status = 'Дыбыс ашылды. Енді камера рұқсатын беріп, қолыңызды көрсетіңіз.';
  }
  renderState();
}

function stopAllAudio() {
  Object.values(tracks).forEach((audio) => {
    audio.pause();
    audio.currentTime = 0;
  });
}

async function playTrack(trackKey) {
  if (!trackKey || !tracks[trackKey]) return;

  stopAllAudio();

  if (!audioUnlocked) {
    state.status = 'Әуенді автоматты ойнату үшін алдымен "Дыбысты іске қосу" батырмасын басыңыз.';
    renderState();
    return;
  }

  try {
    tracks[trackKey].currentTime = 0;
    await tracks[trackKey].play();
  } catch (error) {
    state.status = 'Браузер әуенді бұғаттады. Дыбысты қайта іске қосып көріңіз.';
    renderState();
  }
}

function setCharacterState(scene) {
  [charEstemes, charOraz, charZhanyl].forEach((node) => {
    node.classList.remove('is-active', 'is-muted', 'is-hidden');
  });

  if (scene === 'one') {
    charEstemes.classList.add('is-active');
    charOraz.classList.add('is-muted');
    charZhanyl.classList.add('is-muted');
  } else if (scene === 'two_wait' || scene === 'milk') {
    charOraz.classList.add('is-active');
    charEstemes.classList.add('is-muted');
    charZhanyl.classList.add('is-hidden');
  }
}

function setMilkingVisual(scene) {
  const milking = scene === 'two_wait' || scene === 'milk';
  milkingScene.classList.toggle('is-active', milking);

  if (scene === 'milk') {
    milkFill.setAttribute('y', '20');
    milkFill.setAttribute('height', '38');
  } else {
    milkFill.setAttribute('y', '58');
    milkFill.setAttribute('height', '0');
  }
}

function renderScene() {
  const scene = SCENES[state.scene] ? state.scene : 'idle';
  stage.dataset.scene = scene;
  sceneLabel.textContent = SCENES[scene].label;
  sceneText.textContent = SCENES[scene].text;
  sceneState.textContent = SCENES[scene].label;
  setCharacterState(scene);
  setMilkingVisual(scene);

  cardOne.classList.toggle('is-active', state.active_track === 'one');
  cardTwo.classList.toggle('is-active', state.active_track === 'two');

  milkPill.classList.toggle('is-alert', scene === 'milk');
  if (scene === 'milk') {
    milkPill.textContent = 'Нар сүт беріп тұр. Бұл сәт тек дұрыс күйден кейін пайда болады.';
  } else if (scene === 'two_wait') {
    milkPill.textContent = `Сүтке дейін уақыт: ${state.milk_countdown.toFixed(1)} с`;
  } else if (scene === 'one') {
    milkPill.textContent = '1 саусақта нар сүт бермейді.';
  } else {
    milkPill.textContent = 'Сүт тек 2 саусақ пен дұрыс күйде ғана ағады.';
  }
}

function renderCameraOverlay() {
  holdFill.style.width = `${Math.round(state.hold_progress * 100)}%`;

  if (!state.camera_ready) {
    cameraOverlayLabel.textContent = 'Камера қосылып жатыр...';
    return;
  }

  if (state.candidate_gesture) {
    cameraOverlayLabel.textContent = `${state.gesture_label} • Ұстау: ${Math.round(state.hold_progress * 100)}%`;
    return;
  }

  cameraOverlayLabel.textContent = state.gesture_label;
}

function renderState() {
  cameraFrame.classList.toggle('is-ready', state.camera_ready);
  cameraFallback.classList.toggle('is-hidden', state.camera_ready);
  cameraFallback.textContent = state.status;

  cameraState.textContent = state.camera_ready
    ? 'Браузер камерасы жұмыс істеп тұр'
    : 'Камера рұқсатын күтіп тұр';
  gestureState.textContent = state.gesture_label;
  systemState.textContent = state.status;
  stageStatus.textContent = state.status;
  gesturePill.textContent = `Белгі: ${state.gesture_label}`;

  if (state.milk_countdown > 0) {
    countdownBox.textContent = `Сүтке дейін: ${state.milk_countdown.toFixed(1)} с`;
    countdownBox.classList.add('is-live');
  } else if (state.scene === 'milk') {
    countdownBox.textContent = 'Сүт ағып тұр';
    countdownBox.classList.add('is-live');
  } else {
    countdownBox.textContent = 'Сүтке дейін: —';
    countdownBox.classList.remove('is-live');
  }

  const trackOneMeta = state.tracks.one ?? { title: '1 саусақ: басқа әуен', available: false };
  const trackTwoMeta = state.tracks.two ?? { title: '2 саусақ: Төлеген Момбеков - Бозінген (күй)', available: false };

  trackOneText.textContent = trackOneMeta.available
    ? `${trackOneMeta.title}. Бұл кезде нар сүт бермейді.`
    : `${trackOneMeta.title}. Файл табылмады.`;
  trackTwoText.textContent = trackTwoMeta.available
    ? `${trackTwoMeta.title}. 3 секундтан кейін ғана сүт ағады.`
    : `${trackTwoMeta.title}. Файл табылмады.`;

  renderScene();
  renderCameraOverlay();
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

  // ✅ Рисуем видео с камеры на canvas (чтобы было видно лицо)
  cameraContext.save();
  cameraContext.scale(-1, 1);
  cameraContext.drawImage(cameraVideo, -cameraCanvas.width, 0, cameraCanvas.width, cameraCanvas.height);
  cameraContext.restore();

  if (!result.landmarks.length) return;

  cameraContext.save();
  cameraContext.lineWidth = Math.max(2, cameraCanvas.width / 320);
  cameraContext.strokeStyle = 'rgba(93, 184, 243, 0.95)';
  cameraContext.fillStyle = 'rgba(242, 210, 145, 0.95)';

  result.landmarks.forEach((handLandmarks) => {
    HandLandmarker.HAND_CONNECTIONS.forEach((connection) => {
      const start = handLandmarks[connection.start];
      const end = handLandmarks[connection.end];
      cameraContext.beginPath();
      cameraContext.moveTo(start.x * cameraCanvas.width, start.y * cameraCanvas.height);
      cameraContext.lineTo(end.x * cameraCanvas.width, end.y * cameraCanvas.height);
      cameraContext.stroke();
    });

    handLandmarks.forEach((landmark) => {
      cameraContext.beginPath();
      cameraContext.arc(
        landmark.x * cameraCanvas.width,
        landmark.y * cameraCanvas.height,
        Math.max(4, cameraCanvas.width / 150),
        0,
        Math.PI * 2
      );
      cameraContext.fill();
    });
  });

  cameraContext.restore();
}

function detectGesture(result) {
  if (!result.landmarks.length) {
    return { gesture: 0, label: 'Қол көрінбей тұр' };
  }

  const landmarks = result.landmarks[0];

  function raised(tipIndex, pipIndex, mcpIndex) {
    const tip = landmarks[tipIndex];
    const pip = landmarks[pipIndex];
    const mcp = landmarks[mcpIndex];
    return tip.y < pip.y - 0.02 && pip.y < mcp.y;
  }

  const indexUp = raised(8, 6, 5);
  const middleUp = raised(12, 10, 9);
  const ringUp = raised(16, 14, 13);
  const pinkyUp = raised(20, 18, 17);

  if (indexUp && !middleUp && !ringUp && !pinkyUp) {
    return { gesture: 1, label: '1 саусақ танылды' };
  }

  if (indexUp && middleUp && !ringUp && !pinkyUp) {
    return { gesture: 2, label: '2 саусақ танылды' };
  }

  return { gesture: 0, label: 'Тек 1 немесе 2 саусақ көрсетіңіз' };
}

function triggerGesture(now, gesture) {
  lastTriggerGesture = gesture;
  state.event_id += 1;

  if (gesture === 1) {
    pendingMilkAt = null;
    state.active_track = 'one';
    state.scene = 'one';
    state.milk_visible = false;
    state.status = '1 саусақ танылды. Бірінші әуен іске қосылды.';
    renderState();
    playTrack('one');
    return;
  }

  pendingMilkAt = now + MILK_DELAY_SECONDS;
  state.active_track = 'two';
  state.scene = 'two_wait';
  state.milk_visible = false;
  state.status = '2 саусақ танылды. Дұрыс күй ойнап жатыр.';
  renderState();
  playTrack('two');
}

function advanceState(now, gesture, label) {
  state.current_gesture = gesture;
  state.gesture_label = label;

  if (VALID_GESTURES.has(gesture)) {
    neutralSince = now;

    if (gesture !== candidateGesture) {
      candidateGesture = gesture;
      candidateSince = now;
    }

    state.candidate_gesture = candidateGesture;
    state.hold_progress = Math.min(1, (now - candidateSince) / HOLD_SECONDS);

    if (state.hold_progress >= 1 && lastTriggerGesture !== gesture) {
      triggerGesture(now, gesture);
    }
  } else {
    candidateGesture = 0;
    state.candidate_gesture = 0;
    state.hold_progress = 0;

    if (now - neutralSince >= REARM_SECONDS) {
      lastTriggerGesture = null;
    }

    if (state.active_track === null && pendingMilkAt === null) {
      state.status = 'Камера жұмыс істеп тұр. Енді 1 немесе 2 саусақ көрсетіңіз.';
    }
  }

  if (pendingMilkAt !== null) {
    const remaining = pendingMilkAt - now;
    if (remaining <= 0) {
      pendingMilkAt = null;
      state.milk_visible = true;
      state.milk_countdown = 0;
      state.scene = 'milk';
      state.status = '2 саусақ расталды. 3 секундтан кейін нар сүт берді.';
    } else {
      state.milk_countdown = Number(remaining.toFixed(1));
      if (state.scene === 'two_wait') {
        state.status = `Жаңыл доюға дайын. Сүтке дейін: ${remaining.toFixed(1)} с`;
      }
    }
  } else {
    state.milk_countdown = 0;
  }

  renderState();
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
    const { gesture, label } = detectGesture(result);
    advanceState(now / 1000, gesture, label);
  }

  rafId = requestAnimationFrame(processFrame);
}

async function loadBootstrapState() {
  try {
    const response = await fetch('/api/state', { cache: 'no-store' });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const payload = await response.json();
    state.tracks = payload.tracks ?? state.tracks;
    state.status = payload.status ?? state.status;
  } catch (error) {
    state.status = 'Серверден бастапқы күйді оқу мүмкін болмады, бірақ локалды камерамен жұмыс жалғасады.';
  }
  renderState();
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
  state.camera_ready = true;
  state.status = 'Камера жұмыс істеп тұр. Енді 1 немесе 2 саусақ көрсетіңіз.';
  renderState();
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

function attachAudioListeners() {
  audioOne.addEventListener('error', () => {
    state.tracks.one = { ...state.tracks.one, available: false };
    renderState();
  });

  audioTwo.addEventListener('error', () => {
    state.tracks.two = { ...state.tracks.two, available: false };
    renderState();
  });
}

async function init() {
  createStars();
  attachAudioListeners();
  renderState();
  await loadBootstrapState();

  try {
    await Promise.all([createHandLandmarker(), startCamera()]);
    processFrame();
  } catch (error) {
    state.camera_ready = false;
    state.status = `MediaPipe JS не камера іске қосылмады: ${error.message}`;
    renderState();
  }
}

unlockBtn.addEventListener('click', unlockAudio);

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
