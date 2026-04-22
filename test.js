import {
  FilesetResolver,
  HandLandmarker
} from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.34/vision_bundle.mjs';

const WASM_ROOT = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.34/wasm';
const MODEL_ASSET_PATH = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';

const HOLD_SECONDS = 0.82;
const REARM_SECONDS = 0.55;
const VALID_GESTURES = new Set([1, 2, 3, 4]);
const OPTION_MARKS = ['A', 'Ә', 'Б', 'В'];
const FINGER_CONNECTIONS = [
  [1, 2], [2, 3], [3, 4],
  [5, 6], [6, 7], [7, 8],
  [9, 10], [10, 11], [11, 12],
  [13, 14], [14, 15], [15, 16],
  [17, 18], [18, 19], [19, 20]
];
const FINGER_TIPS = new Set([4, 8, 12, 16, 20]);

const QUIZ = [
  {
    prompt: 'Т. Ахтановтың ең алғашқы әңгімесі қалай аталады?',
    detail: 'Тахауи Ахтановтың әдебиетке келген алғашқы әңгімесін таңдаңыз.',
    options: ['«Шырағың сөнбесін»', '«Күй аңызы»', '«Боран»', '«Махаббат мұңы»'],
    correct: 1
  },
  {
    prompt: 'Әңгіменің негізгі тақырыбы қандай?',
    detail: 'Мәтіндегі басты идеяны табыңыз.',
    options: ['Табиғат көрінісі', 'Өнер құдіреті', 'Соғыс зардабы', 'Аңшылық саятшылық'],
    correct: 1
  },
  {
    prompt: 'Шығармадағы басты кейіпкер, тағдыры ауыр күйшіні табыңыз:',
    detail: 'Оқиға желісінің өзегіндегі күйшіні белгілеңіз.',
    options: ['Оразымбет', 'Төлеген', 'Естемес', 'Құрманғазы'],
    correct: 2
  },
  {
    prompt: 'Естемес күйшімен бірге жол жүрген жас жігіт кім?',
    detail: 'Әңгімеде Естемеспен қатар көрінетін жас серікті атаңыз.',
    options: ['Оразымбет', 'Ақан', 'Біржан', 'Сәкен'],
    correct: 0
  },
  {
    prompt: 'Естеместің күйі қандай жануарға әсер етіп, иітеді?',
    detail: '«Нар идірген» аңызына қатысты жануарды таңдаңыз.',
    options: ['Маңыраған қойға', 'Ботасы өлген боз інгенге', 'Құлынды биеге', 'Жалын тартқан арғымаққа'],
    correct: 1
  },
  {
    prompt: '«Нар идірген» аңызының негізгі мәні неде?',
    detail: 'Аңыздың өзегіндегі басты ойды белгілеңіз.',
    options: ['Түйенің жүйріктігін сипаттау', 'Күйдің құдіретімен жануардың сүтін жібіту', 'Мал шаруашылығын дамыту', 'Түйешілердің еңбегін дәріптеу'],
    correct: 1
  },
  {
    prompt: 'Естемес күйшінің көңілі не себепті түседі?',
    detail: 'Күйшінің жан күйзелісінің себебін таңдаңыз.',
    options: ['Аспабы сынып қалғандықтан', 'Ел-жұртынан айырылғандықтан', 'Жас қыздың өзіне деген сезімі жоқ екенін білгендіктен', 'Күйін ешкім түсінбегендіктен'],
    correct: 2
  },
  {
    prompt: 'Көшпелілер өміріндегі өнердің атқаратын қызметі қандай?',
    detail: 'Өнердің қоғамдық және рухани рөлін анықтаңыз.',
    options: ['Тек көңіл көтеру', 'Эстетикалық әсер беріп, қоғамдық қарым-қатынасқа дәнекер болу', 'Тек қана сауда-саттыққа көмектесу', 'Тек жаугершілікте ұран болу'],
    correct: 1
  },
  {
    prompt: '«Қайыс жоны жұмсарып, тіршілікке нәр бергендей болды» деген тіркес нені білдіреді?',
    detail: 'Осы бейнелі тіркестің мағынасын табыңыз.',
    options: ['Жаңбыр жауғанын', 'Түйенің семіргенін', 'Күйдің әсерінен малдың жібіп, иігенін', 'Табиғаттың жылынғанын'],
    correct: 2
  },
  {
    prompt: 'Қазақ күйшілік дәстүрі туралы мәтінде не айтылған?',
    detail: 'Күйдің жалпы мәнін сипаттайтын нұсқаны таңдаңыз.',
    options: ['Күй тек өткенді ғана жырлайды', 'Күйшілердің есімдері ұмытылып барады', 'Күй – тарихтың өткені мен бүгінін көз алдына әкелетін ерекше қасиет', 'Күй тек жастарға арналған өнер'],
    correct: 2
  }
];

const state = {
  cameraReady: false,
  cameraStatus: 'Камера мен MediaPipe жүктеліп жатыр...',
  gestureLabel: 'Қол көрінбей тұр',
  currentGesture: 0,
  holdProgress: 0,
  questionIndex: 0,
  selectedOption: null,
  correctCount: 0,
  wrongCount: 0,
  completed: false,
  locked: false,
  status: 'Камераға рұқсат беріңіз немесе жауап батырмасын басыңыз.',
  selectedText: '—'
};

const questionCard = document.getElementById('questionCard');
const resultCard = document.getElementById('resultCard');
const questionCounter = document.getElementById('questionCounter');
const questionPrompt = document.getElementById('questionPrompt');
const questionText = document.getElementById('questionText');
const statusLine = document.getElementById('statusLine');
const optionGrid = document.getElementById('optionGrid');

const progressLabel = document.getElementById('progressLabel');
const progressFill = document.getElementById('progressFill');
const scorePill = document.getElementById('scorePill');
const correctCountEl = document.getElementById('correctCount');
const wrongCountEl = document.getElementById('wrongCount');
const selectedLabelEl = document.getElementById('selectedLabel');

const cameraVideo = document.getElementById('cameraVideo');
const cameraCanvas = document.getElementById('cameraCanvas');
const cameraFallback = document.getElementById('cameraFallback');
const cameraOverlayLabel = document.getElementById('cameraOverlayLabel');
const holdFill = document.getElementById('holdFill');
const cameraState = document.getElementById('cameraState');
const gestureState = document.getElementById('gestureState');
const questionState = document.getElementById('questionState');
const systemState = document.getElementById('systemState');
const cameraContext = cameraCanvas.getContext('2d');

const fingerCards = [
  document.getElementById('fingerCard1'),
  document.getElementById('fingerCard2'),
  document.getElementById('fingerCard3'),
  document.getElementById('fingerCard4')
];

const resultTitle = document.getElementById('resultTitle');
const resultText = document.getElementById('resultText');
const resultScore = document.getElementById('resultScore');
const resultCorrect = document.getElementById('resultCorrect');
const resultWrong = document.getElementById('resultWrong');
const answerKey = document.getElementById('answerKey');
const restartBtn = document.getElementById('restartBtn');

const fireworksCanvas = document.getElementById('fireworksCanvas');
const fireworksContext = fireworksCanvas.getContext('2d');

let handLandmarker;
let stream;
let rafId = 0;
let lastVideoTime = -1;
let candidateGesture = 0;
let candidateSince = performance.now() / 1000;
let neutralSince = performance.now() / 1000;
let lastAcceptedGesture = null;
let transitionTimer = 0;

const fireworks = {
  particles: [],
  rafId: 0,
  lastTime: 0
};

function clamp(value, min = 0, max = 1) {
  return Math.min(max, Math.max(min, value));
}

function clearTransitionTimer() {
  if (transitionTimer) {
    clearTimeout(transitionTimer);
    transitionTimer = 0;
  }
}

function getCurrentQuestion() {
  return QUIZ[state.questionIndex];
}

function resizeCameraCanvas() {
  const width = cameraVideo.videoWidth || 960;
  const height = cameraVideo.videoHeight || 540;
  if (cameraCanvas.width !== width || cameraCanvas.height !== height) {
    cameraCanvas.width = width;
    cameraCanvas.height = height;
  }
}

function resizeFireworksCanvas() {
  const rect = fireworksCanvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const nextWidth = Math.max(1, Math.round(rect.width * dpr));
  const nextHeight = Math.max(1, Math.round(rect.height * dpr));

  if (fireworksCanvas.width !== nextWidth || fireworksCanvas.height !== nextHeight) {
    fireworksCanvas.width = nextWidth;
    fireworksCanvas.height = nextHeight;
  }

  fireworksContext.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function clearCanvas() {
  cameraContext.clearRect(0, 0, cameraCanvas.width, cameraCanvas.height);
}

function paintCameraBackground() {
  const width = cameraCanvas.width;
  const height = cameraCanvas.height;
  const gradient = cameraContext.createLinearGradient(0, 0, 0, height);
  gradient.addColorStop(0, '#030708');
  gradient.addColorStop(1, '#0a1416');
  cameraContext.fillStyle = gradient;
  cameraContext.fillRect(0, 0, width, height);

  const glow = cameraContext.createRadialGradient(width * 0.5, height * 0.45, 8, width * 0.5, height * 0.45, width * 0.38);
  glow.addColorStop(0, 'rgba(112, 255, 152, 0.16)');
  glow.addColorStop(1, 'rgba(112, 255, 152, 0)');
  cameraContext.fillStyle = glow;
  cameraContext.fillRect(0, 0, width, height);
}

function drawResults(result) {
  resizeCameraCanvas();
  clearCanvas();
  paintCameraBackground();

  if (!result.landmarks.length) {
    return;
  }

  cameraContext.save();
  cameraContext.lineCap = 'round';
  cameraContext.lineJoin = 'round';

  result.landmarks.forEach((handLandmarks) => {
    cameraContext.strokeStyle = 'rgba(112, 255, 152, 0.92)';
    cameraContext.lineWidth = Math.max(3, cameraCanvas.width / 260);
    cameraContext.shadowBlur = 18;
    cameraContext.shadowColor = 'rgba(112, 255, 152, 0.32)';

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

function raised(landmarks, tipIndex, pipIndex, mcpIndex) {
  const tip = landmarks[tipIndex];
  const pip = landmarks[pipIndex];
  const mcp = landmarks[mcpIndex];
  return tip.y < pip.y - 0.02 && pip.y < mcp.y;
}

function detectGesture(result) {
  if (!result.landmarks.length) {
    return { gesture: 0, label: 'Қол көрінбей тұр' };
  }

  const landmarks = result.landmarks[0];
  const raisedCount = [
    raised(landmarks, 8, 6, 5),
    raised(landmarks, 12, 10, 9),
    raised(landmarks, 16, 14, 13),
    raised(landmarks, 20, 18, 17)
  ].filter(Boolean).length;

  if (VALID_GESTURES.has(raisedCount)) {
    return { gesture: raisedCount, label: `${raisedCount} саусақ • ${OPTION_MARKS[raisedCount - 1]} нұсқасы` };
  }

  return { gesture: 0, label: '1-4 саусақ көрсетіңіз' };
}

function updateFingerCards() {
  fingerCards.forEach((card, index) => {
    card.classList.toggle('is-active', state.currentGesture === index + 1);
  });
}

function updateResultView() {
  questionCard.classList.toggle('hidden', state.completed);
  resultCard.classList.toggle('is-visible', state.completed);

  if (!state.completed) {
    return;
  }

  resultTitle.textContent = state.correctCount === QUIZ.length
    ? 'Керемет нәтиже'
    : 'Тест аяқталды';
  resultText.textContent = `Сіз ${state.correctCount} дұрыс, ${state.wrongCount} қате жауап бердіңіз. Қажет болса тестті қайта бастауға болады.`;
  resultScore.textContent = `${state.correctCount} / ${QUIZ.length}`;
  resultCorrect.textContent = String(state.correctCount);
  resultWrong.textContent = String(state.wrongCount);

  answerKey.innerHTML = '';
  QUIZ.forEach((question, index) => {
    const answer = document.createElement('div');
    answer.className = 'answer-key__item';
    answer.textContent = `${index + 1}. ${OPTION_MARKS[question.correct]} — ${question.options[question.correct]}`;
    answerKey.appendChild(answer);
  });
}

function renderOptions() {
  if (state.completed) {
    optionGrid.innerHTML = '';
    return;
  }

  const question = getCurrentQuestion();
  const previewOption = VALID_GESTURES.has(state.currentGesture) ? state.currentGesture - 1 : null;
  const statusLower = state.status.toLowerCase();
  optionGrid.innerHTML = '';

  question.options.forEach((option, index) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'option-btn';
    button.dataset.optionIndex = String(index);
    button.disabled = state.locked;
    button.innerHTML = `
      <span class="option-mark">${OPTION_MARKS[index]}</span>
      <span class="option-copy">
        <strong>${OPTION_MARKS[index]} нұсқасы</strong>
        <span>${option}</span>
      </span>
    `;

    if (!state.locked && previewOption === index) {
      button.classList.add('is-active');
    }

    if (state.selectedOption === index && state.locked && statusLower.includes('қате')) {
      button.classList.add('is-wrong');
    }

    if (state.locked && statusLower.includes('дұрыс') && question.correct === index) {
      button.classList.add('is-correct');
    }

    button.addEventListener('click', () => handleAnswer(index));
    optionGrid.appendChild(button);
  });
}

function renderQuestionView() {
  if (state.completed) {
    return;
  }

  const question = getCurrentQuestion();
  questionCounter.textContent = `Сұрақ ${state.questionIndex + 1} / ${QUIZ.length}`;
  questionPrompt.textContent = question.prompt;
  questionText.textContent = question.detail;
}

function renderStatus() {
  const statusLower = state.status.toLowerCase();
  statusLine.textContent = state.status;
  statusLine.classList.toggle('is-correct', statusLower.includes('дұрыс'));
  statusLine.classList.toggle('is-wrong', statusLower.includes('қате'));
}

function renderProgress() {
  const visibleIndex = state.completed ? QUIZ.length : state.questionIndex + 1;
  const ratio = state.completed ? 1 : state.questionIndex / QUIZ.length;

  progressLabel.textContent = `Сұрақ ${visibleIndex} / ${QUIZ.length}`;
  progressFill.style.width = `${Math.round(ratio * 100)}%`;
  scorePill.textContent = `Ұпай: ${state.correctCount} / ${QUIZ.length}`;
  correctCountEl.textContent = String(state.correctCount);
  wrongCountEl.textContent = String(state.wrongCount);
  selectedLabelEl.textContent = state.selectedText;
}

function renderCameraState() {
  cameraFallback.textContent = state.cameraStatus;
  cameraFallback.classList.toggle('is-hidden', state.cameraReady);
  cameraOverlayLabel.textContent = state.cameraReady
    ? `${state.gestureLabel}${state.holdProgress > 0 ? ` • Ұстау: ${Math.round(state.holdProgress * 100)}%` : ''}`
    : 'Камера қосылып жатыр...';
  holdFill.style.width = `${Math.round(state.holdProgress * 100)}%`;

  cameraState.textContent = state.cameraReady ? 'Браузер камерасы жұмыс істеп тұр' : 'Камера рұқсатын күтіп тұр';
  gestureState.textContent = state.gestureLabel;
  questionState.textContent = `${Math.min(state.questionIndex + 1, QUIZ.length)} / ${QUIZ.length}`;
  systemState.textContent = state.completed
    ? 'Тест аяқталды'
    : state.locked
      ? 'Жауап тіркелді, келесі күйге өтуде'
      : state.cameraReady
        ? 'Gesture жауап беру дайын'
        : 'Gesture жүйесі іске қосылып жатыр...';
}

function render() {
  renderQuestionView();
  renderOptions();
  renderStatus();
  renderProgress();
  renderCameraState();
  updateFingerCards();
  updateResultView();
}

function startFireworksLoop() {
  if (!fireworks.rafId) {
    fireworks.lastTime = performance.now();
    fireworks.rafId = requestAnimationFrame(stepFireworks);
  }
}

function stopFireworksLoop() {
  if (fireworks.rafId) {
    cancelAnimationFrame(fireworks.rafId);
    fireworks.rafId = 0;
  }
  fireworks.particles = [];
  fireworksContext.clearRect(0, 0, fireworksCanvas.width, fireworksCanvas.height);
}

function burstAtElement(element) {
  if (!element) {
    return;
  }

  resizeFireworksCanvas();
  const canvasRect = fireworksCanvas.getBoundingClientRect();
  const rect = element.getBoundingClientRect();
  const x = rect.left - canvasRect.left + rect.width / 2;
  const y = rect.top - canvasRect.top + rect.height / 2;
  const colors = ['#75ff9e', '#b6ff8b', '#e8ffd9', '#28d070'];

  for (let i = 0; i < 30; i += 1) {
    const angle = (Math.PI * 2 * i) / 30 + Math.random() * 0.36;
    const speed = 90 + Math.random() * 150;
    fireworks.particles.push({
      x,
      y,
      vx: Math.cos(angle) * speed,
      vy: Math.sin(angle) * speed - 34,
      life: 0.95 + Math.random() * 0.35,
      radius: 2 + Math.random() * 3.6,
      color: colors[i % colors.length]
    });
  }

  startFireworksLoop();
}

function stepFireworks(now) {
  const dt = Math.min(0.033, (now - fireworks.lastTime) / 1000 || 0.016);
  fireworks.lastTime = now;

  resizeFireworksCanvas();
  fireworksContext.clearRect(0, 0, fireworksCanvas.width, fireworksCanvas.height);

  fireworks.particles = fireworks.particles.filter((particle) => {
    particle.life -= dt;
    if (particle.life <= 0) {
      return false;
    }

    particle.vy += 220 * dt;
    particle.x += particle.vx * dt;
    particle.y += particle.vy * dt;

    fireworksContext.save();
    fireworksContext.globalAlpha = clamp(particle.life / 1.1);
    fireworksContext.fillStyle = particle.color;
    fireworksContext.shadowBlur = 18;
    fireworksContext.shadowColor = particle.color;
    fireworksContext.beginPath();
    fireworksContext.arc(particle.x, particle.y, particle.radius, 0, Math.PI * 2);
    fireworksContext.fill();
    fireworksContext.restore();

    return true;
  });

  if (fireworks.particles.length) {
    fireworks.rafId = requestAnimationFrame(stepFireworks);
  } else {
    stopFireworksLoop();
  }
}

function finishQuiz() {
  state.completed = true;
  state.locked = false;
  state.selectedOption = null;
  state.selectedText = 'Тест аяқталды';
  state.status = 'Тест аяқталды. Қорытынды төменде көрсетілді.';
  state.holdProgress = 0;
  render();
}

function queueNextQuestion() {
  clearTransitionTimer();
  transitionTimer = window.setTimeout(() => {
    state.questionIndex += 1;
    state.selectedOption = null;
    state.locked = false;
    state.holdProgress = 0;
    state.selectedText = '—';
    candidateGesture = 0;

    if (state.questionIndex >= QUIZ.length) {
      finishQuiz();
      return;
    }

    state.status = 'Келесі сұраққа дайын. Жауапты батырмамен не жестпен таңдаңыз.';
    render();
  }, 1500);
}

function handleCorrectAnswer(index) {
  state.correctCount += 1;
  state.selectedOption = index;
  state.selectedText = `${OPTION_MARKS[index]} нұсқасы`;
  state.status = `Дұрыс жауап: ${OPTION_MARKS[index]} — ${getCurrentQuestion().options[index]}.`;
  render();
  burstAtElement(optionGrid.querySelector(`[data-option-index="${index}"]`));
  queueNextQuestion();
}

function handleWrongAnswer(index) {
  clearTransitionTimer();
  state.wrongCount += 1;
  state.selectedOption = index;
  state.selectedText = `${OPTION_MARKS[index]} нұсқасы`;
  state.status = `Қате жауап: ${OPTION_MARKS[index]}. Қайта байқап көріңіз.`;
  render();

  transitionTimer = window.setTimeout(() => {
    state.selectedOption = null;
    state.locked = false;
    state.holdProgress = 0;
    state.selectedText = '—';
    state.status = 'Осы сұраққа қайта жауап беріңіз.';
    render();
  }, 950);
}

function handleAnswer(index) {
  if (state.completed || state.locked) {
    return;
  }

  const question = getCurrentQuestion();
  state.locked = true;

  if (index === question.correct) {
    handleCorrectAnswer(index);
    return;
  }

  handleWrongAnswer(index);
}

function advanceSelection(now, gesture, label) {
  state.currentGesture = gesture;
  state.gestureLabel = label;

  if (state.completed) {
    state.selectedText = 'Тест аяқталды';
    state.holdProgress = 0;
    render();
    return;
  }

  if (state.locked) {
    state.holdProgress = 0;
    render();
    return;
  }

  state.selectedText = VALID_GESTURES.has(gesture)
    ? `${OPTION_MARKS[gesture - 1]} (${gesture} саусақ)`
    : '—';

  if (VALID_GESTURES.has(gesture)) {
    neutralSince = now;

    if (candidateGesture !== gesture) {
      candidateGesture = gesture;
      candidateSince = now;
    }

    state.holdProgress = Math.min(1, (now - candidateSince) / HOLD_SECONDS);

    if (state.holdProgress >= 1 && lastAcceptedGesture !== gesture) {
      lastAcceptedGesture = gesture;
      handleAnswer(gesture - 1);
      return;
    }
  } else {
    candidateGesture = 0;
    state.holdProgress = 0;

    if (now - neutralSince >= REARM_SECONDS) {
      lastAcceptedGesture = null;
    }
  }

  render();
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
    advanceSelection(now / 1000, gesture, label);
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
  state.cameraStatus = 'Қол камерасы дайын';
  state.status = 'Gesture жауап беру дайын. 1-4 саусақпен таңдауға болады.';
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

function resetQuiz() {
  clearTransitionTimer();
  state.questionIndex = 0;
  state.selectedOption = null;
  state.correctCount = 0;
  state.wrongCount = 0;
  state.completed = false;
  state.locked = false;
  state.status = state.cameraReady
    ? 'Gesture жауап беру дайын. 1-4 саусақпен таңдауға болады.'
    : 'Камераға рұқсат беріңіз немесе жауап батырмасын басыңыз.';
  state.selectedText = '—';
  state.currentGesture = 0;
  state.gestureLabel = state.cameraReady ? '1-4 саусақ көрсетіңіз' : 'Қол көрінбей тұр';
  state.holdProgress = 0;
  candidateGesture = 0;
  lastAcceptedGesture = null;
  stopFireworksLoop();
  render();
}

function handleKeyboard(event) {
  if (state.completed || state.locked) {
    return;
  }

  if (event.key >= '1' && event.key <= '4') {
    handleAnswer(Number(event.key) - 1);
  }
}

function attachListeners() {
  restartBtn.addEventListener('click', resetQuiz);
  window.addEventListener('resize', resizeFireworksCanvas);
  window.addEventListener('keydown', handleKeyboard);
}

async function init() {
  resizeFireworksCanvas();
  render();
  attachListeners();

  try {
    await Promise.all([createHandLandmarker(), startCamera()]);
    processFrame();
  } catch (error) {
    state.cameraReady = false;
    state.cameraStatus = `Камера не MediaPipe іске қосылмады: ${error.message}`;
    state.gestureLabel = 'Қол тану қолжетімсіз';
    state.status = 'Камера іске қосылмады, бірақ тестті батырмамен тапсыруға болады.';
    render();
  }
}

window.addEventListener('beforeunload', () => {
  clearTransitionTimer();

  if (rafId) {
    cancelAnimationFrame(rafId);
  }

  stopFireworksLoop();
  window.removeEventListener('keydown', handleKeyboard);

  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
  }

  handLandmarker?.close();
});

init();
