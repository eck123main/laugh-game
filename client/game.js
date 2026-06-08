// ─── Config ───────────────────────────────────────────────
const RTC_CONFIG = { iceServers: [{ urls: 'stun:stun.l.google.com:19302' }] };
const TOTAL_ROUNDS = 5;

// Laugh detection config
const LAUGH_THRESHOLD_DEFAULT = 0.45; // mouth openness ratio to trigger laugh
const LAUGH_SUSTAIN_MS = 600;         // must hold open this long to count
const CALIBRATION_FRAMES = 60;        // frames to sample neutral face

// ─── State ────────────────────────────────────────────────
let socket, pc, localStream;
let roomCode = null;
let isHost = false;
let timerInterval = null;
let timerSecs = 0;
let scoreYou = 0, scoreThem = 0, round = 1;
let roundActive = false;

// Laugh detection state
let faceMesh = null;
let laughDetectionActive = false;
let laughStartTime = null;
let laughTriggered = false;
let baselineMouthRatio = 0.02; // calibrated neutral ratio
let sensitivityMultiplier = 1.0;
let calibrating = false;
let calibrationSamples = [];
let detectionCanvas, detectionCtx;

// ─── Socket setup ─────────────────────────────────────────
socket = io();

socket.on('room_created', (code) => {
  roomCode = code;
  isHost = true;
  document.getElementById('room-code-display').textContent = code;
  showScreen('waiting');
  startCamera();
});

socket.on('room_joined', (code) => {
  roomCode = code;
  isHost = false;
});

socket.on('error', (msg) => {
  document.getElementById('lobby-error').textContent = msg;
});

socket.on('game_start', async () => {
  await startCamera();
  await setupPeerConnection();
  showScreen('game');
  await initFaceDetection();
  if (isHost) {
    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);
    socket.emit('game_event', { code: roomCode, type: 'offer', data: offer });
    beginRoundCountdown();
  }
});

socket.on('game_event', async ({ type, data }) => {
  switch (type) {
    case 'offer':
      await pc.setRemoteDescription(new RTCSessionDescription(data));
      const answer = await pc.createAnswer();
      await pc.setLocalDescription(answer);
      socket.emit('game_event', { code: roomCode, type: 'answer', data: answer });
      showScreen('game');
      await initFaceDetection();
      beginRoundCountdown();
      break;
    case 'answer':
      await pc.setRemoteDescription(new RTCSessionDescription(data));
      break;
    case 'ice':
      if (data) await pc.addIceCandidate(new RTCIceCandidate(data));
      break;
    case 'i_laughed':
      endRound(true);
      break;
    case 'next_round':
      nextRoundGo();
      break;
    case 'rematch_accept':
      resetGame();
      break;
    case 'rematch_request':
      if (confirm('Opponent wants a rematch! Accept?')) {
        socket.emit('game_event', { code: roomCode, type: 'rematch_accept' });
        resetGame();
      }
      break;
  }
});

socket.on('opponent_left', () => {
  alert('Opponent disconnected.');
  goLobby();
});

// ─── WebRTC ───────────────────────────────────────────────
async function setupPeerConnection() {
  pc = new RTCPeerConnection(RTC_CONFIG);
  if (localStream) localStream.getTracks().forEach(track => pc.addTrack(track, localStream));
  pc.ontrack = (e) => {
    const remoteVideo = document.getElementById('video-remote');
    remoteVideo.srcObject = e.streams[0];
    remoteVideo.onloadedmetadata = () => {
      hideOverlay('overlay-remote');
    };
  };
  pc.onicecandidate = (e) => {
    socket.emit('game_event', { code: roomCode, type: 'ice', data: e.candidate });
  };
}

// ─── Camera ───────────────────────────────────────────────
async function startCamera() {
  try {
    localStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    const vid = document.getElementById('video-local');
    vid.srcObject = localStream;
    await new Promise(res => vid.onloadedmetadata = res);
    hideOverlay('overlay-local');
  } catch (e) {
    console.warn('Camera unavailable:', e);
    setStatus('⚠ camera unavailable — manual mode');
  }
}

// ─── Face Mesh / Laugh Detection ──────────────────────────
async function initFaceDetection() {
  // Create hidden canvas for processing
  detectionCanvas = document.createElement('canvas');
  detectionCanvas.width = 320;
  detectionCanvas.height = 240;
  detectionCtx = detectionCanvas.getContext('2d');

  try {
    // Load MediaPipe Face Mesh from CDN
    faceMesh = new FaceMesh({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4/${file}`
    });

    faceMesh.setOptions({
      maxNumFaces: 1,
      refineLandmarks: true,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5
    });

    faceMesh.onResults(onFaceResults);
    await faceMesh.initialize();

    console.log('Face detection ready');
    updateSensitivityLabel();

  } catch (e) {
    console.warn('Face detection unavailable, falling back to manual:', e);
    faceMesh = null;
    showManualFallback();
  }
}

// Key mouth landmark indices in MediaPipe Face Mesh
// Upper lip top: 13, Lower lip bottom: 14
// Left mouth corner: 78, Right mouth corner: 308
const UPPER_LIP = 13;
const LOWER_LIP = 14;
const LEFT_CORNER = 78;
const RIGHT_CORNER = 308;

function getMouthRatio(landmarks) {
  const upper = landmarks[UPPER_LIP];
  const lower = landmarks[LOWER_LIP];
  const left  = landmarks[LEFT_CORNER];
  const right = landmarks[RIGHT_CORNER];

  const mouthHeight = Math.abs(lower.y - upper.y);
  const mouthWidth  = Math.abs(right.x - left.x);

  if (mouthWidth < 0.001) return 0;
  return mouthHeight / mouthWidth;
}

function onFaceResults(results) {
  if (!laughDetectionActive && !calibrating) return;
  if (!results.multiFaceLandmarks || results.multiFaceLandmarks.length === 0) return;

  const landmarks = results.multiFaceLandmarks[0];
  const ratio = getMouthRatio(landmarks);

  // Update visual indicator
  updateMouthIndicator(ratio);

  if (calibrating) {
    calibrationSamples.push(ratio);
    if (calibrationSamples.length >= CALIBRATION_FRAMES) {
      finishCalibration();
    }
    return;
  }

  if (!laughDetectionActive || laughTriggered) return;

  // Dynamic threshold = baseline * sensitivity multiplier * laugh factor
  const threshold = baselineMouthRatio + (0.08 * sensitivityMultiplier);

  if (ratio > threshold) {
    if (!laughStartTime) {
      laughStartTime = Date.now();
    } else if (Date.now() - laughStartTime > LAUGH_SUSTAIN_MS) {
      triggerLaughDetected();
    }
  } else {
    laughStartTime = null;
  }
}

async function runFaceDetection() {
  if (!faceMesh || !localStream) return;
  const video = document.getElementById('video-local');
  if (video.readyState < 2) {
    requestAnimationFrame(runFaceDetection);
    return;
  }
  detectionCtx.drawImage(video, 0, 0, 320, 240);
  await faceMesh.send({ image: detectionCanvas });
  requestAnimationFrame(runFaceDetection);
}

function triggerLaughDetected() {
  if (laughTriggered || !roundActive) return;
  laughTriggered = true;
  laughDetectionActive = false;
  socket.emit('game_event', { code: roomCode, type: 'i_laughed' });
  endRound(false);
}

// ─── Calibration ──────────────────────────────────────────
function startCalibration() {
  calibrating = true;
  calibrationSamples = [];
  setStatus('😐 hold a neutral face...', false);
  showCalibrationUI(true);
}

function finishCalibration() {
  calibrating = false;
  const avg = calibrationSamples.reduce((a, b) => a + b, 0) / calibrationSamples.length;
  baselineMouthRatio = avg;
  console.log('Baseline mouth ratio:', baselineMouthRatio.toFixed(4));
  showCalibrationUI(false);
  setStatus('calibrated! get ready...', false);
  setTimeout(() => beginRoundCountdown(), 800);
}

// ─── UI helpers for detection ──────────────────────────────
function updateMouthIndicator(ratio) {
  const el = document.getElementById('mouth-indicator');
  if (!el) return;
  const threshold = baselineMouthRatio + (0.08 * sensitivityMultiplier);
  const pct = Math.min((ratio / (threshold * 1.5)) * 100, 100);
  el.style.width = pct + '%';
  el.style.background = ratio > threshold ? 'var(--danger)' : 'var(--accent)';
}

function showManualFallback() {
  const btn = document.getElementById('laugh-btn');
  if (btn) {
    btn.style.display = 'block';
    btn.textContent = '😂  i laughed  (tap if detected fails)';
  }
  const ind = document.getElementById('detection-bar');
  if (ind) ind.style.display = 'none';
}

function showCalibrationUI(show) {
  const el = document.getElementById('calibration-msg');
  if (el) el.style.display = show ? 'block' : 'none';
}

function updateSensitivityLabel() {
  const el = document.getElementById('sensitivity-label');
  if (el) {
    const labels = { 0.6: 'high', 1.0: 'medium', 1.5: 'low' };
    el.textContent = labels[sensitivityMultiplier] || 'medium';
  }
}

function setSensitivity(val) {
  sensitivityMultiplier = parseFloat(val);
  updateSensitivityLabel();
}

// ─── Lobby actions ────────────────────────────────────────
function createRoom() {
  document.getElementById('lobby-error').textContent = '';
  const code = Math.random().toString(36).substring(2, 6).toUpperCase();
  socket.emit('create_room', code);
}

function joinRoom() {
  document.getElementById('lobby-error').textContent = '';
  const code = document.getElementById('join-code').value.trim().toUpperCase();
  if (code.length !== 4) {
    document.getElementById('lobby-error').textContent = 'enter a 4-letter code';
    return;
  }
  socket.emit('join_room', code);
}

function goLobby() {
  cleanup();
  showScreen('lobby');
}

function leaveGame() {
  if (confirm('Leave the game?')) goLobby();
}

// ─── Game flow ────────────────────────────────────────────
function beginRoundCountdown() {
  roundActive = false;
  laughDetectionActive = false;
  laughTriggered = false;
  laughStartTime = null;
  document.getElementById('laugh-btn').disabled = true;
  document.getElementById('round-num').textContent = round;
  stopTimer();
  timerSecs = 0;
  updateTimerDisplay();

  // Calibrate on first round, then just count down
  if (round === 1 && faceMesh) {
    startCalibration();
    runFaceDetection();
    return;
  }

  let countdown = 3;
  setStatus('get ready...');

  const cd = setInterval(() => {
    if (countdown > 0) {
      setStatus(countdown + '...');
      countdown--;
    } else {
      clearInterval(cd);
      startRound();
    }
  }, 1000);
}

function startRound() {
  setStatus('😐 keep a straight face!', true);
  roundActive = true;
  laughDetectionActive = true;
  laughTriggered = false;
  laughStartTime = null;

  // Manual fallback button (hidden when AI detection is on)
  if (!faceMesh) {
    document.getElementById('laugh-btn').disabled = false;
  }

  startTimer();
  if (faceMesh) runFaceDetection();
}

function startTimer() {
  stopTimer();
  timerSecs = 0;
  timerInterval = setInterval(() => {
    timerSecs++;
    updateTimerDisplay();
    if (timerSecs >= 60) document.getElementById('timer').classList.add('danger');
  }, 1000);
}

function stopTimer() {
  if (timerInterval) { clearInterval(timerInterval); timerInterval = null; }
}

function updateTimerDisplay() {
  const m = Math.floor(timerSecs / 60);
  const s = timerSecs % 60;
  document.getElementById('timer').textContent = m + ':' + String(s).padStart(2, '0');
}

function iLaughed() {
  if (!roundActive) return;
  socket.emit('game_event', { code: roomCode, type: 'i_laughed' });
  endRound(false);
}

function endRound(iWon) {
  roundActive = false;
  laughDetectionActive = false;
  stopTimer();
  document.getElementById('laugh-btn').disabled = true;

  if (iWon) scoreYou++; else scoreThem++;
  updateScores();

  document.getElementById('round-emoji').textContent = iWon ? '🏆' : '😭';
  document.getElementById('round-title').textContent = iWon ? 'they laughed!' : 'you laughed';
  document.getElementById('round-title').className = 'result-title ' + (iWon ? 'win' : 'lose');
  document.getElementById('round-sub').textContent = iWon ? 'you win the round' : 'opponent wins the round';
  document.getElementById('rs-you').textContent = scoreYou;
  document.getElementById('rs-them').textContent = scoreThem;

  const needed = Math.ceil(TOTAL_ROUNDS / 2);
  const btn = document.getElementById('next-btn');
  if (scoreYou >= needed || scoreThem >= needed || round >= TOTAL_ROUNDS) {
    btn.textContent = 'see final result';
    btn.onclick = showGameOver;
  } else {
    btn.textContent = 'next round →';
    btn.onclick = nextRound;
  }

  showScreen('round');
}

function nextRound() {
  round++;
  socket.emit('game_event', { code: roomCode, type: 'next_round' });
  nextRoundGo();
}

function nextRoundGo() {
  showScreen('game');
  document.getElementById('timer').classList.remove('danger');
  beginRoundCountdown();
}

function showGameOver() {
  const iWon = scoreYou > scoreThem;
  document.getElementById('over-emoji').textContent = iWon ? '🏆' : '😭';
  document.getElementById('over-title').textContent = iWon ? 'you win!' : 'you lose';
  document.getElementById('over-title').className = 'result-title ' + (iWon ? 'win' : 'lose');
  document.getElementById('os-you').textContent = scoreYou;
  document.getElementById('os-them').textContent = scoreThem;
  showScreen('over');
}

function rematch() {
  socket.emit('game_event', { code: roomCode, type: 'rematch_request' });
}

function resetGame() {
  scoreYou = 0; scoreThem = 0; round = 1;
  laughTriggered = false;
  updateScores();
  showScreen('game');
  beginRoundCountdown();
}

function updateScores() {
  document.getElementById('score-you').textContent = scoreYou;
  document.getElementById('score-them').textContent = scoreThem;
}

function setStatus(msg, highlight = false) {
  const el = document.getElementById('status-line');
  el.textContent = msg;
  el.className = 'status-line' + (highlight ? ' highlight' : '');
}

function showScreen(id) {
  document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
  document.getElementById('screen-' + id).classList.add('active');
}

function hideOverlay(id) {
  const el = document.getElementById(id);
  el.style.opacity = '0';
  setTimeout(() => el.style.display = 'none', 300);
}

function cleanup() {
  stopTimer();
  laughDetectionActive = false;
  if (localStream) { localStream.getTracks().forEach(t => t.stop()); localStream = null; }
  if (pc) { pc.close(); pc = null; }
  if (faceMesh) { faceMesh.close(); faceMesh = null; }
  roomCode = null; scoreYou = 0; scoreThem = 0; round = 1;
  ['overlay-local', 'overlay-remote'].forEach(id => {
    const el = document.getElementById(id);
    el.style.display = ''; el.style.opacity = '1';
  });
  document.getElementById('overlay-local').textContent = '📷';
  document.getElementById('overlay-remote').textContent = '⏳';
  document.getElementById('video-local').srcObject = null;
  document.getElementById('video-remote').srcObject = null;
  document.getElementById('join-code').value = '';
  document.getElementById('timer').classList.remove('danger');
}

document.addEventListener('keydown', e => {
  if (e.code === 'Space' && document.getElementById('screen-game').classList.contains('active')) {
    e.preventDefault();
    if (!document.getElementById('laugh-btn').disabled) iLaughed();
  }
});