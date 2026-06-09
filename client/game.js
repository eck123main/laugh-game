// ─── Config ───────────────────────────────────────────────
const RTC_CONFIG = { iceServers: [{ urls: 'stun:stun.l.google.com:19302' }] };
const TOTAL_ROUNDS = 5;
const PREP_SECONDS = 10;
const LAUGH_SUSTAIN_MS = 1000;
const CALIBRATION_FRAMES = 80;

// ─── State ────────────────────────────────────────────────
let socket, pc, localStream;
let roomCode = null;
let isHost = false;
let timerInterval = null;
let timerSecs = 0;
let scoreYou = 0, scoreThem = 0, round = 1;
let roundActive = false;

// Detection state
let faceMesh = null;
let laughDetectionActive = false;
let laughStartTime = null;
let laughTriggered = false;
let baselineMouthRatio = 0.025;
let sensitivityMultiplier = 1.0;
let calibrating = false;
let calibrationSamples = [];
let detectionCanvas, detectionCtx;
let detectionRunning = false;

// ─── Socket ───────────────────────────────────────────────
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
  initFaceDetection();
  if (isHost) {
    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);
    socket.emit('game_event', { code: roomCode, type: 'offer', data: offer });
    beginPrepPhase();
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
      initFaceDetection();
      beginPrepPhase();
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
  if (localStream) localStream.getTracks().forEach(t => pc.addTrack(t, localStream));
  pc.ontrack = (e) => {
    const v = document.getElementById('video-remote');
    v.srcObject = e.streams[0];
    v.onloadedmetadata = () => hideOverlay('overlay-remote');
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
    await new Promise(res => { vid.onloadedmetadata = res; });
    hideOverlay('overlay-local');
  } catch (e) {
    console.warn('Camera unavailable:', e);
  }
}

// ─── Face Detection ───────────────────────────────────────
function initFaceDetection() {
  detectionCanvas = document.createElement('canvas');
  detectionCanvas.width = 320;
  detectionCanvas.height = 240;
  detectionCtx = detectionCanvas.getContext('2d');

  try {
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
    faceMesh.initialize().then(() => {
      console.log('FaceMesh ready');
      startDetectionLoop();
    });
  } catch(e) {
    console.warn('FaceMesh failed, using manual mode');
    faceMesh = null;
    enableManualButton();
  }
}

// Mouth landmark indices
const UPPER_LIP = 13, LOWER_LIP = 14, LEFT_CORNER = 78, RIGHT_CORNER = 308;

function getMouthRatio(lm) {
  const h = Math.abs(lm[LOWER_LIP].y - lm[UPPER_LIP].y);
  const w = Math.abs(lm[RIGHT_CORNER].x - lm[LEFT_CORNER].x);
  return w < 0.001 ? 0 : h / w;
}

function onFaceResults(results) {
  if (!results.multiFaceLandmarks || !results.multiFaceLandmarks.length) return;
  const ratio = getMouthRatio(results.multiFaceLandmarks[0]);

  // Update bar
const threshold = baselineMouthRatio + (0.15 * sensitivityMultiplier);

  const pct = Math.min((ratio / (threshold * 1.5)) * 100, 100);
  const bar = document.getElementById('mouth-indicator');
  if (bar) {
    bar.style.width = pct + '%';
    bar.style.background = ratio > threshold ? 'var(--danger)' : 'var(--accent)';
  }

  // Silent calibration during prep
  if (calibrating) {
    calibrationSamples.push(ratio);
    if (calibrationSamples.length >= CALIBRATION_FRAMES) finishCalibration();
    return;
  }

  // Laugh detection during round
  if (!laughDetectionActive || laughTriggered) return;
  if (ratio > threshold) {
    if (!laughStartTime) laughStartTime = Date.now();
    else if (Date.now() - laughStartTime > LAUGH_SUSTAIN_MS) triggerLaugh();
  } else {
    laughStartTime = null;
  }
}

function startDetectionLoop() {
  if (detectionRunning) return;
  detectionRunning = true;
  const loop = async () => {
    if (!faceMesh) return;
    const video = document.getElementById('video-local');
    if (video && video.readyState >= 2) {
      detectionCtx.drawImage(video, 0, 0, 320, 240);
      await faceMesh.send({ image: detectionCanvas });
    }
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
}

function finishCalibration() {
  calibrating = false;
  const avg = calibrationSamples.reduce((a, b) => a + b, 0) / calibrationSamples.length;
  baselineMouthRatio = avg;
  console.log('Baseline:', baselineMouthRatio.toFixed(4));
}

function triggerLaugh() {
  if (laughTriggered || !roundActive) return;
  laughTriggered = true;
  laughDetectionActive = false;
  socket.emit('game_event', { code: roomCode, type: 'i_laughed' });
  endRound(false);
}

function enableManualButton() {
  const btn = document.getElementById('laugh-btn');
  if (btn) btn.style.display = 'block';
  const bar = document.getElementById('detection-bar');
  if (bar) bar.style.display = 'none';
}

function setSensitivity(val) {
  sensitivityMultiplier = parseFloat(val);
}

// ─── Game Flow ────────────────────────────────────────────

// Phase 1: 10 second prep — cameras on, stare at each other
// Calibration happens silently in background
function beginPrepPhase() {
  roundActive = false;
  laughDetectionActive = false;
  laughTriggered = false;
  laughStartTime = null;
  document.getElementById('laugh-btn').disabled = true;
  document.getElementById('round-num').textContent = round;
  stopTimer();
  timerSecs = 0;
  updateTimerDisplay();

  // Start silent calibration
  if (faceMesh) {
    calibrating = true;
    calibrationSamples = [];
  }

  // Show prep countdown
  let prepLeft = PREP_SECONDS;
  setStatus(`round ${round} — stare down begins in ${prepLeft}...`);

  const prepTimer = setInterval(() => {
    prepLeft--;
    if (prepLeft > 3) {
      setStatus(`round ${round} — stare down begins in ${prepLeft}...`);
    } else if (prepLeft > 0) {
      setStatus(prepLeft + '...', true);
    } else {
      clearInterval(prepTimer);
      beginCountdown();
    }
  }, 1000);
}

// Phase 2: 3-2-1 countdown then GO
function beginCountdown() {
  let count = 3;
  setStatus(count + '...', true);
  const cd = setInterval(() => {
    count--;
    if (count > 0) {
      setStatus(count + '...', true);
    } else {
      clearInterval(cd);
      goRound();
    }
  }, 1000);
}

// Phase 3: round is live
function goRound() {
  setStatus('😐  don\'t laugh.', true);
  roundActive = true;
  laughDetectionActive = true;
  laughTriggered = false;
  laughStartTime = null;
  calibrating = false; // stop calibrating, start detecting

  // Manual button enabled as fallback
  if (!faceMesh) {
    document.getElementById('laugh-btn').disabled = false;
  }

  startTimer();
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
  beginPrepPhase();
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
  beginPrepPhase();
}

// ─── Helpers ──────────────────────────────────────────────
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
  document.querySelectorAll('.screen').forEach(s => {
    s.classList.remove('active');
    s.style.display = 'none';
  });
  const el = document.getElementById('screen-' + id);
  el.style.display = '';
  el.classList.add('active');
}

function hideOverlay(id) {
  const el = document.getElementById(id);
  if (!el) return;
  el.style.opacity = '0';
  setTimeout(() => el.style.display = 'none', 300);
}

// ─── Lobby ────────────────────────────────────────────────
function createRoom() {
  document.getElementById('lobby-error').textContent = '';
  const code = Math.random().toString(36).substring(2, 6).toUpperCase();
  socket.emit('create_room', code);
}

function joinRoom() {
  document.getElementById('lobby-error').textContent = '';
  const code = document.getElementById('join-code').value.trim().toUpperCase();
  if (code.length !== 4) { document.getElementById('lobby-error').textContent = 'enter a 4-letter code'; return; }
  socket.emit('join_room', code);
}

function goLobby() { cleanup(); showScreen('lobby'); }
function leaveGame() { if (confirm('Leave the game?')) goLobby(); }

function cleanup() {
  stopTimer();
  detectionRunning = false;
  laughDetectionActive = false;
  calibrating = false;
  if (localStream) { localStream.getTracks().forEach(t => t.stop()); localStream = null; }
  if (pc) { pc.close(); pc = null; }
  if (faceMesh) { faceMesh.close(); faceMesh = null; }
  roomCode = null; scoreYou = 0; scoreThem = 0; round = 1;
  ['overlay-local','overlay-remote'].forEach(id => {
    const el = document.getElementById(id);
    if (el) { el.style.display = ''; el.style.opacity = '1'; }
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