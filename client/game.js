document.addEventListener('DOMContentLoaded', () => {
  showScreen('lobby');
});

const RTC_CONFIG = {
  iceServers: [
    { urls: 'stun:oz-turn-1.xirsys.com' },
    {
      username: '2-TFMaK-hIvZBmLtI4uGRRUoE_xWlyEC_unifkTZ197vLn2zBfWoTPfFuYaWXTjXAAAAAGonjuxyZXRvcm5pdHk=',
      credential: '3176d1aa-63b7-11f1-bb1d-0242ac120004',
      urls: [
        'turn:oz-turn-1.xirsys.com:80?transport=udp',
        'turn:oz-turn-1.xirsys.com:3478?transport=udp',
        'turn:oz-turn-1.xirsys.com:80?transport=tcp',
        'turn:oz-turn-1.xirsys.com:3478?transport=tcp',
        'turns:oz-turn-1.xirsys.com:443?transport=tcp',
        'turns:oz-turn-1.xirsys.com:5349?transport=tcp'
      ]
    }
  ]
};

const LAUGH_SUSTAIN_MS = 1000;
const CALIBRATION_FRAMES = 80;

// ─── State ────────────────────────────────────────────────
let socket, pc, localStream;
let roomCode = null;
let isHost = false;
let timerInterval = null;
let timerSecs = 0;
let scoreYou = 0, scoreThem = 0;
let roundActive = false;
let iRequestedRematch = false;

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
    beginPrepPhase();
  } else {
    socket.emit('game_event', { code: roomCode, type: 'guest_ready' });
    beginPrepPhase();
  }
});

socket.on('game_event', async ({ type, data }) => {
  switch (type) {
    case 'guest_ready': {
      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);
      socket.emit('game_event', { code: roomCode, type: 'offer', data: offer });
      break;
    }
    case 'offer': {
      await pc.setRemoteDescription(new RTCSessionDescription(data));
      const answer = await pc.createAnswer();
      await pc.setLocalDescription(answer);
      socket.emit('game_event', { code: roomCode, type: 'answer', data: answer });
      break;
    }
    case 'answer':
      await pc.setRemoteDescription(new RTCSessionDescription(data));
      break;
    case 'ice':
      if (data) await pc.addIceCandidate(new RTCIceCandidate(data));
      break;
    case 'i_laughed':
      endRound(true);
      break;
    case 'rematch_request': {
      const btn = document.getElementById('next-btn');
      if (iRequestedRematch) {
        // Both sides ready — I already clicked, now they clicked
        iRequestedRematch = false;
        resetGame();
      } else {
        // They clicked first, update button so I can confirm
        btn.disabled = false;
        btn.textContent = 'opponent wants a rematch — play again?';
        btn.onclick = requestRematch;
      }
      break;
    }
    case 'rematch_go':
      resetGame();
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

  if (localStream) {
    localStream.getTracks().forEach(t => pc.addTrack(t, localStream));
  } else {
    console.error('No local stream when setting up peer connection!');
  }

  pc.ontrack = (e) => {
    const v = document.getElementById('video-remote');
    v.srcObject = e.streams[0];
    v.onloadedmetadata = () => hideOverlay('overlay-remote');
    v.play().catch(err => console.warn('Video play failed:', err));
  };

  pc.onicecandidate = (e) => {
    socket.emit('game_event', { code: roomCode, type: 'ice', data: e.candidate });
  };

  pc.oniceconnectionstatechange = () => console.log('ICE state:', pc.iceConnectionState);
  pc.onconnectionstatechange = () => console.log('Connection state:', pc.connectionState);
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
  } catch (e) {
    console.warn('FaceMesh failed, using manual mode');
    faceMesh = null;
    enableManualButton();
  }
}

const UPPER_LIP = 13, LOWER_LIP = 14, LEFT_CORNER = 78, RIGHT_CORNER = 308;

function getLaughScore(lm) {
  const openH = Math.abs(lm[LOWER_LIP].y - lm[UPPER_LIP].y);
  const openW = Math.abs(lm[RIGHT_CORNER].x - lm[LEFT_CORNER].x);
  if (openW < 0.001) return 0;
  const ratio = openH / openW;
  const cornerMidY = (lm[LEFT_CORNER].y + lm[RIGHT_CORNER].y) / 2;
  const jawDrop = lm[LOWER_LIP].y - cornerMidY;
  if (jawDrop < 0.01) return 0;
  return ratio;
}

function onFaceResults(results) {
  if (!results.multiFaceLandmarks || !results.multiFaceLandmarks.length) return;
  const ratio = getLaughScore(results.multiFaceLandmarks[0]);
  const threshold = baselineMouthRatio + (0.15 * sensitivityMultiplier);

  const pct = Math.min((ratio / (threshold * 1.5)) * 100, 100);
  const bar = document.getElementById('mouth-indicator');
  if (bar) {
    bar.style.width = pct + '%';
    bar.style.background = ratio > threshold ? 'var(--danger)' : 'var(--accent)';
  }

  if (calibrating) {
    calibrationSamples.push(ratio);
    if (calibrationSamples.length >= CALIBRATION_FRAMES) finishCalibration();
    return;
  }

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
function beginPrepPhase() {
  roundActive = false;
  laughDetectionActive = false;
  laughTriggered = false;
  laughStartTime = null;
  document.getElementById('laugh-btn').disabled = true;
  stopTimer();
  timerSecs = 0;
  updateTimerDisplay();

  if (faceMesh) {
    calibrating = true;
    calibrationSamples = [];
  }

  let count = 5;
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

function goRound() {
  setStatus("😐  don't laugh.", true);
  roundActive = true;
  laughDetectionActive = true;
  laughTriggered = false;
  laughStartTime = null;
  calibrating = false;
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
  document.getElementById('round-sub').textContent = iWon ? 'you held it together' : 'opponent wins';
  document.getElementById('rs-you').textContent = scoreYou;
  document.getElementById('rs-them').textContent = scoreThem;

  iRequestedRematch = false;
  const btn = document.getElementById('next-btn');
  btn.disabled = false;
  btn.textContent = 'rematch?';
  btn.onclick = requestRematch;
  showScreen('round');
}

function requestRematch() {
  iRequestedRematch = true;
  const btn = document.getElementById('next-btn');
  btn.disabled = true;
  btn.textContent = 'waiting for opponent...';
  socket.emit('game_event', { code: roomCode, type: 'rematch_request' });
}

async function resetGame() {
  iRequestedRematch = false;
  scoreYou = 0; scoreThem = 0;
  laughTriggered = false;
  updateScores();

  if (pc) { pc.close(); pc = null; }

  const rv = document.getElementById('video-remote');
  rv.srcObject = null;
  const ol = document.getElementById('overlay-remote');
  if (ol) { ol.style.display = ''; ol.style.opacity = '1'; ol.textContent = '⏳'; }

  showScreen('game');
  document.getElementById('timer').classList.remove('danger');
  beginPrepPhase();

  await new Promise(res => setTimeout(res, 800));
  await setupPeerConnection();

  if (!isHost) {
    socket.emit('game_event', { code: roomCode, type: 'guest_ready' });
  }
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
  iRequestedRematch = false;
  if (localStream) { localStream.getTracks().forEach(t => t.stop()); localStream = null; }
  if (pc) { pc.close(); pc = null; }
  if (faceMesh) { faceMesh.close(); faceMesh = null; }
  roomCode = null; scoreYou = 0; scoreThem = 0;
  ['overlay-local', 'overlay-remote'].forEach(id => {
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