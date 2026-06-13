document.addEventListener('DOMContentLoaded', () => {
  showScreen('lobby');
  setBestOf(5);
  setMaxPlayers(2);
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
const LAUGH_THRESHOLD_OPEN_RATIO    = 0.22;
const LAUGH_THRESHOLD_AUDIO_CONFIRM = 0.6;
const LAUGH_THRESHOLD_AUDIO_ASSIST  = 0.35;

// ─── State ────────────────────────────────────────────────
let localStream;
let roomCode = null;

const socket = io();
const me = () => socket.id;   // always live, never stale

let peers = {};
let playerOrder = [];
let maxPlayers = 2;
let bestOf = 5;
let selectedBestOf = 5;
let selectedMaxPlayers = 2;

let timerInterval = null;
let timerSecs = 0;

let scores = {};
let allTimeScores = {};

let laughedThisRound = new Set();
let roundActive = false;
let audioCtx = null;

let faceMesh = null;
let laughDetectionActive = false;
let laughStartTime = null;
let laughTriggered = false;
let baselineMouthRatio = 0.025;
let calibrating = false;
let calibrationSamples = [];
let detectionCanvas, detectionCtx;
let detectionRunning = false;

// ─── Socket handlers ──────────────────────────────────────
socket.on('room_created', (code) => {
  roomCode = code;
  document.getElementById('room-code-display').textContent = code;
  showScreen('waiting');
  startCamera();
});

socket.on('room_joined', (code) => {
  roomCode = code;
});

socket.on('room_update', ({ players, maxPlayers: mp }) => {
  maxPlayers = mp;
  const el = document.getElementById('waiting-count');
  if (el) el.textContent = players.length + ' / ' + mp + ' players';
});

socket.on('error', (msg) => {
  document.getElementById('lobby-error').textContent = msg;
});

socket.on('game_start', async ({ bestOf: bo, players, maxPlayers: mp }) => {
  bestOf = bo || 5;
  maxPlayers = mp || 2;
  playerOrder = players;

  scores = {};
  allTimeScores = {};
  playerOrder.forEach(id => { scores[id] = 0; allTimeScores[id] = 0; });

  await startCamera();
  buildCamGrid(playerOrder);
  showScreen('game');
  initFaceDetection();
  initAudioDetection();

  // WebRTC mesh: player at index I offers to all players at index < I.
  // Exactly one offerer per pair, no races.
  const myIndex = playerOrder.indexOf(me());
  for (let i = 0; i < myIndex; i++) {
    await offerTo(playerOrder[i]);
  }

  beginPrepPhase();
});

socket.on('game_event', async ({ type, data, fromId }) => {
  switch (type) {

    // ── WebRTC signalling ──
    case 'offer': {
      const pc = getOrCreatePC(fromId);
      await pc.setRemoteDescription(new RTCSessionDescription(data));
      const answer = await pc.createAnswer();
      await pc.setLocalDescription(answer);
      socket.emit('game_event', {
        code: roomCode, type: 'answer',
        data: { sdp: answer, targetId: fromId }
      });
      break;
    }
    case 'answer': {
      const pc = peers[fromId];
      if (pc) await pc.setRemoteDescription(new RTCSessionDescription(data.sdp));
      break;
    }
    case 'ice': {
      const pc = peers[fromId];
      if (pc && data.candidate) pc.addIceCandidate(new RTCIceCandidate(data.candidate)).catch(() => {});
      break;
    }

    // ── Game events ──
    case 'i_laughed':
      handlePlayerLaughed(fromId);
      break;

    // Server tells us how many have voted so far — update button label
    case 'next_round_vote_update': {
      const btn = document.getElementById('next-btn');
      if (btn && btn.disabled) {
        btn.textContent = 'waiting (' + data.votes + '/' + data.needed + ')...';
      }
      break;
    }

    // Server says everyone is ready — all clients begin the round together
    case 'begin_round':
      beginPrepPhase();
      break;

    // Server tells us rematch vote progress
    case 'rematch_vote': {
      const btn = document.getElementById('next-btn');
      if (btn && btn.disabled) {
        btn.textContent = 'waiting (' + data.votes + '/' + data.needed + ')...';
      }
      break;
    }

    case 'rematch_go':
      if (data?.bestOf) bestOf = data.bestOf;
      resetGame();
      break;
  }
});

socket.on('opponent_left', ({ id }) => {
  const slot = document.getElementById('cam-slot-' + id);
  if (slot) slot.remove();
  if (peers[id]) { peers[id].close(); delete peers[id]; }
  playerOrder = playerOrder.filter(p => p !== id);
  alert('A player disconnected.');
  goLobby();
});

// ─── WebRTC helpers ───────────────────────────────────────
function getOrCreatePC(peerId) {
  if (peers[peerId]) return peers[peerId];

  const pc = new RTCPeerConnection(RTC_CONFIG);
  peers[peerId] = pc;

  if (localStream) localStream.getTracks().forEach(t => pc.addTrack(t, localStream));

  pc.ontrack = (e) => {
    const vid = document.getElementById('video-' + peerId);
    if (vid) {
      vid.srcObject = e.streams[0];
      vid.onloadedmetadata = () => hideOverlay('overlay-' + peerId);
      vid.play().catch(() => {});
    }
  };

  pc.onicecandidate = (e) => {
    socket.emit('game_event', {
      code: roomCode, type: 'ice',
      data: { candidate: e.candidate, targetId: peerId }
    });
  };

  return pc;
}

async function offerTo(peerId) {
  const pc = getOrCreatePC(peerId);
  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);
  socket.emit('game_event', {
    code: roomCode, type: 'offer',
    data: { ...offer, targetId: peerId }
  });
}

// ─── Camera ───────────────────────────────────────────────
async function startCamera() {
  if (localStream) return;
  try {
    localStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
    const vid = document.getElementById('video-' + me());
    if (vid) { vid.srcObject = localStream; vid.muted = true; hideOverlay('overlay-' + me()); }
  } catch (e) { console.warn('Camera unavailable:', e); }
}

// ─── Camera grid ──────────────────────────────────────────
function buildCamGrid(players) {
  const cams = document.getElementById('cams');
  cams.innerHTML = '';
  cams.className = 'cams players-' + players.length;

  players.forEach((id, i) => {
    const isMe = id === me();

    const wrap = document.createElement('div');
    wrap.className = 'cam-wrap';
    wrap.id = 'cam-slot-' + id;

    const vid = document.createElement('video');
    vid.id = 'video-' + id;
    vid.autoplay = true;
    vid.playsinline = true;
    if (isMe) { vid.muted = true; if (localStream) vid.srcObject = localStream; }

    const overlay = document.createElement('div');
    overlay.className = 'cam-overlay';
    overlay.id = 'overlay-' + id;
    overlay.textContent = isMe ? '📷' : '⏳';
    if (isMe && localStream) { overlay.style.opacity = '0'; overlay.style.display = 'none'; }

    const label = document.createElement('div');
    label.className = 'cam-label' + (isMe ? ' you' : '');
    label.innerHTML = isMe ? '<span class="live-dot"></span>you' : 'player ' + (i + 1);

    const laughBadge = document.createElement('div');
    laughBadge.className = 'laugh-badge';
    laughBadge.id = 'laugh-badge-' + id;
    laughBadge.textContent = '😂';

    wrap.appendChild(vid);
    wrap.appendChild(overlay);
    wrap.appendChild(label);
    wrap.appendChild(laughBadge);
    cams.appendChild(wrap);
  });
}

// ─── Audio Detection ──────────────────────────────────────
function initAudioDetection() {
  if (!localStream || audioCtx) return;
  try {
    audioCtx = new AudioContext();
    if (audioCtx.state === 'suspended') audioCtx.resume();
    const source = audioCtx.createMediaStreamSource(localStream);
    const analyser = audioCtx.createAnalyser();
    analyser.fftSize = 512;
    source.connect(analyser);

    const dataArray = new Uint8Array(analyser.frequencyBinCount);
    const binHz = audioCtx.sampleRate / analyser.fftSize;
    const laughLow  = Math.floor(300  / binHz);
    const laughHigh = Math.ceil(3000  / binHz);
    const pulseHistory = [];
    let lastEnergyState = false;
    let audioLaughConfidence = 0;
    window._getAudioConfidence = () => audioLaughConfidence;

    const audioLoop = () => {
      if (!audioCtx) return;
      analyser.getByteFrequencyData(dataArray);
      const totalAvg = dataArray.reduce((a, b) => a + b, 0) / dataArray.length;
      if (totalAvg < 20) { audioLaughConfidence = 0; lastEnergyState = false; requestAnimationFrame(audioLoop); return; }

      let bandSum = 0;
      for (let i = laughLow; i <= laughHigh; i++) bandSum += dataArray[i];
      const bandAvg   = bandSum / (laughHigh - laughLow + 1);
      const bandRatio = bandAvg / (totalAvg + 1);
      const now = Date.now();
      const isHigh = bandAvg > 45;
      if (isHigh && !lastEnergyState) pulseHistory.push(now);
      lastEnergyState = isHigh;
      const cutoff = now - 500;
      while (pulseHistory.length && pulseHistory[0] < cutoff) pulseHistory.shift();
      const pulseCount = pulseHistory.length;
      const spectralOk = bandRatio > 0.5;
      const rhythmOk   = pulseCount >= 2 && pulseCount <= 8;
      audioLaughConfidence = (spectralOk ? 0.5 : 0) + (rhythmOk ? 0.5 : 0);

      if (laughDetectionActive && !laughTriggered && !faceMesh) {
        if (audioLaughConfidence >= LAUGH_THRESHOLD_AUDIO_CONFIRM) {
          if (!laughStartTime) laughStartTime = now;
          else if (now - laughStartTime > LAUGH_SUSTAIN_MS) triggerLaugh();
        } else { laughStartTime = null; }
      }
      requestAnimationFrame(audioLoop);
    };
    requestAnimationFrame(audioLoop);
  } catch (e) { console.warn('Audio detection unavailable:', e); }
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
    faceMesh.setOptions({ maxNumFaces: 1, refineLandmarks: true, minDetectionConfidence: 0.5, minTrackingConfidence: 0.5 });
    faceMesh.onResults(onFaceResults);
    faceMesh.initialize().then(() => { console.log('FaceMesh ready'); startDetectionLoop(); });
  } catch (e) { console.warn('FaceMesh failed, manual mode'); faceMesh = null; enableManualButton(); }
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
  const lm = results.multiFaceLandmarks[0];
  const ratio = getLaughScore(lm);
  const threshold = baselineMouthRatio + LAUGH_THRESHOLD_OPEN_RATIO;
  const pct = Math.min((ratio / (threshold * 1.2)) * 100, 100);
  const bar = document.getElementById('mouth-indicator');
  if (bar) { bar.style.width = pct + '%'; bar.style.background = ratio > threshold ? 'var(--danger)' : 'var(--accent)'; }

  if (calibrating) {
    calibrationSamples.push(ratio);
    if (calibrationSamples.length >= CALIBRATION_FRAMES) finishCalibration();
    return;
  }
  if (!laughDetectionActive || laughTriggered) return;

  const faceTriggered = ratio > threshold;
  const audioConf = window._getAudioConfidence ? window._getAudioConfidence() : 0;
  const strongFace = ratio > (baselineMouthRatio + LAUGH_THRESHOLD_OPEN_RATIO * 1.5);
  const combined   = faceTriggered && audioConf >= LAUGH_THRESHOLD_AUDIO_ASSIST;

  if (strongFace || combined) {
    if (!laughStartTime) laughStartTime = Date.now();
    else if (Date.now() - laughStartTime > LAUGH_SUSTAIN_MS) triggerLaugh();
  } else { laughStartTime = null; }
}

function startDetectionLoop() {
  if (detectionRunning) return;
  detectionRunning = true;
  const loop = async () => {
    if (!faceMesh) return;
    const video = document.getElementById('video-' + me());
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
}

function triggerLaugh() {
  if (laughTriggered || !roundActive) return;
  laughTriggered = true;
  laughDetectionActive = false;
  socket.emit('game_event', { code: roomCode, type: 'i_laughed' });
  handlePlayerLaughed(me());
}

function enableManualButton() {
  const btn = document.getElementById('laugh-btn');
  if (btn) btn.style.display = 'block';
  const bar = document.getElementById('detection-bar');
  if (bar) bar.style.display = 'none';
}

// ─── Round logic ──────────────────────────────────────────
function handlePlayerLaughed(id) {
  if (!roundActive) return;
  if (laughedThisRound.has(id)) return;
  laughedThisRound.add(id);

  const badge = document.getElementById('laugh-badge-' + id);
  if (badge) badge.classList.add('visible');

  const activePlayers = playerOrder.filter(p => !laughedThisRound.has(p));

  if (maxPlayers === 2) {
    endRound();
  } else {
    if (activePlayers.length <= 1) endRound();
  }
}

function endRound() {
  if (!roundActive) return;
  roundActive = false;
  laughDetectionActive = false;
  stopTimer();
  document.getElementById('laugh-btn').disabled = true;

  const activePlayers = playerOrder.filter(p => !laughedThisRound.has(p));
  const winnerId = activePlayers.length === 1 ? activePlayers[0] : null;
  if (winnerId) scores[winnerId] = (scores[winnerId] || 0) + 1;
  updateScoreHUD();

  const winsNeeded = Math.ceil(bestOf / 2);
  const matchWinner = playerOrder.find(id => (scores[id] || 0) >= winsNeeded);

  if (matchWinner) {
    playerOrder.forEach(id => { allTimeScores[id] = (allTimeScores[id] || 0) + (scores[id] || 0); });
    showGameOver(matchWinner);
  } else {
    showRoundResult(winnerId);
  }
}

function showRoundResult(winnerId) {
  const iWon = winnerId === me();
  const noWinner = winnerId === null;
  document.getElementById('round-emoji').textContent = noWinner ? '🤝' : iWon ? '🏆' : '😭';
  document.getElementById('round-title').textContent = noWinner ? 'everyone laughed' : iWon ? 'you held it!' : 'you laughed';
  document.getElementById('round-title').className   = 'result-title ' + (iWon ? 'win' : noWinner ? '' : 'lose');
  document.getElementById('round-sub').textContent   = noWinner ? 'no points awarded' : iWon ? 'you win the round' : (winnerId ? 'player ' + (playerOrder.indexOf(winnerId) + 1) + ' wins the round' : '');
  document.getElementById('rs-you').textContent  = scores[me()] || 0;
  document.getElementById('rs-them').textContent = playerOrder.filter(id => id !== me()).map(id => scores[id] || 0).join(' / ');

  const btn = document.getElementById('next-btn');
  btn.disabled = false;
  btn.textContent = 'ready for next round →';
  btn.onclick = requestNextRound;
  showScreen('round');
}

function showGameOver(winnerId) {
  const iWon = winnerId === me();
  document.getElementById('over-emoji').textContent = iWon ? '🏆' : '😭';
  document.getElementById('over-title').textContent = iWon ? 'you win!' : 'you lose';
  document.getElementById('over-title').className   = 'result-title ' + (iWon ? 'win' : 'lose');
  const allTimeMe     = allTimeScores[me()] || 0;
  const allTimeOthers = playerOrder.filter(id => id !== me()).map(id => allTimeScores[id] || 0).join(' / ');
  document.getElementById('os-you').textContent   = allTimeMe;
  document.getElementById('os-them').textContent  = allTimeOthers;
  document.getElementById('over-sub').textContent = 'all time · you ' + allTimeMe + ' vs ' + allTimeOthers;

  const btn = document.getElementById('next-btn');
  btn.disabled = false;
  btn.textContent = 'rematch';
  btn.onclick = requestRematch;
  showScreen('over');
}

// ─── Next round — server-authoritative voting ─────────────
function requestNextRound() {
  const btn = document.getElementById('next-btn');
  btn.disabled = true;
  btn.textContent = 'waiting (1/' + playerOrder.length + ')...';
  // Tell server I'm ready — server counts all votes and fires begin_round for everyone
  socket.emit('game_event', { code: roomCode, type: 'next_round_ready' });
}

function requestRematch() {
  const btn = document.getElementById('next-btn');
  btn.disabled = true;
  btn.textContent = 'waiting (1/' + playerOrder.length + ')...';
  socket.emit('game_event', { code: roomCode, type: 'rematch_request' });
}

async function resetGame() {
  playerOrder.forEach(id => { scores[id] = 0; });
  laughTriggered = false;
  laughedThisRound.clear();
  updateScoreHUD();

  for (const id in peers) { peers[id].close(); }
  peers = {};

  playerOrder.filter(id => id !== me()).forEach(id => {
    const vid = document.getElementById('video-' + id);
    if (vid) vid.srcObject = null;
    const ol = document.getElementById('overlay-' + id);
    if (ol) { ol.style.display = ''; ol.style.opacity = '1'; ol.textContent = '⏳'; }
    const badge = document.getElementById('laugh-badge-' + id);
    if (badge) badge.classList.remove('visible');
  });

  showScreen('game');
  document.getElementById('timer').classList.remove('danger');
  beginPrepPhase();

  await new Promise(res => setTimeout(res, 800));
  const myIndex = playerOrder.indexOf(me());
  for (let i = 0; i < myIndex; i++) await offerTo(playerOrder[i]);
}

// ─── Game flow ────────────────────────────────────────────
function beginPrepPhase() {
  showScreen('game');
  roundActive = false;
  laughDetectionActive = false;
  laughTriggered = false;
  laughStartTime = null;
  laughedThisRound.clear();
  document.getElementById('laugh-btn').disabled = true;

  playerOrder.forEach(id => {
    const badge = document.getElementById('laugh-badge-' + id);
    if (badge) badge.classList.remove('visible');
  });

  stopTimer();
  timerSecs = 0;
  updateTimerDisplay();
  setStatus('', false);

  if (faceMesh) { calibrating = true; calibrationSamples = []; }

  playerOrder.forEach(id => {
    const vid = document.getElementById('video-' + id);
    if (vid) vid.className = 'blurred';
  });

  runCountdown([3, 2, 1, 'GO'], () => {
    playerOrder.forEach(id => {
      const vid = document.getElementById('video-' + id);
      if (vid) vid.className = 'unblurred';
    });
    goRound();
  });
}

function runCountdown(steps, onDone) {
  const existing = document.getElementById('countdown-overlay');
  if (existing) existing.remove();
  const overlay = document.createElement('div');
  overlay.className = 'countdown-overlay';
  overlay.id = 'countdown-overlay';
  document.body.appendChild(overlay);
  let i = 0;
  const showNext = () => {
    if (i >= steps.length) { overlay.remove(); onDone(); return; }
    const val = steps[i++];
    overlay.innerHTML = '';
    const el = document.createElement('div');
    el.className = 'countdown-number' + (val === 'GO' ? ' go' : '');
    el.textContent = val;
    overlay.appendChild(el);
    setTimeout(showNext, 900);
  };
  showNext();
}

function goRound() {
  setStatus("😐  don't laugh.", true);
  roundActive = true;
  laughDetectionActive = true;
  laughTriggered = false;
  laughStartTime = null;
  calibrating = false;
  if (!faceMesh) document.getElementById('laugh-btn').disabled = false;
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
  handlePlayerLaughed(me());
}

function updateScoreHUD() {
  document.getElementById('score-you').textContent  = scores[me()] || 0;
  document.getElementById('score-them').textContent = playerOrder.filter(id => id !== me()).map(id => scores[id] || 0).join('/');
}

// ─── Lobby selectors ──────────────────────────────────────
function setBestOf(n) {
  selectedBestOf = n;
  [1, 3, 5].forEach(v => {
    const btn = document.getElementById('bo-' + v);
    if (btn) btn.className = 'btn' + (v === n ? ' primary' : '');
  });
}

function setMaxPlayers(n) {
  selectedMaxPlayers = n;
  [2, 3, 4].forEach(v => {
    const btn = document.getElementById('mp-' + v);
    if (btn) btn.className = 'btn' + (v === n ? ' primary' : '');
  });
}

// ─── Helpers ──────────────────────────────────────────────
function setStatus(msg, highlight = false) {
  const el = document.getElementById('status-line');
  el.textContent = msg;
  el.className = 'status-line' + (highlight ? ' highlight' : '');
}

function showScreen(id) {
  document.querySelectorAll('.screen').forEach(s => { s.classList.remove('active'); s.style.display = 'none'; });
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
  socket.emit('create_room', { code, bestOf: selectedBestOf, maxPlayers: selectedMaxPlayers });
}

function joinRoom() {
  document.getElementById('lobby-error').textContent = '';
  const code = document.getElementById('join-code').value.trim().toUpperCase();
  if (code.length !== 4) { document.getElementById('lobby-error').textContent = 'enter a 4-letter code'; return; }
  socket.emit('join_room', code);
}

function goLobby()  { cleanup(); showScreen('lobby'); }
function leaveGame() { if (confirm('Leave the game?')) goLobby(); }

function cleanup() {
  stopTimer();
  detectionRunning = false;
  laughDetectionActive = false;
  calibrating = false;
  if (audioCtx) { audioCtx.close(); audioCtx = null; }
  if (localStream) { localStream.getTracks().forEach(t => t.stop()); localStream = null; }
  for (const id in peers) { peers[id].close(); }
  peers = {};
  if (faceMesh) { faceMesh.close(); faceMesh = null; }
  window._getAudioConfidence = null;
  roomCode = null;
  playerOrder = [];
  scores = {};
  allTimeScores = {};
  laughedThisRound.clear();
  document.getElementById('join-code').value = '';
  document.getElementById('timer').classList.remove('danger');
}

document.addEventListener('keydown', e => {
  if (e.code === 'Space' && document.getElementById('screen-game').classList.contains('active')) {
    e.preventDefault();
    if (!document.getElementById('laugh-btn').disabled) iLaughed();
  }
});