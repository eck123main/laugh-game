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

// ─── Detection thresholds ─────────────────────────────────
const LAUGH_SUSTAIN_MS            = 850;   // ms face+audio must agree before trigger
const CALIBRATION_FRAMES          = 90;    // frames to collect during prep phase
const FACE_WEIGHT                 = 0.55;  // face vs audio blend
const AUDIO_WEIGHT                = 0.45;
const COMBINED_TRIGGER_THRESHOLD  = 0.62;  // blended score needed to start sustain timer
const FACE_SOLO_THRESHOLD         = 0.80;  // face alone can trigger if very strong
const AUDIO_SOLO_THRESHOLD        = 0.85;  // audio alone can trigger if extremely strong

// ─── State ────────────────────────────────────────────────
let localStream;
let roomCode = null;

const socket = io();
const me = () => socket.id;

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

// Calibration baselines — mouth, cheek raise delta, eye squint delta
let baselineMouth  = 0.025;
let baselineCheek  = 0;
let baselineEye    = 0;
let calibrating    = false;
let calibrationSamples = [];   // [{mouth, cheek, eye}]

let detectionCanvas, detectionCtx;
let detectionRunning = false;

// Shared audio confidence (written by audio loop, read by face loop)
let _audioLaughConfidence = 0;

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

  const myIndex = playerOrder.indexOf(me());
  for (let i = 0; i < myIndex; i++) {
    await offerTo(playerOrder[i]);
  }

  beginPrepPhase();
});

socket.on('game_event', async ({ type, data, fromId }) => {
  switch (type) {

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

    case 'i_laughed':
      handlePlayerLaughed(fromId);
      break;

    case 'next_round_vote_update': {
      const btn = document.getElementById('next-btn');
      if (btn && btn.disabled) {
        btn.textContent = 'waiting (' + data.votes + '/' + data.needed + ')...';
      }
      break;
    }

    case 'begin_round':
      beginPrepPhase();
      break;

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

// ─── Audio Detection (improved) ───────────────────────────
// Uses mel-filterbank approximation with spectral centroid, rolloff,
// flux, and ZCR — much more discriminative than raw band energy.
function initAudioDetection() {
  if (!localStream || audioCtx) return;
  try {
    audioCtx = new AudioContext();
    if (audioCtx.state === 'suspended') audioCtx.resume();

    const source   = audioCtx.createMediaStreamSource(localStream);
    const analyser = audioCtx.createAnalyser();
    analyser.fftSize = 1024;
    source.connect(analyser);

    const bufLen  = analyser.frequencyBinCount;  // 512
    const dataArr = new Uint8Array(bufLen);
    const binHz   = audioCtx.sampleRate / analyser.fftSize;

    // Mel-spaced filterbank: 12 bands, 300–4000 Hz
    const MEL_BANDS = 12;
    const melLow    = 300;
    const melHigh   = 4000;
    const melMin    = 2595 * Math.log10(1 + melLow  / 700);
    const melMax    = 2595 * Math.log10(1 + melHigh / 700);
    const melPoints = [];
    for (let i = 0; i <= MEL_BANDS + 1; i++) {
      const m  = melMin + (melMax - melMin) * (i / (MEL_BANDS + 1));
      const hz = 700 * (Math.pow(10, m / 2595) - 1);
      melPoints.push(Math.round(hz / binHz));
    }

    let prevBands    = new Array(MEL_BANDS).fill(0);
    let prevFlux     = 0;
    let zcBuffer     = [];   // recent zero-crossing rates
    let energyBuffer = [];   // recent RMS energy values
    let fluxBuffer   = [];
    const BUF_SIZE   = 20;

    const audioLoop = () => {
      if (!audioCtx) return;
      analyser.getByteFrequencyData(dataArr);

      // Overall RMS energy
      let rmsSum = 0;
      for (let i = 0; i < bufLen; i++) rmsSum += (dataArr[i] / 255) ** 2;
      const rms = Math.sqrt(rmsSum / bufLen);

      // Silence gate
      if (rms < 0.02) {
        _audioLaughConfidence = 0;
        prevBands.fill(0);
        requestAnimationFrame(audioLoop);
        return;
      }

      // Mel filterbank energies
      const bands = new Array(MEL_BANDS).fill(0);
      let totalEnergy = 0;
      for (let b = 0; b < MEL_BANDS; b++) {
        const lo = melPoints[b], hi = melPoints[b + 2];
        let sum = 0, cnt = 0;
        for (let k = lo; k <= Math.min(hi, bufLen - 1); k++) {
          sum += dataArr[k];
          cnt++;
        }
        bands[b] = cnt > 0 ? sum / cnt : 0;
        totalEnergy += bands[b];
      }
      if (totalEnergy < 1) { _audioLaughConfidence = 0; requestAnimationFrame(audioLoop); return; }

      // Spectral centroid (normalised to 0–1 over mel range)
      let centNum = 0, centDen = 0;
      for (let b = 0; b < MEL_BANDS; b++) { centNum += b * bands[b]; centDen += bands[b]; }
      const centroid = centDen > 0 ? centNum / (centDen * (MEL_BANDS - 1)) : 0;

      // Spectral rolloff: band index where cumulative energy hits 85%
      let cum = 0, rolloff = MEL_BANDS - 1;
      for (let b = 0; b < MEL_BANDS; b++) {
        cum += bands[b];
        if (cum / totalEnergy >= 0.85) { rolloff = b; break; }
      }
      const rolloffNorm = rolloff / (MEL_BANDS - 1);

      // Spectral flux (frame-to-frame change)
      let flux = 0;
      for (let b = 0; b < MEL_BANDS; b++) flux += Math.abs(bands[b] - prevBands[b]);
      flux /= (MEL_BANDS * 255);
      prevBands = [...bands];

      // Zero-crossing rate proxy: high-freq band ratio (bands 8–11 vs all)
      let hfSum = 0;
      for (let b = 8; b < MEL_BANDS; b++) hfSum += bands[b];
      const zcr = totalEnergy > 0 ? hfSum / totalEnergy : 0;

      // Rolling buffers
      const push = (buf, val) => { buf.push(val); if (buf.length > BUF_SIZE) buf.shift(); };
      push(energyBuffer, rms);
      push(fluxBuffer,   flux);
      push(zcBuffer,     zcr);

      const avgFlux  = fluxBuffer.reduce((a, b) => a + b, 0) / fluxBuffer.length;
      const avgZcr   = zcBuffer.reduce((a, b) => a + b, 0) / zcBuffer.length;

      // Laugh audio signature:
      // - centroid 0.35–0.75  (voiced mid-freq, not low rumble or hiss)
      // - rolloff   0.5–0.9   (energy spread across mid–high bands)
      // - flux > 0.015        (irregular, rhythmic bursts — not sustained tone)
      // - zcr 0.2–0.6         (voiced but not pure noise)
      const centroidScore = (centroid > 0.30 && centroid < 0.80) ? 1 : 0;
      const rolloffScore  = (rolloffNorm > 0.45 && rolloffNorm < 0.92) ? 1 : 0;
      const fluxScore     = Math.min(avgFlux / 0.04, 1);          // 0–1
      const zcrScore      = (avgZcr > 0.15 && avgZcr < 0.65) ? 1 : 0;
      const energyScore   = Math.min(rms / 0.12, 1);              // 0–1, rewards louder

      // Weighted combination — flux and energy are continuous, rest are binary gates
      _audioLaughConfidence = Math.min(
        (centroidScore * 0.20) +
        (rolloffScore  * 0.20) +
        (fluxScore     * 0.25) +
        (zcrScore      * 0.15) +
        (energyScore   * 0.20),
        1
      );

      // Audio-solo mode (no FaceMesh): trigger if extremely confident
      if (laughDetectionActive && !laughTriggered && !faceMesh) {
        if (_audioLaughConfidence >= AUDIO_SOLO_THRESHOLD) {
          if (!laughStartTime) laughStartTime = Date.now();
          else if (Date.now() - laughStartTime > LAUGH_SUSTAIN_MS) triggerLaugh();
        } else {
          laughStartTime = null;
        }
      }

      requestAnimationFrame(audioLoop);
    };
    requestAnimationFrame(audioLoop);
  } catch (e) { console.warn('Audio detection unavailable:', e); }
}

// ─── Face Detection (improved) ────────────────────────────
// Tracks mouth open ratio, cheek raise, and eye squint.
// All three signals are calibrated individually during prep phase.

// Landmark indices (MediaPipe Face Mesh)
const LM_UPPER_LIP    = 13;
const LM_LOWER_LIP    = 14;
const LM_LEFT_CORNER  = 78;
const LM_RIGHT_CORNER = 308;
const LM_LEFT_CHEEK   = 50;   // left cheek apex
const LM_RIGHT_CHEEK  = 280;  // right cheek apex
const LM_LEFT_EYE_TOP = 159;  // left upper eyelid
const LM_LEFT_EYE_BOT = 145;  // left lower eyelid
const LM_RIGHT_EYE_TOP= 386;
const LM_RIGHT_EYE_BOT= 374;
// Nose tip as a stable vertical reference
const LM_NOSE_TIP     = 1;

function getFaceFeatures(lm) {
  // Mouth openness (height / width ratio)
  const mouthH = Math.abs(lm[LM_LOWER_LIP].y - lm[LM_UPPER_LIP].y);
  const mouthW = Math.abs(lm[LM_RIGHT_CORNER].x - lm[LM_LEFT_CORNER].x);
  const mouth  = mouthW > 0.001 ? mouthH / mouthW : 0;

  // Cheek raise: how far cheek apexes are above mouth corners.
  // Laughter pulls cheeks up, reducing this delta.
  const cornerMidY = (lm[LM_LEFT_CORNER].y + lm[LM_RIGHT_CORNER].y) / 2;
  const cheekMidY  = (lm[LM_LEFT_CHEEK].y  + lm[LM_RIGHT_CHEEK].y)  / 2;
  const cheek      = cornerMidY - cheekMidY;  // positive = cheeks above corners

  // Eye squint: eyelid aperture (vertical gap / face height proxy)
  // Genuine laughter narrows the eyes (Duchenne marker)
  const leftEyeH  = Math.abs(lm[LM_LEFT_EYE_TOP].y  - lm[LM_LEFT_EYE_BOT].y);
  const rightEyeH = Math.abs(lm[LM_RIGHT_EYE_TOP].y - lm[LM_RIGHT_EYE_BOT].y);
  const eyeApert  = (leftEyeH + rightEyeH) / 2;  // smaller = more squinted

  return { mouth, cheek, eyeApert };
}

function initFaceDetection() {
  detectionCanvas       = document.createElement('canvas');
  detectionCanvas.width  = 320;
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
    console.warn('FaceMesh failed, falling back to audio-only');
    faceMesh = null;
    enableManualButton();
  }
}

function onFaceResults(results) {
  if (!results.multiFaceLandmarks || !results.multiFaceLandmarks.length) return;
  const lm = results.multiFaceLandmarks[0];
  const { mouth, cheek, eyeApert } = getFaceFeatures(lm);

  // Update mouth indicator bar (unchanged UX)
  const mouthThresh = baselineMouth + 0.22;
  const pct = Math.min((mouth / (mouthThresh * 1.2)) * 100, 100);
  const bar = document.getElementById('mouth-indicator');
  if (bar) {
    bar.style.width      = pct + '%';
    bar.style.background = mouth > mouthThresh ? 'var(--danger)' : 'var(--accent)';
  }

  // Calibration: collect samples
  if (calibrating) {
    calibrationSamples.push({ mouth, cheek, eyeApert });
    if (calibrationSamples.length >= CALIBRATION_FRAMES) finishCalibration();
    return;
  }

  if (!laughDetectionActive || laughTriggered) return;

  // Per-feature laugh scores (0–1)
  // Mouth: how far above baseline + fixed threshold
  const mouthDelta  = mouth - baselineMouth;
  const mouthScore  = Math.min(Math.max(mouthDelta / 0.22, 0), 1);

  // Cheek raise: cheek should be higher than baseline
  const cheekDelta  = cheek - baselineCheek;
  const cheekScore  = Math.min(Math.max(cheekDelta / 0.03, 0), 1);

  // Eye squint: aperture should be SMALLER than baseline (eyes narrowing)
  const eyeDelta    = baselineEye - eyeApert;
  const eyeScore    = Math.min(Math.max(eyeDelta / 0.015, 0), 1);

  // Combined face score — mouth is primary, cheek and eye are supporting
  const faceScore   = Math.min(
    (mouthScore * 0.60) + (cheekScore * 0.25) + (eyeScore * 0.15),
    1
  );

  // Blended confidence
  const blended = (faceScore * FACE_WEIGHT) + (_audioLaughConfidence * AUDIO_WEIGHT);

  // Two trigger paths:
  // 1. Blended signal exceeds threshold (normal case)
  // 2. Face alone is extremely strong (e.g. audio muted)
  const shouldAccumulate = blended >= COMBINED_TRIGGER_THRESHOLD || faceScore >= FACE_SOLO_THRESHOLD;

  if (shouldAccumulate) {
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
  const n = calibrationSamples.length;
  if (n === 0) return;

  // Use mean of the middle 60% of samples (trim outliers)
  const trimmed = (arr) => {
    const sorted = [...arr].sort((a, b) => a - b);
    const lo = Math.floor(n * 0.2), hi = Math.ceil(n * 0.8);
    const slice = sorted.slice(lo, hi);
    return slice.reduce((a, b) => a + b, 0) / slice.length;
  };

  baselineMouth = trimmed(calibrationSamples.map(s => s.mouth));
  baselineCheek = trimmed(calibrationSamples.map(s => s.cheek));
  baselineEye   = trimmed(calibrationSamples.map(s => s.eyeApert));

  console.log('Calibrated — mouth:', baselineMouth.toFixed(4),
              'cheek:', baselineCheek.toFixed(4),
              'eye:', baselineEye.toFixed(4));
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

  const winsNeeded  = Math.ceil(bestOf / 2);
  const matchWinner = playerOrder.find(id => (scores[id] || 0) >= winsNeeded);

  if (matchWinner) {
    playerOrder.forEach(id => { allTimeScores[id] = (allTimeScores[id] || 0) + (scores[id] || 0); });
    showGameOver(matchWinner);
  } else {
    showRoundResult(winnerId);
  }
}

function showRoundResult(winnerId) {
  const iWon     = winnerId === me();
  const noWinner = winnerId === null;
  document.getElementById('round-emoji').textContent = noWinner ? '🤝' : iWon ? '🏆' : '😭';
  document.getElementById('round-title').textContent = noWinner ? 'everyone laughed' : iWon ? 'you held it!' : 'you laughed';
  document.getElementById('round-title').className   = 'result-title ' + (iWon ? 'win' : noWinner ? '' : 'lose');
  document.getElementById('round-sub').textContent   = noWinner ? 'no points awarded'
    : iWon ? 'you win the round'
    : winnerId ? 'player ' + (playerOrder.indexOf(winnerId) + 1) + ' wins the round' : '';
  document.getElementById('rs-you').textContent  = scores[me()] || 0;
  document.getElementById('rs-them').textContent = playerOrder.filter(id => id !== me()).map(id => scores[id] || 0).join(' / ');

  const btn = document.getElementById('next-btn');
  btn.disabled    = false;
  btn.textContent = 'ready for next round →';
  btn.onclick     = requestNextRound;
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
  btn.disabled    = false;
  btn.textContent = 'rematch';
  btn.onclick     = requestRematch;
  showScreen('over');
}

// ─── Next round — server-authoritative voting ─────────────
function requestNextRound() {
  const btn       = document.getElementById('next-btn');
  btn.disabled    = true;
  btn.textContent = 'waiting (1/' + playerOrder.length + ')...';
  socket.emit('game_event', { code: roomCode, type: 'next_round_ready' });
}

function requestRematch() {
  const btn       = document.getElementById('next-btn');
  btn.disabled    = true;
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
  roundActive          = false;
  laughDetectionActive = false;
  laughTriggered       = false;
  laughStartTime       = null;
  _audioLaughConfidence = 0;
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

  // Reset calibration for fresh baseline each round
  if (faceMesh) {
    calibrating        = true;
    calibrationSamples = [];
  }

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
  roundActive          = true;
  laughDetectionActive = true;
  laughTriggered       = false;
  laughStartTime       = null;
  calibrating          = false;
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
  detectionRunning      = false;
  laughDetectionActive  = false;
  calibrating           = false;
  _audioLaughConfidence = 0;
  if (audioCtx) { audioCtx.close(); audioCtx = null; }
  if (localStream) { localStream.getTracks().forEach(t => t.stop()); localStream = null; }
  for (const id in peers) { peers[id].close(); }
  peers = {};
  if (faceMesh) { faceMesh.close(); faceMesh = null; }
  roomCode    = null;
  playerOrder = [];
  scores      = {};
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