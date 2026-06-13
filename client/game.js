document.addEventListener('DOMContentLoaded', () => {
  showScreen('lobby');
  setBestOf(5);
  setMaxPlayers(2);
  setGameMode('standard');
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
const LAUGH_SUSTAIN_MS            = 850;
const CALIBRATION_FRAMES          = 90;
const FACE_WEIGHT                 = 0.55;
const AUDIO_WEIGHT                = 0.45;
const COMBINED_TRIGGER_THRESHOLD  = 0.62;
const FACE_SOLO_THRESHOLD         = 0.80;
const AUDIO_SOLO_THRESHOLD        = 0.75;  // Lowered: ResNet is more accurate than heuristic

// ─── State ────────────────────────────────────────────────
let localStream;
let roomCode = null;

const socket = io();
const me = () => socket.id;

let peers = {};
let playerOrder    = [];
let maxPlayers     = 2;
let bestOf         = 5;
let selectedBestOf = 5;
let selectedMaxPlayers = 2;
let selectedGameMode   = 'standard';
let gameMode           = 'standard';

// Elimination state
let activePlayers     = [];
let eliminatedPlayers = [];
let eliminationFinals = false;

let timerInterval = null;
let timerSecs     = 0;

let scores        = {};
let allTimeScores = {};

let laughedThisRound     = new Set();
let roundActive          = false;
let audioCtx             = null;
let scriptProcessor      = null;   // kept so we can disconnect on cleanup

let faceMesh             = null;
let laughDetectionActive = false;
let laughStartTime       = null;
let laughTriggered       = false;

let baselineMouth  = 0.025;
let baselineCheek  = 0;
let baselineEye    = 0;
let calibrating    = false;
let calibrationSamples = [];

let detectionCanvas, detectionCtx;
let detectionRunning = false;

let _audioLaughConfidence = 0;

// ─── ResNet ONNX model state ──────────────────────────────
let onnxSession    = null;   // ort.InferenceSession once loaded
let onnxReady      = false;  // true after model loads successfully
let onnxRingBuffer = null;   // Float32Array ring buffer at 8 kHz
let onnxWritePtr   = 0;
let onnxFramesSinceInference = 0;

const ONNX_SAMPLE_RATE = 8000;
const ONNX_WINDOW_SIZE = 2680;   // 1-second window
const ONNX_HOP_SIZE    = 800;    // run inference every 100 ms
const ONNX_MODEL_PATH  = '/models/laughter_resnet.onnx';

// ─── Mel spectrogram constants (match Gillick et al.) ─────
const MEL_N_BINS   = 128;
const MEL_WIN_SAMP = 200;   // 25 ms at 8 kHz
const MEL_HOP_SAMP = 80;    // 10 ms at 8 kHz
const MEL_F_MIN    = 60;
const MEL_F_MAX    = 3800;

// Pre-compute Hann window
const HANN = new Float32Array(MEL_WIN_SAMP);
for (let i = 0; i < MEL_WIN_SAMP; i++) {
  HANN[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / (MEL_WIN_SAMP - 1)));
}

// Pre-compute mel filterbank (MEL_N_BINS filters × (MEL_WIN_SAMP/2+1) FFT bins)
const MEL_FILTERS = buildMelFilterbank(MEL_N_BINS, MEL_WIN_SAMP, ONNX_SAMPLE_RATE, MEL_F_MIN, MEL_F_MAX);

function hzToMel(hz) { return 2595 * Math.log10(1 + hz / 700); }
function melToHz(m)  { return 700 * (Math.pow(10, m / 2595) - 1); }

function buildMelFilterbank(nFilters, winSamp, sr, fMin, fMax) {
  const nFft  = winSamp;
  const nBins = Math.floor(nFft / 2) + 1;
  const melMin = hzToMel(fMin);
  const melMax = hzToMel(fMax);

  // nFilters + 2 evenly-spaced mel points → Hz → FFT bin indices
  const melPoints = new Float32Array(nFilters + 2);
  for (let i = 0; i < nFilters + 2; i++) {
    melPoints[i] = melToHz(melMin + (melMax - melMin) * i / (nFilters + 1));
  }
  const binPoints = melPoints.map(hz => Math.floor((nFft + 1) * hz / sr));

  // Build filter matrix as flat Float32Array[nFilters * nBins]
  const filters = new Float32Array(nFilters * nBins);
  for (let m = 1; m <= nFilters; m++) {
    const lo  = binPoints[m - 1];
    const ctr = binPoints[m];
    const hi  = binPoints[m + 1];
    for (let k = lo; k < ctr && k < nBins; k++) {
      filters[(m - 1) * nBins + k] = (k - lo) / Math.max(ctr - lo, 1);
    }
    for (let k = ctr; k <= hi && k < nBins; k++) {
      filters[(m - 1) * nBins + k] = (hi - k) / Math.max(hi - ctr, 1);
    }
  }
  return { filters, nBins, nFilters };
}

// Real-valued FFT magnitude via DFT (good enough for 200-sample windows)
function rfftMag(frame) {
  const N    = frame.length;
  const nOut = Math.floor(N / 2) + 1;
  const mag  = new Float32Array(nOut);
  for (let k = 0; k < nOut; k++) {
    let re = 0, im = 0;
    for (let n = 0; n < N; n++) {
      const angle = -2 * Math.PI * k * n / N;
      re += frame[n] * Math.cos(angle);
      im += frame[n] * Math.sin(angle);
    }
    mag[k] = Math.sqrt(re * re + im * im);
  }
  return mag;
}

// Compute log mel spectrogram from a 1-second mono float32 clip at ONNX_SAMPLE_RATE
// Returns Float32Array of shape [MEL_N_BINS × nFrames], row-major (bins fastest)
function computeLogMelSpectrogram(audio) {
  const nFrames = Math.floor((audio.length - MEL_WIN_SAMP) / MEL_HOP_SAMP) + 1;
  const { filters, nBins, nFilters } = MEL_FILTERS;
  // Output: [nFilters × nFrames], stored as [frame * nFilters + bin]
  const spec = new Float32Array(nFilters * nFrames);

  for (let f = 0; f < nFrames; f++) {
    const start = f * MEL_HOP_SAMP;
    // Windowed frame
    const frame = new Float32Array(MEL_WIN_SAMP);
    for (let i = 0; i < MEL_WIN_SAMP; i++) frame[i] = audio[start + i] * HANN[i];
    // FFT magnitudes
    const mag = rfftMag(frame);
    // Apply mel filterbank
    for (let m = 0; m < nFilters; m++) {
      let energy = 0;
      const offset = m * nBins;
      for (let k = 0; k < nBins; k++) energy += filters[offset + k] * mag[k];
      spec[f * nFilters + m] = Math.log(Math.max(energy, 1e-9));
    }
  }

  // Transpose to [nFilters × nFrames] (bins × time — what ResNet-18 expects as channels × H × W)
  // Here we return it as a flat [1 × 1 × nFilters × nFrames] tensor buffer
  const transposed = new Float32Array(nFilters * nFrames);
  for (let f = 0; f < nFrames; f++) {
    for (let m = 0; m < nFilters; m++) {
      transposed[m * nFrames + f] = spec[f * nFilters + m];
    }
  }
  return { data: transposed, nFilters, nFrames };
}

// ─── Load ONNX model ──────────────────────────────────────
async function loadOnnxModel() {
  if (typeof ort === 'undefined') {
    console.warn('onnxruntime-web not loaded — falling back to heuristic audio');
    return false;
  }
  try {
    onnxSession    = await ort.InferenceSession.create(ONNX_MODEL_PATH);
    onnxReady      = true;
    onnxRingBuffer = new Float32Array(ONNX_WINDOW_SIZE * 2);

    onnxWritePtr   = 0;
    console.log('ResNet laughter model loaded ✓');
    return true;
  } catch (e) {
    console.warn('ONNX model failed to load, falling back to heuristic audio:', e);
    return false;
  }
}

// Run model on the current 1-second ring-buffer window.
// Returns a probability in [0, 1] for the "laugh" class.
async function runOnnxInference() {
  if (!onnxReady || !onnxSession) return null;
  try {
    const clip = new Float32Array(ONNX_WINDOW_SIZE);
    for (let i = 0; i < ONNX_WINDOW_SIZE; i++) {
      clip[i] = onnxRingBuffer[(onnxWritePtr - ONNX_WINDOW_SIZE + i + onnxRingBuffer.length) % onnxRingBuffer.length];
    }

    const { data, nFilters, nFrames } = computeLogMelSpectrogram(clip);
    const tensor = new ort.Tensor('float32', data, [1, 1, nFilters, nFrames]);
    const output = await onnxSession.run({ input: tensor });

    const raw = output[Object.keys(output)[0]].data;
    // Model outputs a single sigmoid value — just use it directly
    return Math.min(Math.max(raw[0], 0), 1);
  } catch (e) {
    console.warn('ONNX inference error:', e);
    return null;
  }
}
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

socket.on('game_start', async ({ bestOf: bo, players, maxPlayers: mp, gameMode: gm }) => {
  bestOf     = bo || 5;
  maxPlayers = mp || 2;
  gameMode   = gm || 'standard';
  playerOrder = players;

  activePlayers     = [...players];
  eliminatedPlayers = [];
  eliminationFinals = false;

  scores        = {};
  allTimeScores = {};
  playerOrder.forEach(id => { scores[id] = 0; allTimeScores[id] = 0; });

  await startCamera();
  buildCamGrid(playerOrder);
  showScreen('game');
  initFaceDetection();
  await initAudioDetection();

  const myIndex = playerOrder.indexOf(me());
  for (let i = 0; i < myIndex; i++) await offerTo(playerOrder[i]);

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
      if (btn && btn.disabled) btn.textContent = 'waiting (' + data.votes + '/' + data.needed + ')...';
      break;
    }
    case 'begin_round':
      beginPrepPhase();
      break;
    case 'rematch_vote': {
      const btn = document.getElementById('next-btn');
      if (btn && btn.disabled) btn.textContent = 'waiting (' + data.votes + '/' + data.needed + ')...';
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
  playerOrder       = playerOrder.filter(p => p !== id);
  activePlayers     = activePlayers.filter(p => p !== id);
  eliminatedPlayers = eliminatedPlayers.filter(p => p !== id);
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
  updateCamsClass(cams, activePlayers.length, eliminatedPlayers.length);

  players.forEach((id) => {
    const isMe        = id === me();
    const isEliminated = eliminatedPlayers.includes(id);

    const wrap = document.createElement('div');
    wrap.className = 'cam-wrap' + (isEliminated ? ' spectator' : '');
    wrap.id = 'cam-slot-' + id;

    const vid = document.createElement('video');
    vid.id = 'video-' + id;
    vid.autoplay = true;
    vid.playsinline = true;
    if (isMe) {
      vid.muted = true;
      if (localStream) vid.srcObject = localStream;
    }

    const overlay = document.createElement('div');
    overlay.className = 'cam-overlay';
    overlay.id = 'overlay-' + id;
    overlay.textContent = isMe ? '📷' : '⏳';
    if (isMe && localStream) { overlay.style.opacity = '0'; overlay.style.display = 'none'; }

    const label = document.createElement('div');
    label.className = 'cam-label' + (isMe ? ' you' : '');
    label.innerHTML = isMe ? '<span class="live-dot"></span>you' : 'player ' + (playerOrder.indexOf(id) + 1);

    const laughBadge = document.createElement('div');
    laughBadge.className = 'laugh-badge';
    laughBadge.id = 'laugh-badge-' + id;
    laughBadge.textContent = '😂';

    const outBadge = document.createElement('div');
    outBadge.className = 'out-badge';
    outBadge.id = 'out-badge-' + id;
    outBadge.textContent = '💀';
    if (isEliminated) outBadge.classList.add('visible');

    wrap.appendChild(vid);
    wrap.appendChild(overlay);
    wrap.appendChild(label);
    wrap.appendChild(laughBadge);
    wrap.appendChild(outBadge);
    cams.appendChild(wrap);
  });
}

function updateCamsClass(cams, activeCount, spectatorCount) {
  cams.className = cams.className.replace(/players-\d+/g, '').replace('has-spectators', '').trim();
  cams.classList.add('cams');
  cams.classList.add('players-' + Math.max(activeCount, 1));
  if (spectatorCount > 0) cams.classList.add('has-spectators');
}

function eliminatePlayer(id) {
  if (eliminatedPlayers.includes(id)) return;
  eliminatedPlayers.push(id);
  activePlayers = activePlayers.filter(p => p !== id);

  const wrap = document.getElementById('cam-slot-' + id);
  if (wrap) {
    wrap.classList.add('spectator');
    const outBadge = document.getElementById('out-badge-' + id);
    if (outBadge) outBadge.classList.add('visible');
    const vid = document.getElementById('video-' + id);
    if (vid && vid.srcObject) vid.srcObject.getAudioTracks().forEach(t => { t.enabled = false; });
  }

  const cams = document.getElementById('cams');
  updateCamsClass(cams, activePlayers.length, eliminatedPlayers.length);
}

// ─── Audio Detection (ResNet ONNX + heuristic fallback) ───
async function initAudioDetection() {
  if (!localStream || audioCtx) return;

  // Try to load the ONNX model first
  const modelLoaded = await loadOnnxModel();

  try {
    // Use 8 kHz if ONNX model loaded (matches training), otherwise browser default
    const ctxOptions = modelLoaded ? { sampleRate: ONNX_SAMPLE_RATE } : {};
    audioCtx = new AudioContext(ctxOptions);
    if (audioCtx.state === 'suspended') audioCtx.resume();

    const source = audioCtx.createMediaStreamSource(localStream);

    if (modelLoaded) {
      // ── ResNet path: ScriptProcessor feeds ring buffer; inference runs on hop ──
      const bufferSize = 512;
      scriptProcessor = audioCtx.createScriptProcessor(bufferSize, 1, 1);
      source.connect(scriptProcessor);
      scriptProcessor.connect(audioCtx.destination);

      scriptProcessor.onaudioprocess = async (e) => {
  if (!audioCtx) return;
  const input = e.inputBuffer.getChannelData(0);
  const nativeSR = audioCtx.sampleRate;
  
  // Downsample to 8kHz if needed
  if (nativeSR !== ONNX_SAMPLE_RATE) {
    const ratio = nativeSR / ONNX_SAMPLE_RATE;
    for (let i = 0; i < input.length; i += ratio) {
      const idx = Math.floor(i);
      onnxRingBuffer[onnxWritePtr % onnxRingBuffer.length] = input[idx];
      onnxWritePtr++;
      onnxFramesSinceInference++;
    }
  } else {
    for (let i = 0; i < input.length; i++) {
      onnxRingBuffer[onnxWritePtr % onnxRingBuffer.length] = input[i];
      onnxWritePtr++;
    }
    onnxFramesSinceInference += input.length;
  }

  if (onnxFramesSinceInference < ONNX_HOP_SIZE) return;
  onnxFramesSinceInference = 0;

  if (!laughDetectionActive || laughTriggered) {
    _audioLaughConfidence = 0;
    return;
  }

  let rmsSum = 0;
  for (let i = 0; i < ONNX_WINDOW_SIZE; i++) {
    const s = onnxRingBuffer[(onnxWritePtr - ONNX_WINDOW_SIZE + i + onnxRingBuffer.length) % onnxRingBuffer.length];
    rmsSum += s * s;
  }
  const rms = Math.sqrt(rmsSum / ONNX_WINDOW_SIZE);
  if (rms < 0.01) { _audioLaughConfidence = 0; return; }

  const prob = await runOnnxInference();
  if (prob === null) return;
  _audioLaughConfidence = prob;
  updateAudioTrigger();
};

      console.log('Audio detection: ResNet ONNX mode');
    } else {
      // ── Heuristic fallback (original code) ────────────────
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 1024;
      source.connect(analyser);
      runHeuristicAudioLoop(analyser);
      console.log('Audio detection: heuristic fallback mode');
    }
  } catch (e) {
    console.warn('Audio detection unavailable:', e);
  }
}

// Called after each ONNX inference (or heuristic frame) to check for laugh trigger
function updateAudioTrigger() {
  if (!laughDetectionActive || laughTriggered || faceMesh) return;

  if (_audioLaughConfidence >= AUDIO_SOLO_THRESHOLD) {
    if (!laughStartTime) laughStartTime = Date.now();
    else if (Date.now() - laughStartTime > LAUGH_SUSTAIN_MS) triggerLaugh();
  } else {
    laughStartTime = null;
  }
}

// Original heuristic audio loop (unchanged, used as fallback)
function runHeuristicAudioLoop(analyser) {
  const bufLen  = analyser.frequencyBinCount;
  const dataArr = new Uint8Array(bufLen);
  const binHz   = audioCtx.sampleRate / analyser.fftSize;

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
  let zcBuffer     = [];
  let energyBuffer = [];
  let fluxBuffer   = [];
  const BUF_SIZE   = 20;

  const audioLoop = () => {
    if (!audioCtx) return;
    analyser.getByteFrequencyData(dataArr);

    let rmsSum = 0;
    for (let i = 0; i < bufLen; i++) rmsSum += (dataArr[i] / 255) ** 2;
    const rms = Math.sqrt(rmsSum / bufLen);

    if (rms < 0.02) {
      _audioLaughConfidence = 0;
      prevBands.fill(0);
      requestAnimationFrame(audioLoop);
      return;
    }

    const bands = new Array(MEL_BANDS).fill(0);
    let totalEnergy = 0;
    for (let b = 0; b < MEL_BANDS; b++) {
      const lo = melPoints[b], hi = melPoints[b + 2];
      let sum = 0, cnt = 0;
      for (let k = lo; k <= Math.min(hi, bufLen - 1); k++) { sum += dataArr[k]; cnt++; }
      bands[b] = cnt > 0 ? sum / cnt : 0;
      totalEnergy += bands[b];
    }
    if (totalEnergy < 1) { _audioLaughConfidence = 0; requestAnimationFrame(audioLoop); return; }

    let centNum = 0, centDen = 0;
    for (let b = 0; b < MEL_BANDS; b++) { centNum += b * bands[b]; centDen += bands[b]; }
    const centroid = centDen > 0 ? centNum / (centDen * (MEL_BANDS - 1)) : 0;

    let cum = 0, rolloff = MEL_BANDS - 1;
    for (let b = 0; b < MEL_BANDS; b++) {
      cum += bands[b];
      if (cum / totalEnergy >= 0.85) { rolloff = b; break; }
    }
    const rolloffNorm = rolloff / (MEL_BANDS - 1);

    let flux = 0;
    for (let b = 0; b < MEL_BANDS; b++) flux += Math.abs(bands[b] - prevBands[b]);
    flux /= (MEL_BANDS * 255);
    prevBands = [...bands];

    let hfSum = 0;
    for (let b = 8; b < MEL_BANDS; b++) hfSum += bands[b];
    const zcr = totalEnergy > 0 ? hfSum / totalEnergy : 0;

    const push = (buf, val) => { buf.push(val); if (buf.length > BUF_SIZE) buf.shift(); };
    push(energyBuffer, rms);
    push(fluxBuffer,   flux);
    push(zcBuffer,     zcr);

    const avgFlux = fluxBuffer.reduce((a, b) => a + b, 0) / fluxBuffer.length;
    const avgZcr  = zcBuffer.reduce((a, b)  => a + b, 0)  / zcBuffer.length;

    const centroidScore = (centroid > 0.30 && centroid < 0.80) ? 1 : 0;
    const rolloffScore  = (rolloffNorm > 0.45 && rolloffNorm < 0.92) ? 1 : 0;
    const fluxScore     = Math.min(avgFlux / 0.04, 1);
    const zcrScore      = (avgZcr > 0.15 && avgZcr < 0.65) ? 1 : 0;
    const energyScore   = Math.min(rms / 0.12, 1);

    _audioLaughConfidence = Math.min(
      (centroidScore * 0.20) +
      (rolloffScore  * 0.20) +
      (fluxScore     * 0.25) +
      (zcrScore      * 0.15) +
      (energyScore   * 0.20),
      1
    );

    updateAudioTrigger();
    requestAnimationFrame(audioLoop);
  };

  requestAnimationFrame(audioLoop);
}

// ─── Face Detection ───────────────────────────────────────
const LM_UPPER_LIP    = 13;
const LM_LOWER_LIP    = 14;
const LM_LEFT_CORNER  = 78;
const LM_RIGHT_CORNER = 308;
const LM_LEFT_CHEEK   = 50;
const LM_RIGHT_CHEEK  = 280;
const LM_LEFT_EYE_TOP = 159;
const LM_LEFT_EYE_BOT = 145;
const LM_RIGHT_EYE_TOP= 386;
const LM_RIGHT_EYE_BOT= 374;

function getFaceFeatures(lm) {
  const mouthH = Math.abs(lm[LM_LOWER_LIP].y - lm[LM_UPPER_LIP].y);
  const mouthW = Math.abs(lm[LM_RIGHT_CORNER].x - lm[LM_LEFT_CORNER].x);
  const mouth  = mouthW > 0.001 ? mouthH / mouthW : 0;

  const cornerMidY = (lm[LM_LEFT_CORNER].y + lm[LM_RIGHT_CORNER].y) / 2;
  const cheekMidY  = (lm[LM_LEFT_CHEEK].y  + lm[LM_RIGHT_CHEEK].y)  / 2;
  const cheek      = cornerMidY - cheekMidY;

  const leftEyeH  = Math.abs(lm[LM_LEFT_EYE_TOP].y  - lm[LM_LEFT_EYE_BOT].y);
  const rightEyeH = Math.abs(lm[LM_RIGHT_EYE_TOP].y - lm[LM_RIGHT_EYE_BOT].y);
  const eyeApert  = (leftEyeH + rightEyeH) / 2;

  return { mouth, cheek, eyeApert };
}

function initFaceDetection() {
  detectionCanvas        = document.createElement('canvas');
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

  const mouthThresh = baselineMouth + 0.22;
  const pct = Math.min((mouth / (mouthThresh * 1.2)) * 100, 100);
  const bar = document.getElementById('mouth-indicator');
  if (bar) {
    bar.style.width      = pct + '%';
    bar.style.background = mouth > mouthThresh ? 'var(--danger)' : 'var(--accent)';
  }

  if (calibrating) {
    calibrationSamples.push({ mouth, cheek, eyeApert });
    if (calibrationSamples.length >= CALIBRATION_FRAMES) finishCalibration();
    return;
  }

  if (!laughDetectionActive || laughTriggered) return;

  const mouthDelta = mouth - baselineMouth;
  const mouthScore = Math.min(Math.max(mouthDelta / 0.22, 0), 1);
  const cheekDelta = cheek - baselineCheek;
  const cheekScore = Math.min(Math.max(cheekDelta / 0.10, 0), 1);

  const eyeDelta   = baselineEye - eyeApert;
  const eyeScore   = Math.min(Math.max(eyeDelta / 0.015, 0), 1);
  const faceScore = Math.min((mouthScore * 0.80) + (cheekScore * 0.10) + (eyeScore * 0.10), 1);

  const blended    = (faceScore * FACE_WEIGHT) + (_audioLaughConfidence * AUDIO_WEIGHT);

  // ── ADD THIS ──
  if (Math.random() < 0.05) { // log ~5% of frames to avoid spam
    console.log(
      'face:', faceScore.toFixed(2),
      '| audio:', _audioLaughConfidence.toFixed(2),
      '| blended:', blended.toFixed(2),
      '| mouth:', mouthScore.toFixed(2),
      '| cheek:', cheekScore.toFixed(2),
      '| eye:', eyeScore.toFixed(2)
    );
  }
  // ─────────────

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

  const trimmed = (arr) => {
    const sorted = [...arr].sort((a, b) => a - b);
    const lo    = Math.floor(n * 0.2);
    const hi    = Math.ceil(n * 0.8);
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
  laughTriggered       = true;
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
  if (badge) {
    badge.classList.add('visible');
    if (gameMode === 'elimination' && !eliminationFinals) {
      setTimeout(() => badge.classList.remove('visible'), 800);
    }
  }

  if (gameMode === 'elimination' && !eliminationFinals) {
    handleEliminationLaugh(id);
  } else {
    const remaining = activePlayers.filter(p => !laughedThisRound.has(p));
    if (activePlayers.length <= 2) {
      endRound();
    } else {
      if (remaining.length <= 1) endRound();
    }
  }
}

function handleEliminationLaugh(id) {
  eliminatePlayer(id);
  const stillActive = activePlayers.length;

  if (stillActive === 1) {
    roundActive          = false;
    laughDetectionActive = false;
    stopTimer();
    document.getElementById('laugh-btn').disabled = true;
    playerOrder.forEach(pid => { allTimeScores[pid] = (allTimeScores[pid] || 0) + (scores[pid] || 0); });
    setTimeout(() => showGameOver(activePlayers[0]), 1200);
    return;
  }

  if (stillActive === 2) {
    eliminationFinals = true;
    activePlayers.forEach(pid => { scores[pid] = 0; });
    updateScoreHUD();
    showFinalsFlash(() => {});
    return;
  }
}

function endRound() {
  if (!roundActive) return;
  roundActive          = false;
  laughDetectionActive = false;
  stopTimer();
  document.getElementById('laugh-btn').disabled = true;

  const remaining = activePlayers.filter(p => !laughedThisRound.has(p));
  const winnerId  = remaining.length >= 1 ? remaining[0] : null;

  if (winnerId) scores[winnerId] = (scores[winnerId] || 0) + 1;
  updateScoreHUD();

  const winsNeeded  = Math.ceil(bestOf / 2);
  const matchWinner = activePlayers.find(id => (scores[id] || 0) >= winsNeeded);

  if (matchWinner) {
    playerOrder.forEach(id => { allTimeScores[id] = (allTimeScores[id] || 0) + (scores[id] || 0); });
    showGameOver(matchWinner);
  } else {
    showRoundResult(winnerId);
  }
}

function showFinalsFlash(onDone) {
  const existing = document.getElementById('finals-flash');
  if (existing) existing.remove();

  const flash = document.createElement('div');
  flash.id = 'finals-flash';
  flash.innerHTML = `
    <div class="finals-flash-inner">
      <div class="finals-flash-label">finals</div>
      <div class="finals-flash-sub">${
        activePlayers.map(pid => 'player ' + (playerOrder.indexOf(pid) + 1)).join(' vs ')
      }</div>
    </div>
  `;
  document.body.appendChild(flash);

  setTimeout(() => {
    flash.classList.add('finals-flash-out');
    setTimeout(() => { flash.remove(); onDone && onDone(); }, 500);
  }, 2000);
}

// ─── Result screens ───────────────────────────────────────
function showRoundResult(winnerId) {
  const iWon     = winnerId === me();
  const noWinner = winnerId === null;
  const iAmActive = activePlayers.includes(me());

  document.getElementById('round-emoji').textContent = noWinner ? '🤝' : iWon ? '🏆' : (iAmActive ? '😅' : '💀');
  document.getElementById('round-title').textContent = noWinner ? 'everyone laughed' : iWon ? 'you held it!' : 'you laughed';
  document.getElementById('round-title').className   = 'result-title ' + (iWon ? 'win' : noWinner ? '' : 'lose');
  document.getElementById('round-sub').textContent   = noWinner ? 'no points awarded'
    : iWon ? 'you win the round'
    : winnerId ? 'player ' + (playerOrder.indexOf(winnerId) + 1) + ' wins the round' : '';

  document.getElementById('rs-you').textContent  = scores[me()] || 0;
  const others = activePlayers.filter(id => id !== me());
  document.getElementById('rs-them').textContent = others.map(id => scores[id] || 0).join(' / ');

  const btn = document.getElementById('next-btn');
  btn.disabled    = false;
  btn.textContent = iAmActive ? 'ready for next round →' : 'watching... →';
  btn.onclick     = requestNextRound;
  showScreen('round');
}

function showGameOver(winnerId) {
  const iWon = winnerId === me();
  document.getElementById('over-emoji').textContent = iWon ? '🏆' : (gameMode === 'elimination' ? '💀' : '😭');
  document.getElementById('over-title').textContent = iWon ? 'you win!' : 'you lose';
  document.getElementById('over-title').className   = 'result-title ' + (iWon ? 'win' : 'lose');

  const allTimeMe     = allTimeScores[me()] || 0;
  const allTimeOthers = playerOrder.filter(id => id !== me()).map(id => allTimeScores[id] || 0).join(' / ');
  document.getElementById('os-you').textContent  = allTimeMe;
  document.getElementById('os-them').textContent = allTimeOthers;
  document.getElementById('over-sub').textContent = 'all time · you ' + allTimeMe + ' vs ' + allTimeOthers;

  const btn = document.getElementById('next-btn');
  btn.disabled    = false;
  btn.textContent = 'rematch';
  btn.onclick     = requestRematch;
  showScreen('over');
}

// ─── Next round / rematch ─────────────────────────────────
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
  activePlayers     = [...playerOrder];
  eliminatedPlayers = [];
  eliminationFinals = false;

  playerOrder.forEach(id => { scores[id] = 0; });
  laughTriggered = false;
  laughedThisRound.clear();
  updateScoreHUD();

  for (const id in peers) { peers[id].close(); }
  peers = {};

  buildCamGrid(playerOrder);
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
  roundActive           = false;
  laughDetectionActive  = false;
  laughTriggered        = false;
  laughStartTime        = null;
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

  const iAmActive = activePlayers.includes(me());
  if (faceMesh && iAmActive) {
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
  const iAmActive = activePlayers.includes(me());
  setStatus(iAmActive ? "😐  don't laugh." : '👀  watching...', iAmActive);

  roundActive          = true;
  laughDetectionActive = iAmActive;
  laughTriggered       = !iAmActive;
  laughStartTime       = null;
  calibrating          = false;

  if (!faceMesh && iAmActive) document.getElementById('laugh-btn').disabled = false;
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
  const others = (eliminationFinals ? activePlayers : playerOrder).filter(id => id !== me());
  document.getElementById('score-them').textContent = others.map(id => scores[id] || 0).join('/');
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
  const modeRow = document.getElementById('mode-row');
  if (modeRow) modeRow.style.display = n >= 3 ? '' : 'none';
  if (n < 3) setGameMode('standard');
}

function setGameMode(mode) {
  selectedGameMode = mode;
  ['standard', 'elimination'].forEach(m => {
    const btn = document.getElementById('gm-' + m);
    if (btn) btn.className = 'btn' + (m === mode ? ' primary' : '');
  });
  const desc = document.getElementById('mode-desc');
  if (desc) {
    desc.textContent = mode === 'elimination'
      ? "laugh = you're out · last one standing wins"
      : 'first to laugh loses the round · best of ' + selectedBestOf;
  }
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
  socket.emit('create_room', {
    code,
    bestOf: selectedBestOf,
    maxPlayers: selectedMaxPlayers,
    gameMode: selectedGameMode,
  });
}

function joinRoom() {
  document.getElementById('lobby-error').textContent = '';
  const code = document.getElementById('join-code').value.trim().toUpperCase();
  if (code.length !== 4) { document.getElementById('lobby-error').textContent = 'enter a 4-letter code'; return; }
  socket.emit('join_room', code);
}

function goLobby()   { cleanup(); showScreen('lobby'); }
function leaveGame() { if (confirm('Leave the game?')) goLobby(); }

function cleanup() {
  stopTimer();
  detectionRunning      = false;
  laughDetectionActive  = false;
  calibrating           = false;
  _audioLaughConfidence = 0;
  onnxFramesSinceInference = 0;

  if (scriptProcessor) {
    try { scriptProcessor.disconnect(); } catch (_) {}
    scriptProcessor = null;
  }
  if (audioCtx) { audioCtx.close(); audioCtx = null; }
  if (localStream) { localStream.getTracks().forEach(t => t.stop()); localStream = null; }
  for (const id in peers) { peers[id].close(); }
  peers = {};
  if (faceMesh) { faceMesh.close(); faceMesh = null; }

  roomCode          = null;
  playerOrder       = [];
  activePlayers     = [];
  eliminatedPlayers = [];
  eliminationFinals = false;
  scores            = {};
  allTimeScores     = {};
  laughedThisRound.clear();
  document.getElementById('join-code').value = '';
  document.getElementById('timer').classList.remove('danger');
}

// ─── Keyboard shortcut ────────────────────────────────────
document.addEventListener('keydown', e => {
  if (e.code === 'Space' && document.getElementById('screen-game').classList.contains('active')) {
    e.preventDefault();
    if (!document.getElementById('laugh-btn').disabled) iLaughed();
  }
});