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
let micCaptureNode       = null;   // AudioWorkletNode or ScriptProcessor; disconnected on cleanup

let faceMesh             = null;
let laughDetectionActive = false;
let laughTriggered       = false;
// Hysteresis + leaky-evidence trigger (see LaughGate in laugh-audio.js).
// Scores fed to it are normalised so 1.0 == the relevant trigger threshold;
// it tolerates the short dips between laugh syllables, which the old
// "reset the timer on any dip" rule did not.
const GATE_OFF_RATIO = 0.85;   // hysteresis: hold evidence down to 85% of threshold
const laughGate = new LaughAudio.LaughGate({
  on: 1, off: GATE_OFF_RATIO, sustainMs: LAUGH_SUSTAIN_MS, decayRate: 0.5,
});

let faceBaseline   = { ...LaughFace.DEFAULT_BASELINE };   // see laugh-face.js
// Drops smile/squint spikes from spread vowels and blinks (see laugh-face.js).
const facePersistence = new LaughFace.CuePersistence();
let calibrating    = false;
// Set `localStorage.ogiggleDebug = 1` to log detector scores to the console.
const DEBUG_LAUGH  = (() => { try { return !!localStorage.getItem('ogiggleDebug'); } catch (_) { return false; } })();
let lastDebugLogAt = 0;
let calibrationSamples = [];

let detectionCanvas, detectionCtx;
let detectionRunning = false;

let _audioLaughConfidence = 0;
let audioUpdatedAt        = 0;   // ms timestamp of the last ResNet audio estimate
// Audio alone may fire even while the face is visible (laughing behind a
// hand, weak face calibration), but needs more than when the face is lost.
const AUDIO_SOLO_THRESHOLD_FACE_VISIBLE = 0.88;

// ─── ResNet ONNX model state ──────────────────────────────
let onnxSession    = null;   // ort.InferenceSession once loaded
let onnxReady      = false;  // true after model loads successfully
let onnxTimeMajor  = true;   // model input laid out [1,1,time,mel] (as trained)
let laughStream    = null;   // LaughAudio.LogMelStream (8 kHz log-mel frames)
let audioResampler = null;   // LaughAudio.Resampler (native rate → 8 kHz)
let probSmoother   = null;   // LaughAudio.ProbSmoother
// Burst-rhythm cue: laughter's "ha-ha-ha" calls repeat regularly, speech and
// noise don't. Scales the model probability by 0.9–1 (see BurstRhythm).
// localStorage.ogiggleNoRhythm = 1 disables it for A/B testing.
const RHYTHM_ENABLED = (() => { try { return !localStorage.getItem('ogiggleNoRhythm'); } catch (_) { return true; } })();
let burstRhythm    = null;   // LaughAudio.BurstRhythm
let lastRhythmScale = 1;     // last applied rhythm multiplier (debug log)
// Voicing cue: laughter alternates voiced vowels with unvoiced "h" breaths;
// humming, singing, held vowels and music stay voiced. Scales the model
// probability by 0.9–1 (see VoicingCue).
// localStorage.ogiggleNoVoicing = 1 disables it for A/B testing.
const VOICING_ENABLED = (() => { try { return !localStorage.getItem('ogiggleNoVoicing'); } catch (_) { return true; } })();
let voicingCue     = null;   // LaughAudio.VoicingCue
let lastVoicingScale = 1;    // last applied voicing multiplier (debug log)
let inferenceBusy  = false;  // never overlap session.run() calls
let framesSinceInference = 0;
let lastFaceSeenAt = 0;      // ms timestamp of the last frame with a face

const ONNX_SAMPLE_RATE    = 8000;
const INFER_EVERY_FRAMES  = 4;      // 4 × 186 samples ≈ 93 ms between inferences
const AUDIO_RMS_GATE      = 0.008;  // loudest ~256 ms frame below this → window is silence
const FACE_LOST_MS        = 600;    // after this long without a face, audio decides alone
const ONNX_MODEL_PATH     = '/models/laughter_resnet.onnx';
const MIC_WORKLET_PATH    = '/mic-worklet.js';
// While opponents' audio is playing (see FarEndGuard), local audio may be
// their laugh leaking past echo cancellation: demand more from audio alone
// and lean on the face in the fusion.
const AUDIO_SOLO_THRESHOLD_FAR_END = 0.92;
const FAR_END_AUDIO_SCALE          = 0.6;
const farEndGuard  = new LaughAudio.FarEndGuard();
const farEndMeters = new Map();   // remote MediaStream → AnalyserNode (or null if unsupported)
let   farEndBuf    = null;

// Meter the loudest remote player's audio and update the far-end guard.
function updateFarEnd() {
  const now = Date.now();
  if (!audioCtx) return farEndGuard.update(0, now);
  let level = 0;
  for (const id of playerOrder) {
    if (id === me()) continue;
    const vid = document.getElementById('video-' + id);
    const stream = vid && vid.srcObject;
    if (!stream || !stream.getAudioTracks().some(t => t.enabled && t.readyState === 'live')) continue;
    let an = farEndMeters.get(stream);
    if (an === undefined) {
      try {
        an = audioCtx.createAnalyser();
        an.fftSize = 1024;
        audioCtx.createMediaStreamSource(stream).connect(an);
      } catch (e) {
        console.warn('Cannot meter remote audio:', e);
        an = null;
      }
      farEndMeters.set(stream, an);
    }
    if (!an) continue;
    if (!farEndBuf || farEndBuf.length !== an.fftSize) farEndBuf = new Float32Array(an.fftSize);
    an.getFloatTimeDomainData(farEndBuf);
    level = Math.max(level, LaughAudio.rmsOf(farEndBuf));
  }
  return farEndGuard.update(level, now);
}

// ─── Load ONNX model ──────────────────────────────────────
// Features + model notes live in laugh-audio.js. The shipped model declares
// its input as [1,1,128,time] but the Gillick weights were trained on
// [1,1,44 time,128 mel]; we patch the declaration in memory so we can feed the
// layout the network actually learned.
async function loadOnnxModel() {
  if (typeof ort === 'undefined' || typeof LaughAudio === 'undefined') {
    console.warn('onnxruntime-web / laugh-audio not loaded — falling back to heuristic audio');
    return false;
  }
  try {
    const resp  = await fetch(ONNX_MODEL_PATH);
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    const bytes   = new Uint8Array(await resp.arrayBuffer());
    const patched = LaughAudio.patchOnnxInputShape(bytes);
    onnxTimeMajor = !!patched;
    if (!patched) console.warn('ONNX input signature not found; feeding mel×time layout');
    onnxSession   = await ort.InferenceSession.create(patched || bytes);
    onnxReady     = true;
    laughStream   = new LaughAudio.LogMelStream();
    probSmoother  = new LaughAudio.ProbSmoother(0.5);
    burstRhythm   = new LaughAudio.BurstRhythm();
    voicingCue    = new LaughAudio.VoicingCue();
    console.log('ResNet laughter model loaded ✓ (layout:', onnxTimeMajor ? 'time×mel' : 'mel×time', ')');
    return true;
  } catch (e) {
    console.warn('ONNX model failed to load, falling back to heuristic audio:', e);
    return false;
  }
}

// Run model on the current ~1.26 s window (44 frames).
// Returns a probability in [0, 1] for the "laugh" class.
async function runOnnxInference() {
  if (!onnxReady || !onnxSession || !laughStream || !laughStream.ready()) return null;
  const { N_FRAMES, N_MELS } = LaughAudio;
  try {
    let data = laughStream.getWindow();
    let dims = [1, 1, N_FRAMES, N_MELS];
    if (!onnxTimeMajor) {
      const t = new Float32Array(data.length);
      for (let f = 0; f < N_FRAMES; f++)
        for (let m = 0; m < N_MELS; m++) t[m * N_FRAMES + f] = data[f * N_MELS + m];
      data = t; dims = [1, 1, N_MELS, N_FRAMES];
    }
    const tensor = new ort.Tensor('float32', data, dims);
    const output = await onnxSession.run({ [onnxSession.inputNames[0]]: tensor });
    const raw = output[onnxSession.outputNames[0]].data;
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
    // Echo cancellation keeps the opponents' laughter (played through our
    // speakers) from being picked up by our mic and counted as *our* laugh.
    // Speech-tuned noise suppression and AGC may weaken or compress laughter.
    // For A/B testing, set localStorage.ogiggleRawMic = 1 to turn both off
    // (echo cancellation stays on). The applied settings are logged, because
    // browsers may ignore the request.
    const rawMic = (() => { try { return !!localStorage.getItem('ogiggleRawMic'); } catch (_) { return false; } })();
    localStream = await navigator.mediaDevices.getUserMedia({
      video: true,
      audio: { echoCancellation: true, noiseSuppression: !rawMic, autoGainControl: !rawMic },
    });
    const at = localStream.getAudioTracks()[0];
    if (at && (rawMic || DEBUG_LAUGH) && at.getSettings) {
      const s = at.getSettings();
      console.log('Mic settings — echoCancellation:', s.echoCancellation,
                  'noiseSuppression:', s.noiseSuppression, 'autoGainControl:', s.autoGainControl);
    }
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
    // Prefer an 8 kHz context (browser resamples with a proper filter). Some
    // browsers (Firefox) refuse to connect a mic stream to a context at a
    // different rate, so fall back to the native rate + our own resampler.
    audioCtx = null;
    let source = null;
    if (modelLoaded) {
      try {
        audioCtx = new AudioContext({ sampleRate: ONNX_SAMPLE_RATE });
        source   = audioCtx.createMediaStreamSource(localStream);
      } catch (e) {
        console.warn('8 kHz AudioContext unavailable, resampling in JS:', e);
        if (audioCtx) { try { audioCtx.close(); } catch (_) {} }
        audioCtx = null;
      }
    }
    if (!audioCtx) {
      audioCtx = new AudioContext();
      source   = audioCtx.createMediaStreamSource(localStream);
    }
    if (audioCtx.state === 'suspended') audioCtx.resume();

    if (modelLoaded) {
      // ── ResNet path: mic capture → resampler → log-mel stream → model ──
      audioResampler = new LaughAudio.Resampler(audioCtx.sampleRate, ONNX_SAMPLE_RATE);
      laughStream.reset();
      probSmoother.reset();
      burstRhythm.reset();
      voicingCue.reset();
      framesSinceInference = 0;

      const onMicSamples = (chunk) => {
        if (!audioCtx || !laughStream) return;
        const samples = audioResampler.process(chunk);
        framesSinceInference += laughStream.push(samples);
        burstRhythm.push(samples);
        voicingCue.push(samples);

        if (framesSinceInference < INFER_EVERY_FRAMES || inferenceBusy) return;
        framesSinceInference = 0;
        updateFarEnd();

        if (!laughDetectionActive || laughTriggered) {
          _audioLaughConfidence = 0;
          probSmoother.reset();
          return;
        }
        if (!laughStream.ready()) return;
        if (laughStream.peakRms() < AUDIO_RMS_GATE) {
          _audioLaughConfidence = probSmoother.decay();
          audioUpdatedAt = Date.now();
          updateAudioTrigger();
          return;
        }

        inferenceBusy = true;
        runOnnxInference().then((prob) => {
          inferenceBusy = false;
          if (prob === null || !laughDetectionActive) return;
          const rhythm = RHYTHM_ENABLED ? burstRhythm.scale() : 1;
          lastRhythmScale = rhythm;
          const voicing = VOICING_ENABLED ? voicingCue.scale() : 1;
          lastVoicingScale = voicing;
          _audioLaughConfidence = probSmoother.update(prob * rhythm * voicing);
          audioUpdatedAt = Date.now();
          updateAudioTrigger();
        });
      };

      // Prefer an AudioWorklet (audio thread, no dropped buffers when the
      // main thread is busy with FaceMesh); fall back to ScriptProcessor.
      const ctx = audioCtx;
      let captureMode = 'ScriptProcessor';
      if (ctx.audioWorklet && typeof AudioWorkletNode !== 'undefined') {
        try {
          await ctx.audioWorklet.addModule(MIC_WORKLET_PATH);
          if (audioCtx !== ctx) return;          // cleaned up while loading
          const node = new AudioWorkletNode(ctx, 'mic-capture', {
            numberOfInputs: 1, numberOfOutputs: 0, channelCount: 1, channelCountMode: 'explicit',
          });
          node.port.onmessage = (e) => onMicSamples(e.data);
          source.connect(node);
          micCaptureNode = node;
          captureMode = 'AudioWorklet';
        } catch (e) {
          console.warn('AudioWorklet unavailable, using ScriptProcessor:', e);
        }
      }
      if (!micCaptureNode) {
        const sp = ctx.createScriptProcessor(1024, 1, 1);
        source.connect(sp);
        sp.connect(ctx.destination);   // output stays silent
        sp.onaudioprocess = (e) => onMicSamples(e.inputBuffer.getChannelData(0));
        micCaptureNode = sp;
      }

      console.log('Audio detection: ResNet ONNX mode (' + captureMode + ')');
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
// When FaceMesh is running and sees a face, onFaceResults fuses face+audio and
// owns the trigger. If the face is lost (turned away, covered by a hand —
// common while laughing!) audio decides on its own instead of being ignored.
function updateAudioTrigger() {
  if (!laughDetectionActive || laughTriggered) return;
  // (Only the neural model is trusted alone; the heuristic is too speech-prone.)
  if (faceMesh && (!onnxReady || Date.now() - lastFaceSeenAt < FACE_LOST_MS)) return;

  const now = Date.now();
  const threshold = farEndGuard.active(now) ? AUDIO_SOLO_THRESHOLD_FAR_END : AUDIO_SOLO_THRESHOLD;
  if (laughGate.update(_audioLaughConfidence / threshold, now)) triggerLaugh();
  updateLaughMeter();
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
// Landmark features + laughter scoring live in laugh-face.js (LaughFace).

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
  lastFaceSeenAt = Date.now();
  const lm = results.multiFaceLandmarks[0];
  const features = LaughFace.getFaceFeatures(lm, detectionCanvas.width / detectionCanvas.height);
  if (!features) return;
  const { mouth } = features;

  // Outside a live round the bar shows mouth opening (useful while
  // calibrating); during a round updateLaughMeter() shows the trigger gate.
  if (!laughDetectionActive) {
    const mouthThresh = faceBaseline.mouth + LaughFace.SCALE.open;
    setDetectionBar('mouth openness', Math.min(mouth / (mouthThresh * 1.2), 1), mouth > mouthThresh);
  }

  if (calibrating) {
    calibrationSamples.push(features);
    if (calibrationSamples.length >= CALIBRATION_FRAMES) finishCalibration();
    return;
  }

  if (!laughDetectionActive || laughTriggered) return;

  // Mouth opening alone (talking, yawning) caps at 0.45 — below the face-only
  // threshold — so a face-only trigger needs smile/eye cues too.
  const now       = Date.now();
  const face      = facePersistence.update(LaughFace.scoreFace(features, faceBaseline), now);
  const faceScore = face.score;
  const farEnd    = farEndGuard.active(now);
  const audioConf = _audioLaughConfidence * (farEnd ? FAR_END_AUDIO_SCALE : 1);
  const blended   = (faceScore * FACE_WEIGHT) + (audioConf * AUDIO_WEIGHT);
  // Either the fused score, a very strong face, or (neural model only) very
  // confident audio counts; each is normalised by its own threshold so the
  // gate sees 1.0 at whichever fires first. Far-end scaling (×0.6) keeps the
  // audio-alone route below 1.0 while opponents are audible.
  const gateScore = LaughAudio.fusedGateScore({
    face: faceScore,
    audio: _audioLaughConfidence,
    audioScale: farEnd ? FAR_END_AUDIO_SCALE : 1,
    audioAgeMs: onnxReady ? now - audioUpdatedAt : 0,   // heuristic updates every rAF
  }, {
    faceWeight: FACE_WEIGHT, audioWeight: AUDIO_WEIGHT,
    combinedThreshold: COMBINED_TRIGGER_THRESHOLD, faceSoloThreshold: FACE_SOLO_THRESHOLD,
    audioSoloThreshold: onnxReady ? AUDIO_SOLO_THRESHOLD_FACE_VISIBLE : Infinity,
  });

  if (DEBUG_LAUGH && Date.now() - lastDebugLogAt > 250) {
    lastDebugLogAt = Date.now();
    console.log(
      'face:', faceScore.toFixed(2),
      '| audio:', _audioLaughConfidence.toFixed(2),
      '| rhythm:', lastRhythmScale.toFixed(2),
      '| voicing:', lastVoicingScale.toFixed(2),
      '| blended:', blended.toFixed(2),
      '| open:', face.open.toFixed(2),
      '| smile:', face.smile.toFixed(2) + '/' + face.rawSmile.toFixed(2),
      '| eye:', face.eye.toFixed(2) + '/' + face.rawEye.toFixed(2),
      '| gate:', laughGate.progress().toFixed(2),
      '| farEnd:', farEnd ? farEndGuard.level.toFixed(3) : 'off'
    );
  }

  if (laughGate.update(gateScore, now)) triggerLaugh();
  updateLaughMeter();
}

// Detection bar under the cams. During a round it shows how close the trigger
// gate is to firing (what actually decides), not just one facial cue.
function setDetectionBar(label, frac, hot) {
  const bar = document.getElementById('mouth-indicator');
  if (bar) {
    bar.style.width      = (Math.max(0, Math.min(frac, 1)) * 100) + '%';
    bar.style.background = hot ? 'var(--danger)' : 'var(--accent)';
  }
  const lbl = document.getElementById('detection-label');
  if (lbl && lbl.textContent !== label) lbl.textContent = label;
}
function updateLaughMeter() {
  if (!laughDetectionActive) return;
  const p = laughGate.progress();
  setDetectionBar('crack meter', p, p > 0.6);
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

// Called when CALIBRATION_FRAMES have been collected, or when the round starts.
// The countdown lasts ~3.6 s, so below ~25 fps (common for FaceMesh on
// laptops, and in round 1 while the model is still loading) the 90-frame
// target is never reached; previously those samples were silently dropped
// and the round ran on the default baseline (no width/eye cues at all).
// Too few frames → keep the previous baseline rather than trust noise.
function finishCalibration() {
  calibrating = false;
  if (calibrationSamples.length < LaughFace.MIN_CALIBRATION_FRAMES) {
    console.warn('Calibration: only', calibrationSamples.length, 'face frames — keeping previous baseline');
    return;
  }
  faceBaseline = LaughFace.calibrate(calibrationSamples);
  console.log('Calibrated — mouth:', faceBaseline.mouth.toFixed(4),
              'width:', faceBaseline.width.toFixed(4),
              'lift:', faceBaseline.lift.toFixed(4),
              'eye:', faceBaseline.eye.toFixed(4));
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
  laughGate.reset();
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
  laughGate.reset();
  facePersistence.reset();
  if (calibrating) finishCalibration();

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
  framesSinceInference  = 0;
  inferenceBusy         = false;
  audioResampler        = null;
  if (laughStream)  laughStream.reset();
  if (probSmoother) probSmoother.reset();
  if (burstRhythm)  burstRhythm.reset();
  if (voicingCue)   voicingCue.reset();

  if (micCaptureNode) {
    try {
      if (micCaptureNode.port) micCaptureNode.port.onmessage = null;
      micCaptureNode.disconnect();
    } catch (_) {}
    micCaptureNode = null;
  }
  farEndMeters.clear();
  farEndGuard.reset();
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