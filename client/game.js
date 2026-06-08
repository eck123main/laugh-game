// ─── Config ───────────────────────────────────────────────
const RTC_CONFIG = { iceServers: [{ urls: 'stun:stun.l.google.com:19302' }] };
const TOTAL_ROUNDS = 5;

// ─── State ────────────────────────────────────────────────
let socket, pc, localStream;
let roomCode = null;
let isHost = false;
let timerInterval = null;
let timerSecs = 0;
let scoreYou = 0, scoreThem = 0, round = 1;
let roundActive = false;

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
  if (isHost) {
    // Host creates the WebRTC offer
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
      beginRoundCountdown();
      break;

    case 'answer':
      await pc.setRemoteDescription(new RTCSessionDescription(data));
      break;

    case 'ice':
      if (data) await pc.addIceCandidate(new RTCIceCandidate(data));
      break;

    case 'i_laughed':
      endRound(true); // opponent laughed, I win
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

  // Add local tracks
  if (localStream) {
    localStream.getTracks().forEach(track => pc.addTrack(track, localStream));
  }

  // Receive remote video
  pc.ontrack = (e) => {
    const remoteVideo = document.getElementById('video-remote');
    remoteVideo.srcObject = e.streams[0];
    remoteVideo.onloadedmetadata = () => {
      const overlay = document.getElementById('overlay-remote');
      overlay.style.opacity = '0';
      setTimeout(() => overlay.style.display = 'none', 300);
    };
  };

  // Send ICE candidates
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
    vid.onloadedmetadata = () => {
      document.getElementById('overlay-local').style.opacity = '0';
      setTimeout(() => document.getElementById('overlay-local').style.display = 'none', 300);
    };
  } catch (e) {
    console.warn('Camera unavailable:', e);
  }
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
  document.getElementById('laugh-btn').disabled = true;
  document.getElementById('round-num').textContent = round;
  stopTimer();
  timerSecs = 0;
  updateTimerDisplay();

  let countdown = 3;
  setStatus('get ready...');

  const cd = setInterval(() => {
    if (countdown > 0) {
      setStatus(countdown + '...');
      countdown--;
    } else {
      clearInterval(cd);
      setStatus('😐 keep a straight face!', true);
      roundActive = true;
      document.getElementById('laugh-btn').disabled = false;
      startTimer();
    }
  }, 1000);
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

// ─── Screens ──────────────────────────────────────────────
function showScreen(id) {
  document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
  document.getElementById('screen-' + id).classList.add('active');
}

// ─── Cleanup ──────────────────────────────────────────────
function cleanup() {
  stopTimer();
  if (localStream) { localStream.getTracks().forEach(t => t.stop()); localStream = null; }
  if (pc) { pc.close(); pc = null; }
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

// Spacebar shortcut
document.addEventListener('keydown', e => {
  if (e.code === 'Space' && document.getElementById('screen-game').classList.contains('active')) {
    e.preventDefault();
    if (!document.getElementById('laugh-btn').disabled) iLaughed();
  }
});