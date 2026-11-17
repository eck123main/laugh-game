# Laugh Game

A multiplayer "try not to laugh" video game: players join a room over WebRTC,
and whoever laughs first is eliminated. Laughter is detected in the browser
from the webcam (face) and the microphone (audio).

## Run

```
npm install
npm start          # http://localhost:3000 (server/index.js, serves client/)
npm test           # unit tests (node --test)
```

## How laugh detection works

| Stage | Where | What |
|---|---|---|
| Mic capture | `client/mic-worklet.js` | AudioWorklet forwards mic samples (falls back to ScriptProcessor) |
| Audio features | `client/laugh-audio.js` (`Resampler`, `LogMelStream`) | 8 kHz, 128-band Slaney log-mel, 44-frame (~1.26 s) windows, librosa-compatible |
| Audio model | `client/models/laughter_resnet.onnx` | Gillick et al. 2021 ResNet ([jrgillick/laughter-detection](https://github.com/jrgillick/laughter-detection)), run by onnxruntime-web every ~93 ms |
| Face features | `client/laugh-face.js` | MediaPipe FaceMesh landmarks → mouth open (AU25/26), smile width/corner lift (AU12), eye narrowing (AU6), normalised by inter-ocular distance and scored against a per-round neutral calibration. `CuePersistence` keeps smile/eye only when held for ~350 ms, so spread vowels and blinks don't count |
| Fusion | `fusedGateScore` in `client/laugh-audio.js`, called from `onFaceResults` | `0.55·face + 0.45·audio ≥ 0.62`, or face alone ≥ 0.80, or (ResNet only) audio alone ≥ 0.88 while a face is visible / ≥ 0.75 when none is; audio older than 500 ms counts as 0. The bar under the cams shows the trigger gate's progress during a round |
| Far-end guard | `FarEndGuard` | while opponents' audio is playing, audio is down-weighted (their laugh may leak past echo cancellation) |
| Trigger | `LaughGate` | hysteresis + leaky evidence: fires after ~850 ms of evidence and tolerates the dips between "ha – ha – ha" syllables |

Debugging: run `localStorage.ogiggleDebug = 1` in the browser console and
reload. The detector scores are then logged about 4 times a second.
To A/B test the mic processing, set `localStorage.ogiggleRawMic = 1`. This
turns off noise suppression and AGC; echo cancellation stays on.

Offline audio evaluation: put WAV clips under `clips/laugh/` and other
directories (e.g. `clips/speech/`), then run
`npm i --no-save onnxruntime-node` once, followed by
`node scripts/eval-audio.js clips`. The script replays each clip through the
same pipeline and sweeps the audio-alone threshold.
