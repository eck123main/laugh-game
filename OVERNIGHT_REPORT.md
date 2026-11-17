# Overnight report

## Round 1 — 2026-11-18

### ⚠️ Not tested and not committed

In this session every shell command was auto-denied: `node --test`, `git status`
and `git commit`, from both Bash and PowerShell. So **none of the changes below
have been run or committed.** They are uncommitted edits in the working tree. To
verify and commit:

```
npm test
git add -A && git commit -m "feat(laugh): hysteresis/evidence trigger gate, floored dB reference, npm test"
```

I traced each new test by hand against the code. Still, run `npm test` before
you trust them.

### Starting point

The previous session left a WIP commit (`accaa25`). It replaced the heuristic
audio detector with the Gillick et al. ResNet (ONNX). That commit included a
streaming librosa-compatible log-mel front-end in `client/laugh-audio.js`, an
in-memory patch to the ONNX input shape, resampling to 8 kHz, an RMS gate and
EMA smoothing. It also had unit tests. I built on that work and didn't rewrite it.

### What changed and why

1. **New trigger gate: hysteresis plus leaky evidence** (`LaughGate` in
   `client/laugh-audio.js`, used in `client/game.js`).
   - The old rule was "score above threshold continuously for 850 ms; reset the
     timer on *any* frame below threshold". Laughter is bursty ("ha – ha – ha"),
     and the per-window score dips between syllables. Each dip restarted the
     timer, so real laughs were often missed or detected late. Meanwhile one
     long false positive still fired.
   - The new gate works like this:
     - Evidence builds up while the score is at or above the threshold.
     - It is held in a hysteresis band (85–100 % of the threshold).
     - Below that band it leaks away at half speed.
     - It fires once 850 ms of evidence has built up.
     - Time steps are capped at 250 ms, so a stalled or backgrounded tab can't
       count as sustained laughter.
   - This is a causal (live) version of Gillick's offline post-processing:
     zero-phase low-pass on the probabilities, then threshold 0.5, then a
     minimum duration of 0.2 s.
   - Both trigger paths now feed the same gate, each score normalised by its own
     threshold:
     - face + audio fused: `max(blended/0.62, face/0.80)`
     - audio alone when no face is visible: `audio/0.75`
   - In the hand-traced tests, a bursty laugh (300 ms on / 150 ms off) fires at
     about 1.55 s. Sparse 100 ms speech-like spikes never fire.
   - The old `laughStartTime` variable is gone.
2. **dB reference floor in `LogMelStream`.** The decaying peak used as the
   `ref=np.max` stand-in fell without limit, halving in power every second.
   After about 20 s of quiet it sank to the room-noise level. The model would
   then see background hiss normalised to 0 dB, i.e. "as loud as anything this
   session", which invites false positives. The model was trained with file-level
   `ref=np.max`, which keeps quiet noise far below the loudest event. The
   reference is now floored at `PEAK_FLOOR_DB = 60` (in the code's
   20·log10(power) units, ≈30 dB of real power) below the loudest frame of the
   session.
3. **`npm test` works.** It was `echo "Error: no test specified" && exit 1`; it
   is now `node --test tests/`.
4. **Tests added** in `tests/laugh-audio.test.js`:
   - seven `LaughGate` tests: sustained, sub-threshold, bursty, sparse spikes,
     hysteresis hold, dt cap, validation
   - one test for the dB-floor behaviour

### Research

- Gillick, Deng, Ryokai & Bamman, *Robust Laughter Detection in Noisy
  Environments* (Interspeech 2021), code at
  https://github.com/jrgillick/laughter-detection. I read
  [`models.py`](https://raw.githubusercontent.com/jrgillick/laughter-detection/master/models.py)
  (ResNetBigger ends in `torch.sigmoid`, so the ONNX output is already a
  probability and clamping is enough),
  [`laugh_segmenter.py`](https://raw.githubusercontent.com/jrgillick/laughter-detection/master/laugh_segmenter.py)
  (2nd-order Butterworth `filtfilt` with cutoff 0.01 × Nyquist; threshold
  > 0.5; min length > 0.2 s) and
  [`utils/data_loaders.py`](https://raw.githubusercontent.com/jrgillick/laughter-detection/master/utils/data_loaders.py)
  (features are computed over the whole file, so dB is relative to the file
  max; windows of 44 frames, stepped one frame at a time).
- Posterior smoothing of frame-level laughter scores consistently helps:
  Brückner & Schuller (Interspeech 2017),
  https://www.isca-archive.org/interspeech_2017/brueckner17_interspeech.pdf ;
  Gosztolya et al. (Interspeech 2015),
  https://www.isca-archive.org/interspeech_2015/gosztolya15b_interspeech.pdf .
- Real-time on-device laughter detection with a DNN: Hagerer et al. (Interspeech
  2017), https://www.isca-archive.org/interspeech_2017/hagerer17_interspeech.html .
- Real-time audiovisual (face + audio) laughter detection, which supports the
  fusion approach already in the game:
  https://research.hub.ku.edu.tr/entities/publication/772166fe-b4b1-4af4-89e3-f14366534841/full

### Ideas for next time

- **Run and commit this round's changes first** (see the top of this section).
- Put the `node --test` and `git` commands on the session allowlist so later
  rounds can test and commit.
- Collect a few labelled recordings (laughing, talking, silence, game audio
  leaking from speakers). Add an offline eval script that streams them through
  `LogMelStream` + ONNX (`onnxruntime-node`), then tune `AUDIO_SOLO_THRESHOLD`,
  `GATE_OFF_RATIO`, `decayRate` and `PEAK_FLOOR_DB` against those recordings
  instead of guessing.
- Move the `ScriptProcessorNode` (deprecated, runs on the main thread) to an
  `AudioWorklet`, so the face-mesh loop can't starve audio capture.
- Reuse the `getWindow()` output buffer instead of allocating every ~93 ms.
- Face side: replace the hand-tuned FaceMesh landmark ratios with MediaPipe
  Face Landmarker blendshapes (`mouthSmileLeft/Right`, `jawOpen`,
  `cheekSquintLeft/Right`), which track laughing faces more directly.
- Add a second audio opinion, e.g. YAMNet's AudioSet laughter classes
  (Laughter, Giggle, Chuckle/chortle, Belly laugh, Snicker), and ensemble it
  with the ResNet.
- Show `laughGate.progress()` in the UI as a "you're about to crack" meter.
- Remove or put behind a flag the random 5 % debug `console.log` in
  `onFaceResults`.

## Round 2 — 2026-11-18 (second session, labelled "round 1" in the prompt)

### Finishing the previous round
- `npm test` worked here (it was blocked last time). The round-1 script
  `node --test tests/` failed on Node 24, which treats the directory as a
  module path. It is now `node --test "tests/**/*.test.js"`.
- One round-1 test was wrong: `LogMelStream window is time-major…`. Its 10
  hops of "silence" (1860 samples) are shorter than N_FFT (2048), so the
  newest frame still held a cut-off tone tail. That tail leaks energy across
  the whole spectrum, and the frame mean is dominated by the −80 dB floor
  anyway. The test now pushes 20 hops and compares per-frame peaks.
- Then I committed round 1's work as `dd8ebb1`.

### What changed and why (all committed; tests went from 21 to 41, all pass)
1. **Facial laugh score based on FACS, independent of face size**
   (`client/laugh-face.js`, new, `188db8b`, `ec2799b`).
   - The old score was `0.8·mouthOpen + 0.1·cheek + 0.1·eye`. A wide-open
     mouth alone (talking, yawning, surprise) scored 0.8, which was enough
     for the face-only trigger. The "cheek" cue (corner y − cheek y) also
     has the wrong sign for a smile: smiling lifts the corners, so the
     value goes down and it clamped to 0.
   - Eye and cheek were raw image coordinates, so they changed with
     distance from the camera.
   - The landmark x/y values were not corrected for the 4:3 aspect ratio.
   - New cues follow the FACS amusement face:
     - AU25/26: mouth opening
     - AU12: mouth width ÷ inter-ocular distance (IOD), and corner lift
       relative to the lip centre
     - AU6: eye aperture ÷ IOD
   - Every cue is scored against the player's own trimmed-mean calibration.
     Weights are open 0.45, smile 0.40, eye 0.15, so mouth opening alone
     tops out at 0.45 and can no longer fire the face-only trigger (0.80).
   - The mouth baseline is clamped (≤ 0.08). A player who laughs through
     calibration would otherwise get an open-mouth "neutral" baseline.
   - The mouth scale was converted (0.22 → 0.165) to keep the old
     sensitivity after the aspect fix.
2. **Mic capture through an AudioWorklet** (`client/mic-worklet.js`,
   `c979aaa`).
   - `ScriptProcessorNode` is deprecated and runs on the main thread, where
     FaceMesh inference can starve it and drop buffers, leaving gaps in the
     log-mel stream.
   - The worklet batches 128-sample blocks into 1024-sample chunks and posts
     them to the main thread. Queued chunks are never lost.
   - Falls back to ScriptProcessor if worklets are unavailable.
3. **Far-end guard** (`FarEndGuard` in `laugh-audio.js`, `8338a6f`).
   - In this game the opponents are trying to make you laugh, and their
     laughter plays through your speakers. Echo cancellation leaks on laptop
     mics and loud speakers.
   - Remote players' audio is now metered. While it is loud (plus a 700 ms
     hold for the echo tail):
     - the audio-only threshold rises from 0.75 to 0.92
     - audio counts at 60 % in the face + audio fusion
4. **Debug logging is now opt-in.** The random 5 % `console.log` only runs
   when `localStorage.ogiggleDebug = 1` is set. It logs about 4 times a second
   and includes the new cue scores, gate progress and far-end level.
5. **Tests**: new `tests/laugh-face.test.js` (11) and
   `tests/mic-worklet.test.js` (2). New `tests/client-syntax.test.js` checks
   that every browser script parses and that `index.html` loads scripts in
   the right order. Also new `FarEndGuard`/`rmsOf` tests.
6. **README** now describes the detection pipeline (`30e5ffb`).

### Not verified
- Nothing was run in a real browser. Direct `node`/`node --check`
  commands were blocked in this session; only `npm test` was allowed. Please
  check by hand:
  - the console shows `ResNet ONNX mode (AudioWorklet)`
  - the calibration log shows width/lift/eye values
  - laughing triggers, but talking with your mouth wide open does not
- The face weights and scales (`SCALE`, `WEIGHTS` in `laugh-face.js`) and
  the far-end numbers (`rmsOn 0.02`, 0.92, 0.6) are reasoned defaults, not
  fitted to data.

### Research
- Petridis & Pantic, audiovisual laughter vs. speech: facial features are
  among the strongest cues, and adding video to audio helps:
  https://www.ibug.doc.ic.ac.uk/media/uploads/documents/petridispantic_2011_tmm.pdf ,
  https://www.ibug.doc.ic.ac.uk/media/uploads/documents/petridislevequepantic_acii2013.pdf ,
  https://amlap.coli.uni-saarland.de/conf/laughter-09/presentations/Petridis_Pantic.pdf
- FACS for laughter: AU6 + AU12 + AU25/26. AU25/26 also appear in speech
  and surprise, so they can't decide on their own:
  https://imotions.com/blog/learning/research-fundamentals/duchenne-smile/ ,
  https://pmc.ncbi.nlm.nih.gov/articles/PMC4235269
- Normalising landmark distances by inter-ocular distance, and using
  landmarks 61/291 for mouth width:
  https://docs.sunfounder.com/projects/ai-lab-kit/en/latest/mediapipe/mp_2_face_emotion.html ,
  survey https://arxiv.org/pdf/1805.05563

### Ideas for next time
- **Real-browser check** of this round (see "Not verified"), then tune
  `laugh-face.js` `SCALE`/`WEIGHTS` from the debug log on a few people.
- Offline eval harness (still open from round 1): labelled clips → `LogMelStream`
  + onnxruntime-node; also record FaceMesh landmark traces to replay through
  `LaughFace.scoreFace`, so the face side can be tuned offline too.
- Speech vs. laugh on the face: speech moves the mouth quickly and with
  little smile, while laughter keeps the smile up. A cue that tracks how
  much mouth-opening varies over ~0.5 s could down-weight talking further.
- Try `noiseSuppression: false` on a second mic track used only for
  detection. Speech-tuned noise suppression can weaken non-speech sounds like
  laughter.
- Switch to MediaPipe Face Landmarker blendshapes (`mouthSmileLeft/Right`,
  `cheekSquintLeft/Right`, `jawOpen`). These map almost directly onto the
  new AU cues.
- YAMNet laughter classes as a second audio opinion (ensemble).
- Show `laughGate.progress()` as an "about to crack" meter instead of the
  mouth-only bar.
- Reuse the `getWindow()` buffer (an allocation every ~93 ms).

## Round 3 — 2026-11-18 (prompt: "round 2 of tonight")

### Finishing the interrupted run
Two face commits (`ce9a78c` 3D landmark distances, `606fca7` calibration
finishes at round start) had landed after the round-2 report. Changes to
`laugh-audio.js`/`game.js` were left uncommitted. I added tests for them and
committed them as `f939627`:
- **Silence gate on the loudest frame** (`LogMelStream.peakRms()`). Before,
  a short soft "ha" in an otherwise quiet 1.26 s window was diluted ~5× by
  the window-mean RMS. It fell under `AUDIO_RMS_GATE` and never reached the
  model.
- **`getWindow()` reuses its output buffer.** This was an open idea from
  rounds 1 and 2. It is safe because `inferenceBusy` prevents overlapping
  `session.run()` calls.

### What changed and why (all committed; tests went from 45 to 50, all pass)
1. **Confident audio can fire while a face is visible** (`fusedGateScore` in
   `laugh-audio.js`, `1359380`).
   - With a face visible, the only routes were the blend and face-alone. A
     fully confident ResNet score (1.0) with a neutral or barely moving face
     gave `0.45 / 0.62 ≈ 0.73` of the trigger, so it could **never** fire.
     This happens when someone laughs behind a hand, or after a weak
     calibration.
   - The same laugh *did* fire as soon as FaceMesh lost the face. That was
     inconsistent and a real source of missed laughs.
   - New third route: audio alone ≥ 0.88 (stricter than the 0.75 used when
     no face is visible). It applies to the neural model only; the heuristic
     fallback stays disabled for this route.
   - While opponents are audible, far-end scaling (×0.6) keeps this route
     below 1.0, so leaked opponent laughter still can't fire it.
   - Audio estimates older than 500 ms now count as 0. A stalled audio
     pipeline (suspended context, hung inference) used to leave its last,
     possibly high, value feeding the fusion forever.
2. **Crack meter** (`0db4b60`). During a round, the bar under the cams now
   shows `laughGate.progress()`, which is what actually decides a trigger.
   Before, it showed mouth opening only, which no longer drives triggering.
   Outside rounds (including calibration) it still shows mouth opening.
3. README fusion row updated.

### Not verified
- Nothing was run in a browser; only `npm test`. Please check:
  - the bar label switches to "crack meter" when a round starts
  - laughing with a hand over your mouth now triggers
  - opponents laughing through your speakers still doesn't trigger
- The 0.88 threshold is a reasoned default. If speech falsely triggers with a
  face visible, raise `AUDIO_SOLO_THRESHOLD_FACE_VISIBLE` in `game.js`.

### Research
- Frame-wise (non-presegmented) laughter detection with MFCC, autocorrelation
  peak and F0 features reached 7.9 % EER on ICSI. Voicing and periodicity
  are what separate laughter from speech:
  Knox & Mirghafori, Interspeech 2007,
  https://www.isca-archive.org/interspeech_2007/knox07_interspeech.pdf
- Laughter differs from speech in pitch and in its alternating
  voiced/unvoiced pattern:
  Truong & van Leeuwen, https://www.coli.uni-saarland.de/conf/laughter-07/files/TRUONG_LEEUWEN.pdf
- Modulation spectrum (syllable rate) and MFCC + SVM for meeting laughter:
  Kennedy & Ellis, https://www.ee.columbia.edu/~lyndon/pubs/nistrt2004-laughter.pdf ;
  Krikke & Truong, https://www.isca-archive.org/interspeech_2013/krikke13_interspeech.pdf
- Audiovisual laughter: cepstral audio features and facial expression are
  the strongest cues, and pitch/energy alone are weak. This supports keeping
  an audio-alone route next to the face:
  https://amlap.coli.uni-saarland.de/conf/laughter-09/presentations/Petridis_Pantic.pdf
- Browser audio processing: noise suppression targets non-speech sounds, so
  it may attenuate laughter. Chrome's AGC compresses loud bursts. No study
  measures the effect on laughter, so it has to be tested:
  https://blog.addpipe.com/audio-constraints-getusermedia/ ,
  https://developer.chrome.google.cn/blog/disabling-hardware-noise-suppression?hl=en ,
  https://developer.mozilla.org/en-US/docs/Web/API/MediaTrackSettings/noiseSuppression

### Ideas for next time
- **A/B test the mic constraints.** Record the same laughs with
  `noiseSuppression`/`autoGainControl` on and off, and compare ResNet
  probabilities. If off is clearly better, request a second detection-only
  track. Check `track.getSettings()`, because Chrome may share processing
  per device.
- **Speech-vs-laugh audio cue.** A cheap autocorrelation-based
  voiced/unvoiced alternation or a 3–6 Hz energy-modulation score on the
  8 kHz stream could down-weight speech before the gate (Knox 2007, Truong).
  This should be tuned against recordings.
- Offline eval harness: still open, and still the main blocker for tuning any
  threshold with data instead of reasoning.
- MediaPipe Face Landmarker blendshapes; YAMNet as a second audio opinion.

## Round 4 — 2026-11-18 (prompt: "round 3 of tonight")

### Starting state
The working tree was clean; round 3 had finished and committed everything.

### What changed and why (all committed; tests went from 50 to 60, all pass)
1. **Persistence filter on the smile and squint cues** (`CuePersistence` in
   `client/laugh-face.js`, used in `onFaceResults`, commit `1b265ea`).
   - Two face cues spiked on non-laugh events:
     - Spread vowels ("ee", "eh") widen the mouth for ~100–200 ms, which
       fires the AU12 width cue.
     - A blink closes the eye for ~100–150 ms, which reads as a full AU6
       squint.
   - Every spike added evidence to the leaky `LaughGate` at full rate and
     leaked at half rate. So a talking or blinking face slowly filled the
     gate, especially with moderate audio during speech.
   - Smile and eye now use the 25th percentile over a 350 ms window. Spikes
     shorter than ~260 ms are dropped. A held laughing smile passes with
     ~260 ms onset delay.
   - Right after a reset or a face-lost gap, smile and eye report 0 until
     the window has filled, so a spike on the first frames can't pass alone.
   - Mouth opening stays instantaneous. It is already capped at 0.45, and
     "ha-ha" opening oscillates in real laughs too.
   - The debug log now prints each cue as `filtered/raw`.
   - The new tests cover: vowel spikes, blinks, onset delay, gaps at 15 fps,
     and an end-to-end check that a talking-and-blinking face never fires
     the face-only route while a held laugh face does.
2. **`ogiggleRawMic` flag** (`ab7e2c4`). Set `localStorage.ogiggleRawMic = 1`
   to turn off noise suppression and AGC; echo cancellation stays on. The
   settings the browser actually applied are logged (also in debug mode).
   This enables the A/B test that round 3 proposed. The defaults are
   unchanged, because the mic stream also goes to peers.
3. **Offline audio eval harness** (`scripts/eval-audio.js`, `58fa247`).
   - This has been the main blocker since round 1.
   - It replays WAV clips through the same Resampler → LogMelStream → RMS
     gate → ResNet → ProbSmoother → LaughGate chain, with `game.js`
     constants and 1024-sample chunks.
   - Labels come from the folder: `laugh/` or `laughter/` should fire,
     anything else should not.
   - It prints per-file results and sweeps audio-alone thresholds from
     0.5 to 0.95.
   - It needs `npm i --no-save onnxruntime-node`, which is deliberately not a
     dependency.
   - Tested with a fake model: WAV parsing, the fire time on a confident
     clip, silence never reaching the model, ~93 ms cadence, and the sweep.
4. README updated (persistence filter, raw-mic flag, eval usage).

### Not verified
- `npm i onnxruntime-node` was blocked in this session, so **the harness
  has not run the real model**. Run it once on any WAV to confirm the ONNX
  path. The ONNX code mirrors `loadOnnxModel` and `runOnnxInference` in
  `game.js`.
- Nothing was run in a browser. Please check that a held smile still raises
  the crack meter, and that talking or blinking moves it less than before.
- The 350 ms window and 0.25 quantile are reasoned defaults, not fitted to
  data.

### Research
- Petridis & Pantic found that facial geometry during laughter differs
  clearly from speech, and that adding temporal information beats static
  features:
  https://ibug.doc.ic.ac.uk/media/uploads/documents/ICASSP-2008-PetridisPantic-FINAL-CAMERA.pdf ,
  https://ibug.doc.ic.ac.uk/media/uploads/documents/petridismartinezpantic_imavis.pdf ,
  https://Ibug.Doc.Ic.Ac.Uk/media/uploads/documents/ICMI-2009-PetridisEtAl-CAMERA.pdf
- Audio–visual correlation differs between speech and laughter (Petridis et
  al., FG 2011): https://Ibug.Doc.Ic.Ac.Uk/media/uploads/documents/fg2011.pdf
- MAHNOB laughter database, a source of labelled audiovisual laughter and
  speech for tuning:
  https://www.ibug.doc.ic.ac.uk/media/uploads/documents/mahnob_laughter_db.pdf

### Ideas for next time
- **Record 20–30 clips and run `scripts/eval-audio.js`.** Include laughs,
  talking, silence and opponents' laughter played through speakers. Then
  tune `AUDIO_SOLO_THRESHOLD`, `AUDIO_SOLO_THRESHOLD_FACE_VISIBLE` and
  `AUDIO_RMS_GATE` from the sweep, and run each clip with and without
  `ogiggleRawMic`.
- Extend the harness to the face: log `getFaceFeatures` per frame (with
  debug on) to JSON, then replay through `scoreFace` + `CuePersistence` +
  `fusedGateScore` alongside the audio trace.
- Audio–visual correlation cue (FG 2011 above). In speech, mouth opening
  tracks the audio envelope syllable by syllable. That coupling looks
  different in laughter, and it could down-weight talking.
- Still open: MediaPipe Face Landmarker blendshapes; YAMNet as a second
  audio opinion.

## Round 5 — 2026-11-18 (prompt: "round 4 of tonight")

### Starting state
The working tree was clean and round 4 had committed everything.

### ⚠️ Tests were written but NOT run
In this session every `node`/`npm` command was denied automatically (no
approval surface), so I could not run the suite. I checked the new tests by
hand-computing the expected values, but **please run `npm test` first**.
If `tests/burst-rhythm.test.js` fails, the most likely culprit is the
"irregular bursts" test. It uses a synthetic signal, and its margin (0.2)
was estimated, not measured.

### What changed and why
1. **Burst-rhythm cue** (`BurstRhythm` in `client/laugh-audio.js`, wired in
   `game.js`, commit `bbd6883`).
   - Earlier rounds listed a speech-vs-laugh audio cue as the top open idea.
     Laughter is a train of near-identical calls at ~4.4 calls/s
     (Bachorowski et al. 2001), which is a period of ~230 ms. Speech
     syllables vary in length and loudness. Coughs and slams are a single
     burst, and fans are flat.
   - The ResNet sees 256 ms FFT frames, too coarse to resolve 100 ms calls,
     so this cue is independent information.
   - How it works:
     - It builds a 10 ms dB energy envelope over the last 1.5 s, with the
       floor clipped 30 dB below the peak.
     - It takes the normalised autocorrelation at lags 120–400 ms.
     - A lag only counts if it is a local maximum and the envelope
       decorrelated (r < 0) at a shorter lag. Otherwise one long burst or a
       held vowel would look "periodic" at short lags.
     - A sound with under 1 dB of envelope variation counts as steady noise
       and scores 0.
   - The model probability is multiplied by `0.9 + 0.1·min(r/0.45, 1)`
     before smoothing. The cue is deliberately bounded: it can cost a
     non-rhythmic sound at most 10%, so it trims borderline false positives
     without vetoing a real laugh.
   - `localStorage.ogiggleNoRhythm = 1` turns it off for A/B testing. The
     debug log prints `rhythm:`.
2. **Eval harness**: it applies the same scaling, has a `--no-rhythm` flag,
   and prints each file's median regularity. Recordings will show whether
   laughs and speech actually separate on this cue, and how far `minScale`
   and `full` can be pushed.
3. Tests: `tests/burst-rhythm.test.js` (regular vs irregular bursts, steady
   noise, single cough, reset) and an eval test showing that a borderline
   0.8 score on a flat tone no longer fires.
4. README: a pipeline-table row and a note on the A/B flag.

### Research
- Truong & van Leeuwen 2007, "Automatic discrimination between laughter and
  speech": laughter differs from speech in mean pitch and in the ratio of
  unvoiced to voiced durations, and combining prosodic and spectral features
  helps. https://data.zhenzhi.com/details/1/907008030
- Slaney (Microsoft Research), "Characteristic contours of syllabic-level
  units in laughter": the shape of the intensity envelope at the syllable
  level is a key cue. https://www.microsoft.com/en-us/research/?p=165315
- Ludusan & Wagner, Speech Prosody 2024: rhythm is the one prosodic
  dimension that does not differ between laughter types, so it is a stable
  laughter cue.
  https://www.isca-archive.org/speechprosody_2024/ludusan24_speechprosody.html
- Bachorowski, Smoski & Owren 2001, "The acoustic features of human
  laughter" (JASA): 1024 bouts. The rate of ~4.37 calls/s is cited via
  https://www.osu.tests.sfulib4.publicknowledgeproject.org/index.php/EMR/article/view/5608/5176 ;
  abstract at https://scholars.duke.edu/publication/706959
- Oh et al., Interspeech 2013:
  https://www.isca-archive.org/interspeech_2013/oh13_interspeech.html

### Ideas for next time
- Run `npm test`. Then run `node scripts/eval-audio.js clips` with and
  without `--no-rhythm`. If laughs and speech separate well on median
  regularity, lower `minScale` (e.g. to 0.75) and fit `full`.
- Unvoiced/voiced duration ratio (Truong 2007): a per-10 ms voicing flag
  from zero-crossing rate or spectral flatness, added to `BurstRhythm`.
- Pitch cue: laughter has a higher and wider f0. A cheap autocorrelation
  pitch tracker on the 8 kHz stream would provide it.
- Still open: the face replay harness, the audio–visual correlation cue,
  MediaPipe blendshapes, and YAMNet.

## Round 6 — 2026-11-18 (prompt: "round 5 of tonight")

### Starting state
The working tree was clean. Last round's tests had been written but never
run, so I ran `npm test` first: all 65 passed, including
`tests/burst-rhythm.test.js`. Nothing needed fixing.

### What changed and why (all committed; tests went from 65 to 72, all pass)
1. **Voicing cue** (`VoicingCue` in `client/laugh-audio.js`, wired in
   `game.js`, commit `b44aa03`).
   - Last round's top idea was Truong & van Leeuwen's unvoiced/voiced
     duration ratio. Laughter alternates voiced vowels with breathy,
     unvoiced "h" aspiration, and many laughs are unvoiced grunts or snorts.
     Sustained voiced sounds don't do that: humming, singing, a held vowel,
     music, mains hum. The burst-rhythm cue can't catch them when they are
     steady, or when they pulse with a beat.
   - How it works:
     - Every 10 ms, a 40 ms frame (DC removed) gets a normalised
       autocorrelation over lags for 75–615 Hz. The block counts as voiced
       if the peak is ≥ 0.5, which is close to Praat's 0.45 voicing
       threshold.
     - An octave guard takes the shortest-lag local peak that is within 90%
       of the best one.
     - Over the last 1.5 s it counts only "active" blocks (within 25 dB of
       the loudest), so silent gaps between calls count neither way. At
       least 20 active blocks are needed; otherwise the cue is neutral.
     - The scale is `0.9 + 0.1·min(unvoicedRatio/0.2, 1)`, multiplied into
       the model probability together with the rhythm scale.
   - It is bounded at −10% like the rhythm cue, so it trims borderline false
     positives and can't veto a confident laugh. The worst case for a real
     laugh, with both cues low, is ×0.81.
   - A/B flag: `localStorage.ogiggleNoVoicing = 1`. The debug log prints
     `voicing:`.
   - It also reports the f0 median and spread (in semitones) for
     diagnostics, but **does not** use pitch as a cue yet. Laughter's f0 is
     higher and more variable than speech, but absolute ranges overlap
     across speakers (a male laugh is around 270 Hz, female speech around
     220 Hz). A per-speaker baseline is needed first.
2. **Eval harness**: it applies the voicing scale, has a `--no-voicing`
   flag, and prints each file's median unvoiced ratio and `f0=…Hz±…st`.
   Recordings will show whether laughs and speech separate on these.
3. Tests: `tests/voicing-cue.test.js` (6 tests: a held voiced tone, an
   octave-error guard with a strong 2nd harmonic, noise, "h-a" laugh calls,
   voiced calls with silent gaps, too little sound and reset). There is also
   a new eval test: a borderline 0.8 on a steady 220 Hz hum no longer fires.
   I changed the existing rhythm eval test to pass `voicing: false`, so it
   still tests rhythm alone.
4. README: a pipeline-table row and the A/B flag.

### Known trade-off / not verified
- A *voiced* laugh separated by glottal stops (silence, no "h") reads as
  fully voiced and loses up to 10% (there is a test documenting this). Its
  rhythm is usually regular, so the rhythm cue stays at 1.
- Browser noise suppression might attenuate the breathy "h" sounds and make
  laughs look more voiced. Compare `unvoiced(median)` on the same clips
  with and without `ogiggleRawMic`. If NS hurts, raise `full` or lower
  `voicedR`.
- None of this has been tried on real recordings. The thresholds (`voicedR`
  0.5, `activeDb` 25, `full` 0.2) come from the literature and synthetic
  tests.
- Some shell commands (python, piped `node --test`) were auto-denied in this
  session. `npm test` worked, so every change was tested.

### Research
- Truong & van Leeuwen, "Evaluating automatic laughter segmentation in
  meetings using acoustic and acoustic-phonetic features" (Interdisciplinary
  Workshop on the Phonetics of Laughter, 2007): the fraction of unvoiced
  frames is a feature, motivated by laughter's alternating voicing.
  https://www.coli.uni-saarland.de/conf/laughter-07/files/TRUONG_LEEUWEN.pdf
- Petridis & Pantic, laughter vs speech: their prosodic features are the
  mean/std of pitch and energy plus the unvoiced ratio, over 320 ms windows.
  https://amlap.coli.uni-saarland.de/conf/laughter-09/files/Petridis_Pantic.pdf
  and ICME 2009:
  https://Ibug.Doc.Ic.Ac.Uk/media/uploads/documents/ICME2009-PetridisPantic-CAMERA.pdf
- Szameitat et al. 2007: laughter f0 is higher than in speech.
  https://www.coli.uni-saarland.de/conf/laughter-07/files/SZAMEITAT_ET_AL.pdf
- "Analysis of laugh signals for detecting in continuous speech": pitch
  ranges overlap between speech and laughter, but pitch *variation* in
  voiced laughter is larger. This is why pitch is diagnostic-only for now.
  https://typeset.io/pdf/analysis-of-laugh-signals-for-detecting-in-continuous-speech-1yisj45gxh.pdf
- "On Laughter and Speech-Laugh" (child-robot interaction, 2019): voicing
  probability and band-energy modulation capture laughter's repeated
  voiced/unvoiced change. https://arxiv.org/pdf/1908.11593
- Bachorowski et al. 2001 (cited last round): many laugh bouts are unvoiced
  (grunt- or snort-like). That is why the cue only penalises sounds that
  are *too voiced* and never penalises unvoiced sounds.

### Ideas for next time
- **Recordings first.** Run `node scripts/eval-audio.js clips` with
  combinations of `--no-rhythm` and `--no-voicing`, and read the per-file
  `rhythm(median)` and `unvoiced(median)` columns. If they separate laughs
  from speech, widen the cues (`minScale` 0.75–0.8) and fit `full`.
- Pitch relative to the speaker: keep a running median f0 from voiced
  blocks while the model says "not laughing" (p < 0.2). Then treat voiced
  f0 well above that baseline (≥ +5 semitones) or a large f0 spread as
  laugh evidence. `VoicingCue.stats()` already provides the inputs.
- Move `BurstRhythm`/`VoicingCue` into the AudioWorklet if main-thread time
  ever matters. They currently cost roughly 10 MFLOP/s together.
- Still open: the face replay harness, the audio–visual correlation cue,
  MediaPipe blendshapes, and YAMNet as a second opinion.
