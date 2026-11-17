// ─── Laughter audio front-end ───────────────────────────────────────────────
// Feature extraction + helpers for the Gillick et al. (Interspeech 2021)
// "resnet_with_augmentation" laughter model (client/models/laughter_resnet.onnx).
//
// The model was trained on features from
//   librosa.feature.melspectrogram(y, sr=8000, hop_length=186)   (n_fft=2048,
//   n_mels=128, power=2, Slaney mel scale + Slaney norm, fmin=0, fmax=sr/2)
//   .T  →  librosa.amplitude_to_db(S, ref=np.max)                 (top_db=80)
// sliced into windows of 44 frames (~1 s) and fed as [batch, 1, 44 time, 128 mel].
// See https://github.com/jrgillick/laughter-detection (configs.py, utils/audio_utils.py,
// utils/data_loaders.py, models.py).
//
// Works both in the browser (window.LaughAudio) and in Node (module.exports) so
// it can be unit-tested with `node --test`.
(function (root) {
  'use strict';

  const SR       = 8000;
  const N_FFT    = 2048;
  const HOP      = 186;
  const N_MELS   = 128;
  const N_FRAMES = 44;
  const N_BINS   = N_FFT / 2 + 1;
  const AMIN     = 1e-5;
  const TOP_DB   = 80;
  // Max distance (in powerToDb's 20·log10(power) units, i.e. ≈30 dB of real
  // power) the streaming dB reference may fall below the session's loudest frame.
  const PEAK_FLOOR_DB = 60;
  // Samples covered by one model window (no centre padding when streaming).
  const WINDOW_SAMPLES = (N_FRAMES - 1) * HOP + N_FFT;

  // ── Slaney mel scale (librosa default, htk=False) ──────────────────────────
  const F_SP = 200 / 3;
  const MIN_LOG_HZ = 1000;
  const MIN_LOG_MEL = MIN_LOG_HZ / F_SP;          // 15
  const LOGSTEP = Math.log(6.4) / 27;

  function hzToMelSlaney(hz) {
    return hz >= MIN_LOG_HZ
      ? MIN_LOG_MEL + Math.log(hz / MIN_LOG_HZ) / LOGSTEP
      : hz / F_SP;
  }
  function melToHzSlaney(mel) {
    return mel >= MIN_LOG_MEL
      ? MIN_LOG_HZ * Math.exp(LOGSTEP * (mel - MIN_LOG_MEL))
      : F_SP * mel;
  }

  // librosa.filters.mel(sr, n_fft, n_mels, fmin=0, fmax=sr/2, htk=False, norm='slaney')
  // Returns Float32Array[nMels * nBins] (row = mel band).
  function melFilterbank(sr = SR, nFft = N_FFT, nMels = N_MELS, fMin = 0, fMax = sr / 2) {
    const nBins = Math.floor(nFft / 2) + 1;
    const fftFreqs = new Float64Array(nBins);
    for (let k = 0; k < nBins; k++) fftFreqs[k] = k * sr / nFft;

    const melMin = hzToMelSlaney(fMin), melMax = hzToMelSlaney(fMax);
    const melF = new Float64Array(nMels + 2);
    for (let i = 0; i < nMels + 2; i++) {
      melF[i] = melToHzSlaney(melMin + (melMax - melMin) * i / (nMels + 1));
    }

    const w = new Float32Array(nMels * nBins);
    for (let m = 0; m < nMels; m++) {
      const fdLo = melF[m + 1] - melF[m];
      const fdHi = melF[m + 2] - melF[m + 1];
      const enorm = 2 / (melF[m + 2] - melF[m]);
      for (let k = 0; k < nBins; k++) {
        const lower = (fftFreqs[k] - melF[m]) / fdLo;
        const upper = (melF[m + 2] - fftFreqs[k]) / fdHi;
        const v = Math.max(0, Math.min(lower, upper));
        w[m * nBins + k] = v * enorm;
      }
    }
    return w;
  }

  // Periodic Hann window (scipy.signal.get_window('hann', N, fftbins=True)).
  function hannPeriodic(n) {
    const w = new Float32Array(n);
    for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / n);
    return w;
  }

  // ── Radix-2 FFT → power spectrum ───────────────────────────────────────────
  function makeFft(n) {
    if (n & (n - 1)) throw new Error('FFT size must be a power of two');
    const levels = Math.log2(n);
    const rev = new Uint32Array(n);
    for (let i = 0; i < n; i++) {
      let r = 0;
      for (let b = 0; b < levels; b++) r |= ((i >> b) & 1) << (levels - 1 - b);
      rev[i] = r;
    }
    const cos = new Float64Array(n / 2), sin = new Float64Array(n / 2);
    for (let i = 0; i < n / 2; i++) {
      cos[i] = Math.cos(2 * Math.PI * i / n);
      sin[i] = -Math.sin(2 * Math.PI * i / n);
    }
    const re = new Float64Array(n), im = new Float64Array(n);

    // Returns power |X[k]|^2 for k = 0..n/2 written into `out`.
    return function powerSpectrum(frame, out) {
      for (let i = 0; i < n; i++) { re[rev[i]] = frame[i]; im[rev[i]] = 0; }
      for (let size = 2; size <= n; size <<= 1) {
        const half = size >> 1, step = n / size;
        for (let start = 0; start < n; start += size) {
          for (let j = 0; j < half; j++) {
            const a = start + j, b = a + half;
            const c = cos[j * step], s = sin[j * step];
            const tr = re[b] * c - im[b] * s;
            const ti = re[b] * s + im[b] * c;
            re[b] = re[a] - tr; im[b] = im[a] - ti;
            re[a] += tr;        im[a] += ti;
          }
        }
      }
      const nOut = n / 2 + 1;
      out = out || new Float32Array(nOut);
      for (let k = 0; k < nOut; k++) out[k] = re[k] * re[k] + im[k] * im[k];
      return out;
    };
  }

  // ── Streaming log-mel extractor ────────────────────────────────────────────
  // push() 8 kHz samples; every HOP samples a new mel frame is computed. The
  // last N_FRAMES frames are kept and getWindow() returns the model input as a
  // Float32Array laid out [time(44) × mel(128)], in dB like amplitude_to_db.
  class LogMelStream {
    constructor() {
      this.melBasis = melFilterbank();
      this.window   = hannPeriodic(N_FFT);
      this.fft      = makeFft(N_FFT);
      this.samples  = new Float32Array(N_FFT);   // ring of last N_FFT samples
      this.total    = 0;                          // samples pushed so far
      this.nextFrameAt = N_FFT;
      this.frames   = new Float32Array(N_FRAMES * N_MELS); // ring of mel power frames
      this.frameMax = new Float32Array(N_FRAMES);
      this.frameEnergy = new Float32Array(N_FRAMES);       // mean-square of frame samples
      this.nFrames  = 0;                          // frames produced so far
      this._frameBuf = new Float32Array(N_FFT);
      this._power    = new Float32Array(N_BINS);
      // Slowly-decaying peak used as the dB reference. Gillick normalises each
      // file to its own max (ref=np.max); a decaying peak approximates that for
      // a live stream without letting one loud cough dominate forever.
      this.peak = 0;
      this.peakDecay = Math.pow(10, -(3 / 43) / 10); // power ×0.5 per second
      // The decaying peak must not sink to the room-noise level: after ~20 s
      // of quiet it would, and background hiss would then be normalised to
      // 0 dB like a loud laugh. File-level ref=np.max (as trained) keeps quiet
      // noise far below the loudest event, so floor the reference at
      // PEAK_FLOOR_DB (powerToDb units) below the loudest frame this session.
      this.sessionPeak = 0;
      this.peakFloorRatio = Math.pow(10, -PEAK_FLOOR_DB / 20);
    }

    reset() {
      this.total = 0; this.nextFrameAt = N_FFT; this.nFrames = 0;
      this.samples.fill(0); this.frames.fill(0); this.frameMax.fill(0);
      this.frameEnergy.fill(0); this.peak = 0; this.sessionPeak = 0;
    }

    push(input) {
      let produced = 0;
      for (let i = 0; i < input.length; i++) {
        this.samples[this.total % N_FFT] = input[i];
        this.total++;
        if (this.total === this.nextFrameAt) {
          this._computeFrame();
          this.nextFrameAt += HOP;
          produced++;
        }
      }
      return produced;
    }

    _computeFrame() {
      const start = this.total - N_FFT;           // oldest sample in ring
      let energy = 0;
      for (let i = 0; i < N_FFT; i++) {
        const s = this.samples[(start + i) % N_FFT];
        energy += s * s;
        this._frameBuf[i] = s * this.window[i];
      }
      const power = this.fft(this._frameBuf, this._power);
      const slot = this.nFrames % N_FRAMES;
      const base = slot * N_MELS;
      let mx = 0;
      for (let m = 0; m < N_MELS; m++) {
        let acc = 0;
        const off = m * N_BINS;
        for (let k = 0; k < N_BINS; k++) {
          const w = this.melBasis[off + k];
          if (w !== 0) acc += w * power[k];
        }
        this.frames[base + m] = acc;
        if (acc > mx) mx = acc;
      }
      this.frameMax[slot] = mx;
      this.frameEnergy[slot] = energy / N_FFT;
      if (mx > this.sessionPeak) this.sessionPeak = mx;
      this.peak = Math.max(this.peak * this.peakDecay, mx, this.sessionPeak * this.peakFloorRatio);
      this.nFrames++;
    }

    ready() { return this.nFrames >= N_FRAMES; }

    // RMS of the samples covered by the current window (approximate).
    rms() {
      const n = Math.min(this.nFrames, N_FRAMES);
      if (!n) return 0;
      let s = 0;
      for (let i = 0; i < n; i++) s += this.frameEnergy[i];
      return Math.sqrt(s / n);
    }

    getWindow(out) {
      if (!this.ready()) return null;
      out = out || new Float32Array(N_FRAMES * N_MELS);
      let winMax = 0;
      for (let i = 0; i < N_FRAMES; i++) if (this.frameMax[i] > winMax) winMax = this.frameMax[i];
      const ref = Math.max(winMax, this.peak);
      return powerToDb(this.frames, (this.nFrames % N_FRAMES), ref, out);
    }
  }

  // amplitude_to_db (20·log10, amin=1e-5, top_db=80) applied to a ring of
  // frames starting at `startSlot`, written in chronological order.
  function powerToDb(frames, startSlot, ref, out) {
    const refDb = 20 * Math.log10(Math.max(AMIN, ref));
    let maxDb = -Infinity;
    for (let t = 0; t < N_FRAMES; t++) {
      const src = ((startSlot + t) % N_FRAMES) * N_MELS;
      const dst = t * N_MELS;
      for (let m = 0; m < N_MELS; m++) {
        const db = 20 * Math.log10(Math.max(AMIN, frames[src + m])) - refDb;
        out[dst + m] = db;
        if (db > maxDb) maxDb = db;
      }
    }
    const floor = maxDb - TOP_DB;
    for (let i = 0; i < out.length; i++) if (out[i] < floor) out[i] = floor;
    return out;
  }

  // ── Resampler (native rate → 8 kHz) ────────────────────────────────────────
  // Windowed-sinc low-pass at ~0.45·8 kHz followed by fractional decimation
  // with linear interpolation. Avoids the aliasing of naive sample-picking.
  class Resampler {
    constructor(inRate, outRate = SR, taps = 63) {
      this.inRate = inRate; this.outRate = outRate;
      this.ratio = inRate / outRate;
      this.passthrough = Math.abs(this.ratio - 1) < 1e-9;
      const cutoff = 0.45 * outRate / inRate;   // normalised to input Nyquist=0.5
      this.kernel = new Float32Array(taps);
      const mid = (taps - 1) / 2;
      let sum = 0;
      for (let i = 0; i < taps; i++) {
        const x = i - mid;
        const sinc = x === 0 ? 2 * cutoff : Math.sin(2 * Math.PI * cutoff * x) / (Math.PI * x);
        const win = 0.42 - 0.5 * Math.cos(2 * Math.PI * i / (taps - 1)) + 0.08 * Math.cos(4 * Math.PI * i / (taps - 1));
        this.kernel[i] = sinc * win;
        sum += this.kernel[i];
      }
      for (let i = 0; i < taps; i++) this.kernel[i] /= sum;
      this.hist = new Float32Array(taps);  // ring of recent input samples
      this.histPos = 0;
      this.prev = 0;          // previous filtered sample
      this.inIndex = 0;       // index of next filtered input sample
      this.nextOut = 0;       // fractional input position of next output sample
    }

    process(input) {
      if (this.passthrough) return Float32Array.from(input);
      const out = [];
      const k = this.kernel, n = k.length;
      for (let i = 0; i < input.length; i++) {
        this.hist[this.histPos] = input[i];
        this.histPos = (this.histPos + 1) % n;
        let y = 0;
        for (let j = 0; j < n; j++) y += k[j] * this.hist[(this.histPos + j) % n];
        // Emit every output sample whose position lies in (inIndex-1, inIndex].
        while (this.nextOut <= this.inIndex) {
          const frac = this.nextOut - (this.inIndex - 1);
          out.push(this.prev + (y - this.prev) * frac);
          this.nextOut += this.ratio;
        }
        this.prev = y;
        this.inIndex++;
      }
      return Float32Array.from(out);
    }
  }

  // ── Probability post-processing ────────────────────────────────────────────
  // EMA smoothing (Gillick low-pass filters the frame probabilities before
  // thresholding; an EMA is the causal equivalent for live use).
  class ProbSmoother {
    constructor(alpha = 0.5) { this.alpha = alpha; this.value = 0; this.primed = false; }
    reset() { this.value = 0; this.primed = false; }
    update(p) {
      if (!this.primed) { this.value = p; this.primed = true; }
      else this.value = this.alpha * p + (1 - this.alpha) * this.value;
      return this.value;
    }
    decay() { this.value *= (1 - this.alpha); return this.value; }
  }

  // ── Trigger gate: hysteresis + leaky evidence accumulator ─────────────────
  // Laughter is bursty ("ha – ha – ha"): per-window scores dip between
  // syllables. A plain "above threshold for N ms, reset on any dip" rule keeps
  // restarting during real laughs, yet a single long false positive still
  // fires. Gillick et al. solve this offline with a zero-phase low-pass +
  // threshold + minimum duration; this is a causal equivalent:
  //   score ≥ on            → evidence += dt
  //   off ≤ score < on      → hold (hysteresis band, only once evidence > 0)
  //   score < off           → evidence -= dt · decayRate   (floored at 0)
  // and it fires once evidence ≥ sustainMs. Scores are normalised so 1.0 is
  // the caller's trigger threshold. dt is capped so a stalled tab (big gap
  // between updates) can't count as sustained laughter.
  class LaughGate {
    constructor({ on = 1, off = 0.85, sustainMs = 850, decayRate = 0.5, maxDtMs = 250 } = {}) {
      if (!(off <= on)) throw new Error('LaughGate: off threshold must be ≤ on threshold');
      Object.assign(this, { on, off, sustainMs, decayRate, maxDtMs });
      this.reset();
    }
    reset() { this.evidence = 0; this.lastT = null; }
    // Returns true when the accumulated evidence reaches sustainMs.
    update(score, nowMs) {
      const dt = this.lastT === null ? 0 : Math.min(Math.max(nowMs - this.lastT, 0), this.maxDtMs);
      this.lastT = nowMs;
      if (score >= this.on) this.evidence += dt;
      else if (score < this.off || this.evidence === 0) {
        this.evidence = Math.max(0, this.evidence - dt * this.decayRate);
      }
      return this.evidence >= this.sustainMs;
    }
    // 0..1 progress towards a trigger (useful for UI / debugging).
    progress() { return Math.min(this.evidence / this.sustainMs, 1); }
  }

  // ── ONNX model input-shape patch ───────────────────────────────────────────
  // The shipped ONNX declares input "input" as [1, 1, 128, "time"], i.e.
  // mel × time, but the Gillick weights were trained on time × mel
  // ([1, 1, 44, 128]). onnxruntime rejects inputs that violate a fixed dim, so
  // rewrite that dim_value=128 (bytes 08 80 01) as a symbolic dim_param "X"
  // (bytes 12 01 58) — same length, so no protobuf lengths change. Only the
  // graph input declaration is touched. Returns a patched copy, or null if the
  // expected byte pattern isn't found (e.g. a re-exported model).
  const INPUT_SHAPE_SIG = [
    0x0a, 0x05, 0x69, 0x6e, 0x70, 0x75, 0x74,       // name: "input"
    0x12, 0x1b, 0x0a, 0x19, 0x08, 0x01, 0x12, 0x15, // type → tensor_type(elem=float) → shape
    0x0a, 0x02, 0x08, 0x01,                         // dim 1
    0x0a, 0x02, 0x08, 0x01,                         // dim 1
    0x0a, 0x03, 0x08, 0x80, 0x01,                   // dim 128  ← patched
  ];
  function patchOnnxInputShape(bytes) {
    const src = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
    const sig = INPUT_SHAPE_SIG;
    outer:
    for (let i = src.length - sig.length; i >= 0; i--) {
      for (let j = 0; j < sig.length; j++) if (src[i + j] !== sig[j]) continue outer;
      const out = new Uint8Array(src);
      const p = i + sig.length - 3;
      out[p] = 0x12; out[p + 1] = 0x01; out[p + 2] = 0x58;   // dim_param "X"
      return out;
    }
    return null;
  }

  const api = {
    SR, N_FFT, HOP, N_MELS, N_FRAMES, WINDOW_SAMPLES, PEAK_FLOOR_DB,
    hzToMelSlaney, melToHzSlaney, melFilterbank, hannPeriodic, makeFft,
    LogMelStream, powerToDb, Resampler, ProbSmoother, LaughGate, patchOnnxInputShape,
  };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.LaughAudio = api;
})(typeof window !== 'undefined' ? window : globalThis);
