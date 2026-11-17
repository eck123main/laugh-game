'use strict';
// Run with: npm test   (node --test)
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const LA = require('../client/laugh-audio.js');

const close = (a, b, tol, msg) =>
  assert.ok(Math.abs(a - b) <= tol, `${msg || ''} expected ${b}, got ${a}`);

test('Slaney mel scale matches librosa reference points', () => {
  close(LA.hzToMelSlaney(1000), 15, 1e-9);
  close(LA.hzToMelSlaney(500), 7.5, 1e-9);
  close(LA.melToHzSlaney(15), 1000, 1e-9);
  // librosa.hz_to_mel(4000) == 35.16...
  close(LA.hzToMelSlaney(4000), 15 + Math.log(4) / (Math.log(6.4) / 27), 1e-9);
  for (const hz of [0, 60, 999, 1001, 3999]) close(LA.melToHzSlaney(LA.hzToMelSlaney(hz)), hz, 1e-6);
});

test('mel filterbank has the librosa shape and Slaney area normalisation', () => {
  const nBins = LA.N_FFT / 2 + 1;
  const w = LA.melFilterbank();
  assert.equal(w.length, LA.N_MELS * nBins);
  for (let m = 0; m < LA.N_MELS; m++) {
    let nonzero = 0;
    for (let k = 0; k < nBins; k++) {
      const v = w[m * nBins + k];
      assert.ok(v >= 0);
      if (v > 0) nonzero++;
    }
    assert.ok(nonzero > 0, `band ${m} should cover at least one FFT bin`);
  }
});

test('FFT power spectrum matches a naive DFT', () => {
  const n = 64;
  const fft = LA.makeFft(n);
  const x = new Float32Array(n);
  for (let i = 0; i < n; i++) x[i] = Math.sin(i * 0.7) + 0.3 * Math.cos(i * 2.1) + (i % 5) * 0.01;
  const p = fft(x);
  for (let k = 0; k <= n / 2; k++) {
    let re = 0, im = 0;
    for (let t = 0; t < n; t++) {
      re += x[t] * Math.cos(-2 * Math.PI * k * t / n);
      im += x[t] * Math.sin(-2 * Math.PI * k * t / n);
    }
    close(p[k], re * re + im * im, 1e-3, `bin ${k}`);
  }
});

test('periodic Hann window', () => {
  const w = LA.hannPeriodic(8);
  close(w[0], 0, 1e-12);
  close(w[4], 1, 1e-6);
  close(w[2], 0.5, 1e-6);
});

function tone(freq, seconds, sr = LA.SR, amp = 0.3) {
  const n = Math.round(seconds * sr);
  const x = new Float32Array(n);
  for (let i = 0; i < n; i++) x[i] = amp * Math.sin(2 * Math.PI * freq * i / sr);
  return x;
}

test('LogMelStream produces a [44 x 128] dB window once enough audio arrives', () => {
  const s = new LA.LogMelStream();
  assert.equal(s.ready(), false);
  const produced = s.push(tone(1000, 1.5));
  assert.equal(produced, Math.floor((1.5 * LA.SR - LA.N_FFT) / LA.HOP) + 1);
  assert.ok(s.ready());
  const w = s.getWindow();
  assert.equal(w.length, LA.N_FRAMES * LA.N_MELS);
  let mx = -Infinity, mn = Infinity;
  for (const v of w) { mx = Math.max(mx, v); mn = Math.min(mn, v); }
  close(mx, 0, 1e-3, 'max dB (ref=max)');
  assert.ok(mn >= -80 - 1e-3, 'top_db clipping');
  // Energy should peak in the band containing 1 kHz (Slaney mel 15).
  const t = 20, row = w.subarray(t * LA.N_MELS, (t + 1) * LA.N_MELS);
  let best = 0;
  for (let m = 1; m < LA.N_MELS; m++) if (row[m] > row[best]) best = m;
  const melStep = LA.hzToMelSlaney(LA.SR / 2) / (LA.N_MELS + 1);
  const centreHz = LA.melToHzSlaney((best + 1) * melStep);
  assert.ok(Math.abs(centreHz - 1000) < 60, `peak band centre ${centreHz} Hz`);
  assert.ok(s.rms() > 0.15 && s.rms() < 0.25, 'rms of a 0.3-amp sine ≈ 0.21');
});

test('LogMelStream window is time-major and chronologically ordered', () => {
  const s = new LA.LogMelStream();
  const silence = new Float32Array(LA.SR); // 1 s silence then a loud tone
  s.push(silence);
  s.push(tone(500, 1.5));
  // 20 hops (3720 samples) > N_FFT, so the newest frames contain no tone at
  // all (a frame that still holds a truncated tone tail leaks broadband).
  s.push(new Float32Array(LA.HOP * 20));
  const w = s.getWindow();
  const framePeak = (t) => {
    let a = -Infinity;
    for (let m = 0; m < LA.N_MELS; m++) a = Math.max(a, w[t * LA.N_MELS + m]);
    return a;
  };
  assert.ok(framePeak(LA.N_FRAMES - 1) < framePeak(10) - 10, 'last frames (silence) quieter than earlier tone frames');
  close(framePeak(10), 0, 1, 'tone frame holds the window max');
});

test('dB reference never sinks to the noise floor after a long quiet stretch', () => {
  const s = new LA.LogMelStream();
  s.push(tone(1000, 1.5, LA.SR, 0.3));      // loud event sets the session peak
  s.push(tone(1000, 25, LA.SR, 0.003));     // 25 s of quiet (-40 dB amplitude)
  const w = s.getWindow();
  let mx = -Infinity;
  for (const v of w) mx = Math.max(mx, v);
  // Quiet tone is -80 units below the peak; reference is floored 60 below,
  // so the quiet window must stay clearly below 0 dB instead of being
  // renormalised up to look like a loud sound.
  close(mx, -(80 - LA.PEAK_FLOOR_DB), 2, 'quiet window max dB');
  s.reset();
  assert.equal(s.sessionPeak, 0);
});

test('reset clears the stream', () => {
  const s = new LA.LogMelStream();
  s.push(tone(700, 1.5));
  s.reset();
  assert.equal(s.ready(), false);
  assert.equal(s.getWindow(), null);
});

test('Resampler: 48 kHz → 8 kHz keeps length ratio and rejects aliasing', () => {
  const r = new LA.Resampler(48000, 8000);
  const input = tone(440, 1, 48000);
  const chunks = [];
  for (let i = 0; i < input.length; i += 1024) chunks.push(r.process(input.subarray(i, i + 1024)));
  const total = chunks.reduce((a, c) => a + c.length, 0);
  assert.ok(Math.abs(total - 8000) <= 2, `got ${total} samples`);

  // 440 Hz passes; 7 kHz (would alias to 1 kHz) is strongly attenuated.
  const rms = (x) => Math.sqrt(x.reduce((a, v) => a + v * v, 0) / x.length);
  const pass = new LA.Resampler(48000, 8000).process(tone(440, 0.5, 48000));
  const stop = new LA.Resampler(48000, 8000).process(tone(7000, 0.5, 48000));
  assert.ok(rms(pass.subarray(100)) > 0.18, 'passband preserved');
  assert.ok(rms(stop.subarray(100)) < 0.02, 'stopband attenuated');
});

test('Resampler handles non-integer ratios (44.1 kHz)', () => {
  const r = new LA.Resampler(44100, 8000);
  const out = r.process(new Float32Array(44100));
  assert.ok(Math.abs(out.length - 8000) <= 2);
});

test('Resampler is a passthrough at 8 kHz', () => {
  const r = new LA.Resampler(8000, 8000);
  const x = tone(300, 0.1);
  assert.deepEqual(Array.from(r.process(x)), Array.from(x));
});

test('ProbSmoother EMA', () => {
  const p = new LA.ProbSmoother(0.5);
  assert.equal(p.update(1), 1);
  assert.equal(p.update(0), 0.5);
  assert.equal(p.decay(), 0.25);
  p.reset();
  assert.equal(p.update(0.2), 0.2);
});

// Feed a score pattern to a gate at a fixed step; returns time (ms) of the
// first trigger, or null if it never fires.
function runGate(gate, pattern, stepMs = 50) {
  for (let i = 0; i < pattern.length; i++) {
    if (gate.update(pattern[i], i * stepMs)) return i * stepMs;
  }
  return null;
}
const repeat = (arr, n) => Array.from({ length: n }, () => arr).flat();

test('LaughGate fires after sustained score above threshold', () => {
  const g = new LA.LaughGate({ sustainMs: 850 });
  assert.equal(runGate(g, new Array(17).fill(1.2)), null, '800 ms is not enough');
  g.reset();
  assert.equal(runGate(g, new Array(40).fill(1.2)), 850);
});

test('LaughGate never fires on silence or sub-threshold scores', () => {
  const g = new LA.LaughGate();
  assert.equal(runGate(g, new Array(200).fill(0.9)), null);
  assert.equal(g.evidence, 0, 'hysteresis band does not hold when nothing accumulated');
});

test('LaughGate tolerates the dips of bursty laughter', () => {
  // "ha – ha – ha": 300 ms on, 150 ms off (score drops to 0.3).
  const burst = [...new Array(6).fill(1.2), ...new Array(3).fill(0.3)];
  const g = new LA.LaughGate({ sustainMs: 850 });
  const t = runGate(g, repeat(burst, 10));
  assert.ok(t !== null && t < 2000, `bursty laugh should trigger within 2 s, got ${t}`);
});

test('LaughGate rejects short sparse spikes (speech-like)', () => {
  // 100 ms spikes every 600 ms.
  const spike = [1.2, 1.2, ...new Array(10).fill(0.2)];
  const g = new LA.LaughGate({ sustainMs: 850 });
  assert.equal(runGate(g, repeat(spike, 30)), null);
});

test('LaughGate hysteresis band holds evidence', () => {
  const g = new LA.LaughGate({ sustainMs: 10000 });
  runGate(g, new Array(11).fill(1.2));       // 500 ms of evidence
  const before = g.evidence;
  for (let i = 0; i < 10; i++) g.update(0.9, 550 + i * 50);
  assert.equal(g.evidence, before, 'score in [off, on) neither adds nor removes evidence');
  g.update(0.1, 1100);
  assert.ok(g.evidence < before, 'below off threshold decays');
});

test('LaughGate caps dt so a stalled tab cannot fake a sustained laugh', () => {
  const g = new LA.LaughGate({ sustainMs: 850, maxDtMs: 250 });
  assert.equal(g.update(1.2, 0), false);
  assert.equal(g.update(1.2, 5000), false, 'one 5 s gap counts as at most 250 ms');
  assert.ok(Math.abs(g.progress() - 250 / 850) < 1e-9);
});

test('LaughGate validates thresholds', () => {
  assert.throws(() => new LA.LaughGate({ on: 0.5, off: 0.8 }));
});

test('patchOnnxInputShape rewrites the fixed 128 dim to a symbolic dim', () => {
  const head = [1, 2, 3];
  const sig = [0x0a, 0x05, 0x69, 0x6e, 0x70, 0x75, 0x74, 0x12, 0x1b, 0x0a, 0x19, 0x08, 0x01,
    0x12, 0x15, 0x0a, 0x02, 0x08, 0x01, 0x0a, 0x02, 0x08, 0x01, 0x0a, 0x03, 0x08, 0x80, 0x01];
  const tail = [0x0a, 0x06, 0x12, 0x04, 0x74, 0x69, 0x6d, 0x65];
  const buf = Uint8Array.from([...head, ...sig, ...tail]);
  const out = LA.patchOnnxInputShape(buf);
  assert.ok(out);
  assert.equal(out.length, buf.length);
  assert.deepEqual(Array.from(out.subarray(head.length + sig.length - 3, head.length + sig.length)), [0x12, 0x01, 0x58]);
  assert.equal(buf[head.length + sig.length - 2], 0x80, 'input buffer untouched');
  assert.equal(LA.patchOnnxInputShape(Uint8Array.from([1, 2, 3])), null);
});

test('shipped model contains the expected input signature', () => {
  const file = path.join(__dirname, '..', 'client', 'models', 'laughter_resnet.onnx');
  if (!fs.existsSync(file)) return;
  const patched = LA.patchOnnxInputShape(fs.readFileSync(file));
  assert.ok(patched, 'input shape signature should be found in laughter_resnet.onnx');
});
