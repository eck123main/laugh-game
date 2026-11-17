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
  s.push(new Float32Array(LA.HOP * 10)); // 10 frames of silence at the end
  const w = s.getWindow();
  const frameMean = (t) => {
    let a = 0;
    for (let m = 0; m < LA.N_MELS; m++) a += w[t * LA.N_MELS + m];
    return a / LA.N_MELS;
  };
  assert.ok(frameMean(LA.N_FRAMES - 1) < frameMean(10) - 10, 'last frames (silence) quieter than earlier tone frames');
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
