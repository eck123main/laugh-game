#!/usr/bin/env node
// Offline evaluation of the audio laugh pipeline. Each WAV goes through the
// same steps as the browser (client/game.js → onMicSamples):
//   1024-sample chunks → Resampler → LogMelStream → (RMS gate) → ResNet
//   → ProbSmoother → LaughGate (audio-alone route, no face).
//
// Usage:
//   npm i --no-save onnxruntime-node        # once; not a project dependency
//   node scripts/eval-audio.js <wav files or dirs...> [--trace]
//
// Labels come from the path: a file under a directory named "laugh" or
// "laughter" should fire, anything else should not. Example layout:
//   clips/laugh/giggle1.wav   clips/speech/talk1.wav   clips/noise/fan.wav
// Record clips at any rate, mono or stereo, PCM16/24/32 or float32.
//
// Output: one line per file (fired?, when, peak probability), a confusion
// summary, and a sweep over audio-alone thresholds. Use the sweep to tune
// AUDIO_SOLO_THRESHOLD / AUDIO_RMS_GATE in game.js from data, not guesses.
'use strict';
const fs = require('fs');
const path = require('path');
const LA = require('../client/laugh-audio.js');

// Mirrors client/game.js.
const DEFAULTS = Object.freeze({
  chunk: 1024, inferEveryFrames: 4, rmsGate: 0.008, smoothAlpha: 0.5,
  soloThreshold: 0.75, gateOff: 0.85, sustainMs: 850, decayRate: 0.5,
});

// Minimal RIFF/WAVE reader → { sampleRate, samples: Float32Array (mono) }.
function parseWav(buf) {
  if (buf.toString('ascii', 0, 4) !== 'RIFF' || buf.toString('ascii', 8, 12) !== 'WAVE') throw new Error('not a WAV file');
  let fmt = null, data = null;
  for (let p = 12; p + 8 <= buf.length;) {
    const id = buf.toString('ascii', p, p + 4), size = buf.readUInt32LE(p + 4);
    const body = p + 8;
    if (id === 'fmt ') {
      let format = buf.readUInt16LE(body);
      if (format === 0xfffe && size >= 26) format = buf.readUInt16LE(body + 24);   // WAVE_FORMAT_EXTENSIBLE
      fmt = { format, channels: buf.readUInt16LE(body + 2), sampleRate: buf.readUInt32LE(body + 4), bits: buf.readUInt16LE(body + 14) };
    } else if (id === 'data') {
      data = buf.subarray(body, Math.min(body + size, buf.length));
    }
    p = body + size + (size & 1);
  }
  if (!fmt || !data) throw new Error('WAV missing fmt or data chunk');
  const { format, channels, bits } = fmt, bytes = bits / 8;
  const read = format === 3 && bits === 32 ? (o) => data.readFloatLE(o)
    : format === 1 && bits === 16 ? (o) => data.readInt16LE(o) / 32768
    : format === 1 && bits === 24 ? (o) => data.readIntLE(o, 3) / 8388608
    : format === 1 && bits === 32 ? (o) => data.readInt32LE(o) / 2147483648
    : null;
  if (!read) throw new Error(`unsupported WAV encoding (format ${format}, ${bits} bit)`);
  const n = Math.floor(data.length / (bytes * channels));
  const samples = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let c = 0; c < channels; c++) s += read((i * channels + c) * bytes);
    samples[i] = s / channels;
  }
  return { sampleRate: fmt.sampleRate, samples };
}

// Replays one clip. `infer(window: Float32Array[N_FRAMES*N_MELS]) → prob`
// may be async. Returns the smoothed-probability trace ({ t ms, p }) and
// when (if ever) the gate fired with the given threshold.
async function simulate(samples, sampleRate, infer, opts = {}) {
  const o = { ...DEFAULTS, ...opts };
  const resampler = new LA.Resampler(sampleRate, LA.SR || 8000);
  const stream = new LA.LogMelStream();
  const smoother = new LA.ProbSmoother(o.smoothAlpha);
  const trace = [];
  let frames = 0, raw = 0;
  for (let i = 0; i < samples.length; i += o.chunk) {
    const chunk = samples.subarray(i, Math.min(i + o.chunk, samples.length));
    frames += stream.push(resampler.process(chunk));
    if (frames < o.inferEveryFrames || !stream.ready()) continue;
    frames = 0;
    const t = ((i + chunk.length) / sampleRate) * 1000;
    let p;
    if (stream.peakRms() < o.rmsGate) p = smoother.decay();
    else { raw = Math.min(Math.max(await infer(stream.getWindow()), 0), 1); p = smoother.update(raw); }
    trace.push({ t, p });
  }
  return { trace, firedAt: gateOnTrace(trace, o.soloThreshold, o) };
}

// Runs the LaughGate over a recorded trace (cheap, so thresholds can be swept
// without re-running the model).
function gateOnTrace(trace, threshold, opts = {}) {
  const o = { ...DEFAULTS, ...opts };
  const gate = new LA.LaughGate({ on: 1, off: o.gateOff, sustainMs: o.sustainMs, decayRate: o.decayRate });
  for (const { t, p } of trace) if (gate.update(p / threshold, t)) return t;
  return null;
}

const isLaughPath = (f) => f.split(/[\\/]/).slice(0, -1).some((d) => /^laugh(ter)?$/i.test(d));

function collectWavs(args) {
  const out = [];
  const walk = (p) => {
    const st = fs.statSync(p);
    if (st.isDirectory()) for (const e of fs.readdirSync(p).sort()) walk(path.join(p, e));
    else if (/\.wav$/i.test(p)) out.push(p);
  };
  args.forEach(walk);
  return out;
}

async function loadModel() {
  let ort;
  try { ort = require('onnxruntime-node'); } catch (_) {
    throw new Error('onnxruntime-node not found. Install it with: npm i --no-save onnxruntime-node');
  }
  const bytes = new Uint8Array(fs.readFileSync(path.join(__dirname, '..', 'client', 'models', 'laughter_resnet.onnx')));
  const patched = LA.patchOnnxInputShape(bytes);
  if (!patched) throw new Error('ONNX input signature not found');
  const session = await ort.InferenceSession.create(patched);
  const dims = [1, 1, LA.N_FRAMES, LA.N_MELS];
  return async (win) => {
    const out = await session.run({ [session.inputNames[0]]: new ort.Tensor('float32', Float32Array.from(win), dims) });
    return out[session.outputNames[0]].data[0];
  };
}

async function main(argv) {
  const showTrace = argv.includes('--trace');
  const files = collectWavs(argv.filter((a) => !a.startsWith('--')));
  if (!files.length) { console.error('usage: node scripts/eval-audio.js <wav files or dirs...> [--trace]'); process.exit(2); }
  const infer = await loadModel();
  const results = [];
  for (const f of files) {
    const { sampleRate, samples } = parseWav(fs.readFileSync(f));
    const { trace, firedAt } = await simulate(samples, sampleRate, infer);
    const peak = trace.reduce((m, x) => Math.max(m, x.p), 0);
    const laugh = isLaughPath(f);
    results.push({ f, laugh, trace });
    const ok = (firedAt !== null) === laugh ? 'ok  ' : 'MISS';
    console.log(`${ok} ${laugh ? 'laugh' : 'other'}  fired=${firedAt === null ? '-' : (firedAt / 1000).toFixed(2) + 's'}  peak=${peak.toFixed(2)}  ${f}`);
    if (showTrace) console.log('     ' + trace.map((x) => x.p.toFixed(2)).join(' '));
  }
  console.log('\nthreshold  recall(laugh fired)  false-fire(other fired)');
  for (const th of [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.88, 0.92, 0.95]) {
    const fire = results.map((r) => gateOnTrace(r.trace, th) !== null);
    const pos = results.filter((r) => r.laugh).length, neg = results.length - pos;
    const tp = results.filter((r, i) => r.laugh && fire[i]).length;
    const fp = results.filter((r, i) => !r.laugh && fire[i]).length;
    console.log(`  ${th.toFixed(2)}      ${pos ? tp + '/' + pos : '-'}                  ${neg ? fp + '/' + neg : '-'}`);
  }
}

module.exports = { parseWav, simulate, gateOnTrace, isLaughPath, DEFAULTS };
if (require.main === module) main(process.argv.slice(2)).catch((e) => { console.error(e.message); process.exit(1); });
