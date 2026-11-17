'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const EV = require('../scripts/eval-audio.js');

// Builds a PCM16 WAV buffer from per-channel float arrays.
function wav16(channels, sampleRate) {
  const n = channels[0].length, nc = channels.length;
  const buf = Buffer.alloc(44 + n * nc * 2);
  buf.write('RIFF', 0); buf.writeUInt32LE(36 + n * nc * 2, 4); buf.write('WAVE', 8);
  buf.write('fmt ', 12); buf.writeUInt32LE(16, 16); buf.writeUInt16LE(1, 20); buf.writeUInt16LE(nc, 22);
  buf.writeUInt32LE(sampleRate, 24); buf.writeUInt32LE(sampleRate * nc * 2, 28);
  buf.writeUInt16LE(nc * 2, 32); buf.writeUInt16LE(16, 34);
  buf.write('data', 36); buf.writeUInt32LE(n * nc * 2, 40);
  for (let i = 0; i < n; i++) for (let c = 0; c < nc; c++)
    buf.writeInt16LE(Math.round(Math.max(-1, Math.min(1, channels[c][i])) * 32767), 44 + (i * nc + c) * 2);
  return buf;
}

const tone = (sec, sr, amp = 0.3, hz = 440) => Float32Array.from({ length: Math.round(sec * sr) }, (_, i) => amp * Math.sin(2 * Math.PI * hz * i / sr));

test('parseWav reads PCM16 stereo and mixes to mono', () => {
  const l = Float32Array.from([0.5, -0.5, 0.25]), r = Float32Array.from([0.5, 0.5, -0.25]);
  const { sampleRate, samples } = EV.parseWav(wav16([l, r], 16000));
  assert.equal(sampleRate, 16000);
  assert.equal(samples.length, 3);
  assert.ok(Math.abs(samples[0] - 0.5) < 1e-3 && Math.abs(samples[1]) < 1e-3 && Math.abs(samples[2]) < 1e-3);
  assert.throws(() => EV.parseWav(Buffer.from('not a wav file at all, nope')), /not a WAV/);
});

test('simulate: a confident 3 s "laugh" fires the audio-alone gate; silence never runs the model', async () => {
  const sr = 16000;
  let calls = 0;
  const infer = () => { calls++; return 0.95; };
  const loud = await EV.simulate(tone(3, sr), sr, infer);
  assert.ok(loud.firedAt !== null, 'should fire');
  // Window fill (~1.26 s) + smoothing + 850 ms evidence.
  assert.ok(loud.firedAt > 1500 && loud.firedAt < 3000, 'fired at ' + loud.firedAt);

  calls = 0;
  const quiet = await EV.simulate(new Float32Array(3 * sr), sr, infer);
  assert.equal(quiet.firedAt, null);
  assert.equal(calls, 0, 'RMS gate keeps silence away from the model');
  assert.ok(quiet.trace.length > 10);
});

test('simulate: a borderline score on a steady (rhythm-less) sound is pushed below threshold by BurstRhythm', async () => {
  const sr = 8000;
  const off = await EV.simulate(tone(3, sr), sr, () => 0.8, { rhythm: false });
  assert.ok(off.firedAt !== null, 'fires without the rhythm cue');
  const on = await EV.simulate(tone(3, sr), sr, () => 0.8, { rhythm: true });
  assert.equal(on.firedAt, null, '0.8 × 0.9 < 0.75 for a flat tone');
  assert.ok(on.trace.some((x) => x.r === 0));
});

test('simulate: inference cadence is ~93 ms like the browser', async () => {
  const sr = 8000;
  const { trace } = await EV.simulate(tone(4, sr), sr, () => 0.1);
  const dts = trace.slice(1).map((x, i) => x.t - trace[i].t);
  const mean = dts.reduce((a, b) => a + b, 0) / dts.length;
  assert.ok(mean > 85 && mean < 140, 'mean step ' + mean);
});

test('gateOnTrace sweeps thresholds on a recorded trace', () => {
  const trace = Array.from({ length: 40 }, (_, i) => ({ t: i * 93, p: 0.8 }));
  assert.ok(EV.gateOnTrace(trace, 0.75) !== null);
  assert.equal(EV.gateOnTrace(trace, 0.92), null);
});

test('isLaughPath labels by directory name', () => {
  assert.equal(EV.isLaughPath('clips/laugh/a.wav'), true);
  assert.equal(EV.isLaughPath('clips\\Laughter\\b.wav'), true);
  assert.equal(EV.isLaughPath('clips/speech/laugh.wav'), false);
  assert.equal(EV.isLaughPath('clips/laughing-talk/c.wav'), false);
});
