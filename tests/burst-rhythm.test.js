'use strict';
const test = require('node:test');
const assert = require('node:assert');
const { BurstRhythm, SR } = require('../client/laugh-audio.js');

// Deterministic PRNG so the tests are stable.
function rng(seed) { return () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296); }

// Concatenate segments [{ ms, amp }] of noise bursts / gaps at 8 kHz.
function signal(segments, seed = 1) {
  const r = rng(seed);
  const total = segments.reduce((s, x) => s + Math.round(x.ms * SR / 1000), 0);
  const out = new Float32Array(total);
  let p = 0;
  for (const { ms, amp } of segments) {
    const n = Math.round(ms * SR / 1000);
    for (let i = 0; i < n; i++) out[p++] = amp * (r() * 2 - 1) + 0.001 * (r() * 2 - 1);
  }
  return out;
}

function regularityOf(sig) {
  const br = new BurstRhythm();
  for (let i = 0; i < sig.length; i += 1024) br.push(sig.subarray(i, i + 1024));
  return { r: br.regularity(), s: br.scale() };
}

test('regular "ha-ha-ha" bursts (~5 Hz) score high and keep full scale', () => {
  const segs = [];
  for (let i = 0; i < 9; i++) segs.push({ ms: 90, amp: 0.3 }, { ms: 110, amp: 0 });
  const { r, s } = regularityOf(signal(segs));
  assert.ok(r > 0.6, `regularity ${r}`);
  assert.strictEqual(s, 1);
});

test('irregular speech-like bursts score lower than regular laughter', () => {
  const r0 = rng(7);
  const segs = [];
  for (let t = 0; t < 1800;) {
    const on = 60 + r0() * 300, off = 20 + r0() * 120;
    segs.push({ ms: on, amp: 0.05 + r0() * 0.3 }, { ms: off, amp: 0.01 });
    t += on + off;
  }
  const irregular = regularityOf(signal(segs, 3)).r;
  const reg = [];
  for (let i = 0; i < 9; i++) reg.push({ ms: 90, amp: 0.3 }, { ms: 110, amp: 0 });
  const regular = regularityOf(signal(reg)).r;
  assert.ok(irregular < regular - 0.2, `irregular ${irregular} vs regular ${regular}`);
});

test('steady noise and a single burst get the minimum scale', () => {
  const fan = regularityOf(signal([{ ms: 1600, amp: 0.2 }]));
  assert.strictEqual(fan.r, 0);
  assert.ok(Math.abs(fan.s - 0.9) < 1e-9);
  const cough = regularityOf(signal([{ ms: 600, amp: 0 }, { ms: 250, amp: 0.5 }, { ms: 750, amp: 0 }]));
  assert.ok(cough.r < 0.3, `cough regularity ${cough.r}`);
  assert.ok(cough.s < 0.97);
});

test('not enough audio yet → regularity 0; reset clears history', () => {
  const br = new BurstRhythm();
  br.push(new Float32Array(400));
  assert.strictEqual(br.regularity(), 0);
  const segs = [];
  for (let i = 0; i < 9; i++) segs.push({ ms: 90, amp: 0.3 }, { ms: 110, amp: 0 });
  br.push(signal(segs));
  assert.ok(br.regularity() > 0.6);
  br.reset();
  assert.strictEqual(br.regularity(), 0);
});
