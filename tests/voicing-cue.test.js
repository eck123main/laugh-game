'use strict';
const test = require('node:test');
const assert = require('node:assert');
const { VoicingCue, SR } = require('../client/laugh-audio.js');

function rng(seed) { return () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296); }

// Segments: { ms, kind: 'voiced'|'noise'|'silence', f0, amp, h2 } at 8 kHz.
// Voiced = harmonic tone (fundamental + harmonics, h2 = 2nd-harmonic weight).
function signal(segments, seed = 1) {
  const r = rng(seed);
  const total = segments.reduce((s, x) => s + Math.round(x.ms * SR / 1000), 0);
  const out = new Float32Array(total);
  let p = 0, phase = 0;
  for (const { ms, kind, f0 = 200, amp = 0.3, h2 = 0.5 } of segments) {
    const n = Math.round(ms * SR / 1000);
    for (let i = 0; i < n; i++) {
      let v = 0;
      if (kind === 'voiced') {
        phase += 2 * Math.PI * f0 / SR;
        v = amp * (Math.sin(phase) + h2 * Math.sin(2 * phase) + 0.3 * Math.sin(3 * phase));
      } else if (kind === 'noise') v = amp * (r() * 2 - 1);
      out[p++] = v + 0.0005 * (r() * 2 - 1);
    }
  }
  return out;
}

function run(sig) {
  const vc = new VoicingCue();
  for (let i = 0; i < sig.length; i += 1024) vc.push(sig.subarray(i, i + 1024));
  return vc;
}

test('sustained voiced tone (humming / held vowel) is all voiced → minimum scale', () => {
  const vc = run(signal([{ ms: 1600, kind: 'voiced', f0: 200 }]));
  const st = vc.stats();
  assert.ok(st.unvoicedRatio < 0.05, `unvoiced ${st.unvoicedRatio}`);
  assert.ok(Math.abs(st.f0Median - 200) < 10, `f0 ${st.f0Median}`);
  assert.ok(st.f0SpreadSt < 1, `spread ${st.f0SpreadSt}`);
  assert.ok(vc.scale() < 0.92, `scale ${vc.scale()}`);
});

test('strong 2nd harmonic does not cause an octave error', () => {
  const st = run(signal([{ ms: 1600, kind: 'voiced', f0: 150, h2: 1.5 }])).stats();
  assert.ok(Math.abs(st.f0Median - 150) < 8, `f0 ${st.f0Median}`);
});

test('noise (breath, rustle) is unvoiced → full scale', () => {
  const vc = run(signal([{ ms: 1600, kind: 'noise', amp: 0.2 }]));
  assert.ok(vc.stats().unvoicedRatio > 0.9);
  assert.strictEqual(vc.scale(), 1);
});

test('"ha-ha" calls: voiced vowels with breathy h onsets → full scale, high f0', () => {
  const segs = [];
  for (let i = 0; i < 8; i++) {
    segs.push({ ms: 50, kind: 'noise', amp: 0.08 }, { ms: 90, kind: 'voiced', f0: 320 }, { ms: 60, kind: 'silence' });
  }
  const vc = run(signal(segs));
  const st = vc.stats();
  assert.ok(st.unvoicedRatio > 0.2 && st.unvoicedRatio < 0.6, `unvoiced ${st.unvoicedRatio}`);
  assert.ok(Math.abs(st.f0Median - 320) < 20, `f0 ${st.f0Median}`);
  assert.strictEqual(vc.scale(), 1);
});

test('silent gaps between calls are ignored (voiced calls + silence stay "voiced")', () => {
  const segs = [];
  for (let i = 0; i < 8; i++) segs.push({ ms: 100, kind: 'voiced', f0: 250 }, { ms: 100, kind: 'silence' });
  const st = run(signal(segs)).stats();
  assert.ok(st.unvoicedRatio < 0.15, `unvoiced ${st.unvoicedRatio}`);
});

test('too little sound → no stats, neutral scale; reset clears history', () => {
  const vc = new VoicingCue();
  vc.push(new Float32Array(800));
  assert.strictEqual(vc.stats(), null);
  assert.strictEqual(vc.scale(), 1);
  vc.push(signal([{ ms: 1600, kind: 'voiced' }]));
  assert.ok(vc.scale() < 0.92);
  vc.reset();
  assert.strictEqual(vc.stats(), null);
  assert.strictEqual(vc.scale(), 1);
});
