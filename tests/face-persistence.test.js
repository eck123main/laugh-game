'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const LF = require('../client/laugh-face.js');

const cue = (smile, eye = 0, open = 0) => ({ score: 0, open, smile, eye });

test('CuePersistence drops a ~150 ms vowel-spread smile spike', () => {
  const p = new LF.CuePersistence();
  let maxSmile = 0;
  for (let t = 0; t <= 2000; t += 33) {
    const spike = (t % 500) < 150;               // spread vowel every 0.5 s
    maxSmile = Math.max(maxSmile, p.update(cue(spike ? 1 : 0), t).smile);
  }
  assert.equal(maxSmile, 0);
});

test('CuePersistence drops a blink (eye cue spike)', () => {
  const p = new LF.CuePersistence();
  let maxEye = 0;
  for (let t = 0; t <= 1000; t += 33) maxEye = Math.max(maxEye, p.update(cue(0, t >= 400 && t < 520 ? 1 : 0), t).eye);
  assert.equal(maxEye, 0);
});

test('CuePersistence passes a sustained smile within ~300 ms and recomputes score', () => {
  const p = new LF.CuePersistence();
  for (let t = 0; t < 500; t += 33) p.update(cue(0), t);
  let firstFull = null, out;
  for (let t = 500; t <= 1500; t += 33) {
    out = p.update(cue(1, 0.5, 0.8), t);
    if (firstFull === null && out.smile === 1) firstFull = t;
  }
  assert.ok(firstFull !== null && firstFull - 500 <= 300, 'onset delay ' + (firstFull - 500));
  const W = LF.WEIGHTS;
  assert.ok(Math.abs(out.score - (W.open * 0.8 + W.smile * 1 + W.eye * 0.5)) < 1e-9);
  assert.equal(out.rawSmile, 1);
  assert.equal(out.open, 0.8);
});

test('CuePersistence forgets history after a gap or reset, at low fps too', () => {
  const p = new LF.CuePersistence();
  for (let t = 0; t <= 1000; t += 66) p.update(cue(1), t);       // ~15 fps
  assert.equal(p.update(cue(0), 3000).smile, 0);                // face lost 2 s
  p.update(cue(1), 3066);
  p.reset();
  assert.equal(p.update(cue(0), 3100).smile, 0);
  assert.throws(() => new LF.CuePersistence({ windowMs: 0 }), RangeError);
});

test('CuePersistence + LaughGate: blinking/talking face never fires, sustained laugh face does', () => {
  const LA = require('../client/laugh-audio.js');
  const run = (cueAt) => {
    const p = new LF.CuePersistence(), gate = new LA.LaughGate();
    for (let t = 0; t <= 6000; t += 33) {
      const face = p.update(cueAt(t), t).score;
      if (gate.update(face / 0.80, t)) return t;   // face-only route
    }
    return null;
  };
  // Talking: mouth wide open, smile spikes on vowels, blinks every 2 s.
  assert.equal(run((t) => cue((t % 400) < 150 ? 1 : 0, (t % 2000) < 120 ? 1 : 0, 1)), null);
  // Laughing: open mouth, held smile and squint.
  assert.ok(run(() => cue(1, 1, 1)) !== null);
});
