'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const LF = require('../client/laugh-face.js');

// Synthetic FaceMesh landmarks. Coordinates are in "square" units and divided
// by `aspect` on x so the aspect correction in getFaceFeatures is exercised.
// scale: face size (distance to camera); open: lip gap; widen: mouth width
// factor; lift: corner lift; eye: eye aperture factor.
function face({ scale = 1, open = 0.01, widen = 1, lift = 0, eye = 1, aspect = 4 / 3, cx = 0.5, cy = 0.5 } = {}) {
  const lm = Array.from({ length: 478 }, () => ({ x: cx, y: cy, z: 0 }));
  const put = (i, x, y) => { lm[i] = { x: cx + (x * scale) / aspect, y: cy + y * scale, z: 0 }; };
  const L = LF.LM;
  put(L.EYE_OUTER_L, -0.1, -0.1); put(L.EYE_OUTER_R, 0.1, -0.1);        // iod = 0.2·scale
  const eh = 0.02 * eye;
  put(L.L_EYE_TOP, -0.06, -0.1 - eh / 2); put(L.L_EYE_BOT, -0.06, -0.1 + eh / 2);
  put(L.R_EYE_TOP, 0.06, -0.1 - eh / 2);  put(L.R_EYE_BOT, 0.06, -0.1 + eh / 2);
  const mw = 0.08 * widen;
  put(L.MOUTH_LEFT, -mw, 0.1 - lift); put(L.MOUTH_RIGHT, mw, 0.1 - lift);
  put(L.INNER_LEFT, -mw * 0.8, 0.1 - lift); put(L.INNER_RIGHT, mw * 0.8, 0.1 - lift);
  put(L.UPPER_LIP, 0, 0.1 - open / 2); put(L.LOWER_LIP, 0, 0.1 + open / 2);
  return lm;
}

const feats = (o) => LF.getFaceFeatures(face(o));
const neutralBase = () => LF.calibrate(Array.from({ length: 30 }, () => feats({})));

test('features are invariant to distance from the camera', () => {
  const near = feats({ scale: 1.6, open: 0.05, widen: 1.2 });
  const far  = feats({ scale: 0.6, open: 0.05, widen: 1.2 });
  for (const k of ['mouth', 'width', 'lift', 'eye']) {
    assert.ok(Math.abs(near[k] - far[k]) < 1e-6, `${k}: ${near[k]} vs ${far[k]}`);
  }
});

test('aspect correction: features match regardless of image aspect', () => {
  const a = LF.getFaceFeatures(face({ aspect: 4 / 3, widen: 1.1 }), 4 / 3);
  const b = LF.getFaceFeatures(face({ aspect: 16 / 9, widen: 1.1 }), 16 / 9);
  for (const k of ['mouth', 'width', 'eye']) assert.ok(Math.abs(a[k] - b[k]) < 1e-6, k);
});

test('degenerate landmarks return null and score 0', () => {
  const lm = Array.from({ length: 478 }, () => ({ x: 0.5, y: 0.5, z: 0 }));
  assert.equal(LF.getFaceFeatures(lm), null);
  assert.equal(LF.scoreFace(null).score, 0);
});

test('neutral face scores ~0 against its own baseline', () => {
  const s = LF.scoreFace(feats({}), neutralBase());
  assert.ok(s.score < 0.02, `got ${s.score}`);
});

test('wide-open mouth without smile (talking / yawning) stays below face-only threshold', () => {
  const s = LF.scoreFace(feats({ open: 0.2 }), neutralBase());
  assert.equal(s.open, 1);
  assert.ok(s.smile < 0.05);
  assert.ok(s.score <= LF.WEIGHTS.open + 1e-9 && s.score < 0.8, `got ${s.score}`);
});

test('full laugh face (open + smile + squint) exceeds the face-only threshold', () => {
  const s = LF.scoreFace(feats({ open: 0.15, widen: 1.25, lift: 0.01, eye: 0.6 }), neutralBase());
  assert.ok(s.smile > 0.9 && s.eye > 0.9, JSON.stringify(s));
  assert.ok(s.score >= 0.8, `got ${s.score}`);
});

test('closed-mouth grin scores between neutral and laugh', () => {
  const s = LF.scoreFace(feats({ widen: 1.2 }), neutralBase());
  assert.ok(s.score > 0.3 && s.score < 0.8, `got ${s.score}`);
});

test('corner lift counts as a smile cue', () => {
  const s = LF.scoreFace(feats({ lift: 0.01 }), neutralBase());   // 0.05 iod
  assert.ok(s.smile > 0.9, `got ${s.smile}`);
});

test('without calibration, relative cues are skipped instead of misfiring', () => {
  const s = LF.scoreFace(feats({ widen: 1.3, eye: 0.5 }));
  assert.equal(s.eye, 0);
  assert.ok(s.score < 0.8);
});

test('calibrate uses a trimmed mean (robust to a few outlier frames)', () => {
  const samples = [...Array.from({ length: 18 }, () => feats({})), feats({ open: 0.3 }), feats({ open: 0.3 }), null];
  const b = LF.calibrate(samples);
  assert.ok(Math.abs(b.mouth - feats({}).mouth) < 1e-9, `mouth baseline ${b.mouth}`);
  assert.deepEqual(LF.calibrate([]), { ...LF.DEFAULT_BASELINE });
});
