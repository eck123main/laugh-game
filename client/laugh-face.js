// ─── Facial laughter score from MediaPipe FaceMesh landmarks ────────────────
// FACS describes the laughing / amusement face as AU6 (cheek raiser → eyes
// narrow) + AU12 (lip-corner puller → mouth widens, corners lift) + AU25/26
// (lips part / jaw drop). AU25/26 alone is also talking, yawning or surprise,
// so a wide-open mouth by itself must NOT reach the face-only trigger: the
// smile (AU12) and eye-narrowing (AU6) cues carry most of the weight.
//
// Every distance is divided by the inter-ocular distance (outer eye corners)
// so features don't change when the player leans towards / away from the
// camera, and each cue is scored relative to the player's own neutral
// baseline (captured during the pre-round calibration).
//
// Works in the browser (window.LaughFace) and in Node (module.exports).
(function (root) {
  'use strict';

  const LM = {
    UPPER_LIP: 13, LOWER_LIP: 14,               // inner lip midpoints
    INNER_LEFT: 78, INNER_RIGHT: 308,           // inner mouth corners
    MOUTH_LEFT: 61, MOUTH_RIGHT: 291,           // outer mouth corners
    EYE_OUTER_L: 33, EYE_OUTER_R: 263,          // outer eye corners (scale ref)
    L_EYE_TOP: 159, L_EYE_BOT: 145, R_EYE_TOP: 386, R_EYE_BOT: 374,
  };

  // Full-scale deltas (feature change that maps to a cue score of 1).
  const SCALE = {
    // Mouth height/width ratio above baseline. 0.165 here == the old 0.22 on
    // the un-aspect-corrected ratio (×0.75 for a 4:3 detection canvas).
    open:  0.165,
    width: 0.15,   // +15 % mouth width vs. neutral
    lift:  0.05,   // corner lift (in inter-ocular units) vs. neutral
    squint: 0.30,  // −30 % eye aperture vs. neutral
  };
  const WEIGHTS = { open: 0.45, smile: 0.40, eye: 0.15 };

  const clamp01 = (v) => (v > 0 ? (v < 1 ? v : 1) : 0);

  // `aspect` = image width / height. FaceMesh x and y are normalised to the
  // image width and height separately, so x must be rescaled before mixing.
  function getFaceFeatures(lm, aspect = 4 / 3) {
    const dist = (a, b) => Math.hypot((a.x - b.x) * aspect, a.y - b.y);
    const iod = dist(lm[LM.EYE_OUTER_L], lm[LM.EYE_OUTER_R]);
    if (!(iod > 1e-4)) return null;

    const mouthH = dist(lm[LM.UPPER_LIP], lm[LM.LOWER_LIP]);
    const innerW = dist(lm[LM.INNER_LEFT], lm[LM.INNER_RIGHT]);
    const mouth  = innerW > 1e-4 ? mouthH / innerW : 0;

    const width = dist(lm[LM.MOUTH_LEFT], lm[LM.MOUTH_RIGHT]) / iod;
    // Corners above the lip centre → positive. Averaging both corners cancels
    // most of the effect of head roll.
    const lipMidY    = (lm[LM.UPPER_LIP].y + lm[LM.LOWER_LIP].y) / 2;
    const cornerMidY = (lm[LM.MOUTH_LEFT].y + lm[LM.MOUTH_RIGHT].y) / 2;
    const lift = (lipMidY - cornerMidY) / iod;

    const eye = (dist(lm[LM.L_EYE_TOP], lm[LM.L_EYE_BOT]) +
                 dist(lm[LM.R_EYE_TOP], lm[LM.R_EYE_BOT])) / 2 / iod;

    return { mouth, width, lift, eye, iod };
  }

  // Neutral face used before (or without) calibration. width/eye are null:
  // their cues are relative, so without a baseline they are simply skipped.
  const DEFAULT_BASELINE = Object.freeze({ mouth: 0.02, width: null, lift: 0, eye: null });

  // A relaxed closed/slightly-parted mouth has height/width ≲ 0.05.
  const MAX_BASELINE_MOUTH = 0.08;

  // Robust baseline: 20–80 % trimmed mean of each feature over the samples.
  function calibrate(samples) {
    const valid = samples.filter(Boolean);
    if (!valid.length) return { ...DEFAULT_BASELINE };
    const trimmed = (key) => {
      const s = valid.map((f) => f[key]).sort((a, b) => a - b);
      const lo = Math.floor(s.length * 0.2), hi = Math.max(Math.ceil(s.length * 0.8), lo + 1);
      const slice = s.slice(lo, hi);
      return slice.reduce((a, b) => a + b, 0) / slice.length;
    };
    // A player already laughing during calibration would otherwise get an
    // open-mouth "neutral" and become undetectable for the whole round.
    return {
      mouth: Math.min(trimmed('mouth'), MAX_BASELINE_MOUTH),
      width: trimmed('width'), lift: trimmed('lift'), eye: trimmed('eye'),
    };
  }

  function scoreFace(f, base = DEFAULT_BASELINE) {
    if (!f) return { score: 0, open: 0, smile: 0, eye: 0 };
    const open = clamp01((f.mouth - base.mouth) / SCALE.open);
    const widen = base.width ? clamp01((f.width / base.width - 1) / SCALE.width) : 0;
    const lift  = clamp01((f.lift - (base.lift || 0)) / SCALE.lift);
    const smile = Math.max(widen, lift);
    const eye   = base.eye ? clamp01((1 - f.eye / base.eye) / SCALE.squint) : 0;
    const score = WEIGHTS.open * open + WEIGHTS.smile * smile + WEIGHTS.eye * eye;
    return { score, open, smile, eye };
  }

  const api = { LM, SCALE, WEIGHTS, DEFAULT_BASELINE, getFaceFeatures, calibrate, scoreFace };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.LaughFace = api;
})(typeof window !== 'undefined' ? window : globalThis);
