'use strict';
// Runs client/mic-worklet.js in a VM with a fake AudioWorklet global scope.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function loadProcessor() {
  const registered = {};
  const ctx = {
    Float32Array,
    AudioWorkletProcessor: class { constructor() { this.port = { sent: [], postMessage(m) { this.sent.push(Float32Array.from(m)); } }; } },
    registerProcessor: (name, cls) => { registered[name] = cls; },
  };
  const src = fs.readFileSync(path.join(__dirname, '..', 'client', 'mic-worklet.js'), 'utf8');
  vm.runInNewContext(src, ctx);
  assert.ok(registered['mic-capture'], 'registers mic-capture');
  return new registered['mic-capture']();
}

test('mic worklet batches 128-sample quanta into 1024-sample chunks, in order', () => {
  const p = loadProcessor();
  let t = 0;
  for (let q = 0; q < 20; q++) {
    const quantum = new Float32Array(128);
    for (let i = 0; i < 128; i++) quantum[i] = t++;
    assert.equal(p.process([[quantum]]), true, 'keeps the node alive');
  }
  const sent = p.port.sent;
  assert.equal(sent.length, 2, '2560 samples → 2 full chunks');
  sent.forEach((chunk, c) => {
    assert.equal(chunk.length, 1024);
    for (let i = 0; i < 1024; i++) assert.equal(chunk[i], c * 1024 + i);
  });
});

test('mic worklet tolerates a missing input (disconnected source)', () => {
  const p = loadProcessor();
  assert.equal(p.process([[]]), true);
  assert.equal(p.process([]), true);
  assert.equal(p.port.sent.length, 0);
});
