'use strict';
// Compile-only smoke test: catches syntax errors in browser scripts that
// can't be executed under Node (they need DOM / socket.io / MediaPipe).
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

for (const file of ['game.js', 'laugh-audio.js', 'laugh-face.js', 'mic-worklet.js']) {
  test(`client/${file} parses`, () => {
    const src = fs.readFileSync(path.join(__dirname, '..', 'client', file), 'utf8');
    assert.doesNotThrow(() => new vm.Script(src, { filename: file }));
  });
}

test('index.html loads laugh-face.js before game.js', () => {
  const html = fs.readFileSync(path.join(__dirname, '..', 'client', 'index.html'), 'utf8');
  const face = html.indexOf('src="laugh-face.js"'), game = html.indexOf('src="game.js"');
  assert.ok(face !== -1 && face < game);
});
