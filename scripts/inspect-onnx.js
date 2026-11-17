#!/usr/bin/env node
// Minimal ONNX inspector (no dependencies): prints graph inputs/outputs,
// op-type histogram, and initializer shapes. Usage:
//   node scripts/inspect-onnx.js client/models/laughter_resnet.onnx [--nodes]
'use strict';
const fs = require('fs');

function readVarint(buf, pos) {
  let result = 0n, shift = 0n, b;
  do {
    b = buf[pos.i++];
    result |= BigInt(b & 0x7f) << shift;
    shift += 7n;
  } while (b & 0x80);
  return result;
}

// Parse a protobuf message into { fieldNo: [values...] }; length-delimited
// values are returned as Buffers, varints as BigInt.
function parse(buf) {
  const out = {};
  const pos = { i: 0 };
  while (pos.i < buf.length) {
    const key = Number(readVarint(buf, pos));
    const field = key >> 3, wire = key & 7;
    let val;
    if (wire === 0) val = readVarint(buf, pos);
    else if (wire === 1) { val = buf.subarray(pos.i, pos.i + 8); pos.i += 8; }
    else if (wire === 2) {
      const len = Number(readVarint(buf, pos));
      val = buf.subarray(pos.i, pos.i + len); pos.i += len;
    } else if (wire === 5) { val = buf.subarray(pos.i, pos.i + 4); pos.i += 4; }
    else throw new Error('unsupported wire type ' + wire);
    (out[field] = out[field] || []).push(val);
  }
  return out;
}

const str = (b) => (b ? b.toString('utf8') : '');

function valueInfo(buf) {
  const vi = parse(buf);
  const name = str(vi[1] && vi[1][0]);
  let dims = [];
  if (vi[2]) {
    const tp = parse(vi[2][0]);
    if (tp[1]) {
      const tt = parse(tp[1][0]);
      if (tt[2]) {
        const shape = parse(tt[2][0]);
        dims = (shape[1] || []).map((d) => {
          const dd = parse(d);
          if (dd[1]) return Number(dd[1][0]);
          if (dd[2]) return str(dd[2][0]);
          return '?';
        });
      }
    }
  }
  return { name, dims };
}

function main() {
  const file = process.argv[2];
  const showNodes = process.argv.includes('--nodes');
  const model = parse(fs.readFileSync(file));
  const graph = parse(model[7][0]);
  const inits = new Set();
  console.log('initializers:');
  for (const b of graph[5] || []) {
    const t = parse(b);
    const dims = (t[1] || []).map(Number);
    const name = str(t[8] && t[8][0]);
    inits.add(name);
    console.log('  ', name, JSON.stringify(dims));
  }
  console.log('inputs:');
  for (const b of graph[11] || []) {
    const v = valueInfo(b);
    if (!inits.has(v.name)) console.log('  ', v.name, JSON.stringify(v.dims));
  }
  console.log('outputs:');
  for (const b of graph[12] || []) console.log('  ', JSON.stringify(valueInfo(b)));
  const hist = {};
  for (const b of graph[1] || []) {
    const n = parse(b);
    const op = str(n[4] && n[4][0]);
    hist[op] = (hist[op] || 0) + 1;
    if (showNodes) {
      console.log('node', op, (n[1] || []).map(str), '->', (n[2] || []).map(str));
    }
  }
  console.log('ops:', JSON.stringify(hist));
}

main();
