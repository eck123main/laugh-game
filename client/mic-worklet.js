// AudioWorklet that forwards microphone samples to the main thread.
// Unlike the deprecated ScriptProcessorNode, capture runs on the audio
// rendering thread: if the main thread is busy (FaceMesh inference, layout),
// chunks queue up in the message port instead of being dropped, so the
// log-mel stream stays gap-free. Render quanta (128 samples) are batched into
// CHUNK-sample buffers to keep the message rate low (~8 msgs/s at 8 kHz).
const CHUNK = 1024;

class MicCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buf = new Float32Array(CHUNK);
    this.fill = 0;
  }

  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (ch) {
      let i = 0;
      while (i < ch.length) {
        const n = Math.min(ch.length - i, CHUNK - this.fill);
        this.buf.set(ch.subarray(i, i + n), this.fill);
        this.fill += n; i += n;
        if (this.fill === CHUNK) {
          this.port.postMessage(this.buf, [this.buf.buffer]);
          this.buf = new Float32Array(CHUNK);
          this.fill = 0;
        }
      }
    }
    return true;
  }
}

registerProcessor('mic-capture', MicCaptureProcessor);
