const fs = require('node:fs');
const path = require('node:path');
const runtime = process.env.QUIETTUNE_ESSENTIA_DIST;
const EssentiaWASM = require(runtime ? path.join(runtime, 'essentia-wasm.umd.js') : 'essentia.js/dist/essentia-wasm.umd.js');
const Essentia = require(runtime ? path.join(runtime, 'essentia.js-core.umd.js') : 'essentia.js/dist/essentia.js-core.umd.js');
async function main() {
if (!EssentiaWASM.calledRun) await new Promise(resolve => { EssentiaWASM.onRuntimeInitialized = resolve; });
const engine = new Essentia(EssentiaWASM);
const input = fs.readFileSync(process.argv[2]);
const audio = new Float32Array(input.buffer, input.byteOffset, input.length / 4);
const frames = [];
// 与官方 FrameCutter 的中心起始模式一致，首帧左侧补零。
for (let center = 0; center < audio.length; center += 256) {
  const frame = new Float32Array(512);
  for (let j = 0; j < 512; j++) {
    const index = center - 256 + j;
    if (index >= 0 && index < audio.length) frame[j] = audio[index];
  }
  const vector = engine.arrayToVector(frame);
  const bands = engine.TensorflowInputMusiCNN(vector).bands;
  frames.push(engine.vectorToArray(bands));
  vector.delete();
  bands.delete();
}
const output = new Float32Array(frames.length * 96);
frames.forEach((frame, i) => output.set(frame, i * 96));
fs.writeFileSync(process.argv[3], Buffer.from(output.buffer));
engine.shutdown();
}
main().catch(error => { console.error('Essentia 输入计算失败：' + error.message); process.exitCode = 1; });
