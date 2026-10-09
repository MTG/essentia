import hashlib
import json
import os
import platform
import subprocess
import tempfile
import threading
from pathlib import Path

import numpy as np

from .config import MODELS, NODE

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
MODEL_SPECS = {
    "msd-musicnn-1": "feature-extractors/musicnn",
    "mood_aggressive-msd-musicnn-1": "classification-heads/mood_aggressive",
    "mood_relaxed-msd-musicnn-1": "classification-heads/mood_relaxed",
}
INFERENCE_LOCK = threading.Lock()


def model_status() -> dict:
    missing = [name for name in MODEL_SPECS if not (MODELS / f"{name}.pb").exists() or not (MODELS / f"{name}.json").exists()]
    return {"status": "missing" if missing else "available", "missing": missing,
            "reason": "请运行模型下载命令。" if missing else "权重已缓存；实际推理状态以歌曲分析结果为准。",
            "runtime": "Essentia WASM + TensorFlow" if platform.system() == "Windows" else "Essentia Python + TensorFlow",
            "path": str(MODELS)}


def validate_metadata(metadata: dict, target: str | None = None):
    if metadata["inference"]["sample_rate"] != 16000:
        raise ValueError("模型采样率与分析协议不匹配。")
    if target and (target not in metadata["classes"] or metadata["schema"]["inputs"][0]["shape"] != [200]):
        raise ValueError("分类器类别或嵌入维度不匹配。")


def _mel(audio: np.ndarray) -> np.ndarray:
    if platform.system() != "Windows":
        import essentia.standard as es
        algorithm = es.TensorflowInputMusiCNN()
        return np.array([algorithm(frame) for frame in es.FrameGenerator(audio, frameSize=512, hopSize=256)], dtype=np.float32)
    with tempfile.TemporaryDirectory(prefix="quiettune-") as temp:
        source, destination = Path(temp) / "audio.f32", Path(temp) / "mel.f32"
        audio.astype("<f4").tofile(source)
        result = subprocess.run([NODE, str(Path(__file__).with_name("essentia_bridge.cjs")), str(source), str(destination)],
                                capture_output=True, timeout=180)
        if result.returncode:
            raise RuntimeError("Essentia WASM 输入计算失败：" + result.stderr.decode(errors="replace")[-500:])
        return np.fromfile(destination, dtype="<f4").reshape(-1, 96)


def _run_graph(tf, name: str, values: np.ndarray, output_name: str | None = None) -> np.ndarray:
    metadata = json.loads((MODELS / f"{name}.json").read_text("utf-8"))
    graph = tf.Graph()
    definition = tf.compat.v1.GraphDef()
    definition.ParseFromString((MODELS / f"{name}.pb").read_bytes())
    with graph.as_default():
        tf.import_graph_def(definition, name="")
    source = graph.get_tensor_by_name(metadata["schema"]["inputs"][0]["name"] + ":0")
    destination = graph.get_tensor_by_name((output_name or metadata["schema"]["outputs"][0]["name"]) + ":0")
    feeds = {source: values}
    try:
        feeds[graph.get_tensor_by_name("model/is_training:0")] = False
    except KeyError:
        pass
    config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=2, inter_op_parallelism_threads=1, device_count={"GPU": 0})
    with tf.compat.v1.Session(graph=graph, config=config) as session:
        return np.concatenate([session.run(destination, feed_dict={**feeds, source: values[i:i + 16]}) for i in range(0, len(values), 16)])


def predict(audio_16k: np.ndarray, enabled: bool) -> dict:
    if not enabled:
        return {"status": "disabled", "reason": "设置中已关闭 AI；当前使用真实声学特征。", "aggressive": None, "relaxed": None}
    status = model_status()
    if status["missing"]:
        return {**status, "aggressive": None, "relaxed": None}
    try:
        with INFERENCE_LOCK:
            import tensorflow as tf
            metadata = {name: json.loads((MODELS / f"{name}.json").read_text("utf-8")) for name in MODEL_SPECS}
            validate_metadata(metadata["msd-musicnn-1"])
            if metadata["msd-musicnn-1"]["schema"]["inputs"][0]["shape"] != [187, 96]:
                raise ValueError("MusiCNN 输入维度不匹配。")
            mel = _mel(audio_16k)
            patches = []
            for start in range(0, len(mel), 187):
                patch = mel[start:start + 187]
                if len(patch) < 187:
                    patch = np.tile(patch, (int(np.ceil(187 / len(patch))), 1))[:187]
                patches.append(patch)
            patches = np.array(patches, dtype=np.float32)
            embeddings = _run_graph(tf, "msd-musicnn-1", patches, "model/dense/BiasAdd")
            if embeddings.shape != (len(patches), 200):
                raise ValueError("嵌入输出维度不匹配。")
            result = {"status": "ready", "scope": "whole_track", "patches": len(patches),
                      "runtime": status["runtime"], "tensorflow": tf.__version__, "sample_rate": 16000,
                      "input_shape": [187, 96], "embedding_dimension": 200, "last_patch_mode": "repeat", "patch_hop": 187,
                      "models": {}, "embedding_sha256": hashlib.sha256((MODELS / "msd-musicnn-1.pb").read_bytes()).hexdigest()}
            for target in ("aggressive", "relaxed"):
                name = f"mood_{target}-msd-musicnn-1"
                meta = metadata[name]
                validate_metadata(meta, target)
                output = _run_graph(tf, name, embeddings)
                if output.shape != (len(patches), 2) or not np.all(np.isfinite(output)):
                    raise ValueError("分类器输出无效。")
                probability = float(output[:, meta["classes"].index(target)].mean())
                if not 0 <= probability <= 1:
                    raise ValueError("分类器概率超出有效范围。")
                result[target] = probability
                result["models"][target] = {"name": name, "metadata_version": meta["version"], "classes": meta["classes"],
                                            "sha256": hashlib.sha256((MODELS / f"{name}.pb").read_bytes()).hexdigest()}
            return result
    except Exception as exc:
        return {"status": "unavailable", "reason": f"真实模型推理失败：{exc}", "aggressive": None, "relaxed": None}
