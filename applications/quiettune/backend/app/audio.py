import json
import re
import subprocess
from pathlib import Path

import librosa
import numpy as np

from .config import ANALYZER_VERSION, FFMPEG, FFPROBE, MAX_SECONDS
from .models import predict

SR = 22050
HOP = 512


def probe(path: Path) -> dict:
    result = subprocess.run([FFPROBE, "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)],
                            capture_output=True, timeout=30)
    if result.returncode:
        raise ValueError("无法解码该音频，请检查文件是否损坏。")
    data = json.loads(result.stdout)
    streams = [stream for stream in data.get("streams", []) if stream.get("codec_type") == "audio"]
    if not streams:
        raise ValueError("文件没有可用的音频轨。")
    duration = float(data.get("format", {}).get("duration", streams[0].get("duration", 0)))
    if not 0.05 <= duration <= MAX_SECONDS:
        raise ValueError(f"音频时长必须在 0.05 秒到 {MAX_SECONDS // 60} 分钟之间。")
    channels = int(streams[0].get("channels", 1))
    if channels > 2:
        raise ValueError("目前支持单声道和立体声音频，请先转换多声道文件。")
    tags = {key.lower(): value for key, value in {**streams[0].get("tags", {}), **data.get("format", {}).get("tags", {})}.items()}
    return {"duration": duration, "tags": tags,
            "cover": any(s.get("disposition", {}).get("attached_pic") for s in data.get("streams", []))}


def decode(path: Path, sample_rate: int = SR) -> np.ndarray:
    result = subprocess.run([FFMPEG, "-v", "error", "-nostdin", "-i", str(path), "-t", str(MAX_SECONDS),
                             "-map", "0:a:0", "-ac", "1", "-ar", str(sample_rate), "-f", "f32le", "pipe:1"],
                            capture_output=True, timeout=180)
    if result.returncode or not result.stdout:
        raise ValueError("音频解码失败，请确认 FFmpeg 可用且文件未损坏。")
    audio = np.frombuffer(result.stdout, dtype="<f4").copy()
    if not np.all(np.isfinite(audio)):
        raise ValueError("音频包含无效采样值。")
    return audio


def loudness(path: Path) -> dict:
    result = subprocess.run([FFMPEG, "-hide_banner", "-nostdin", "-i", str(path), "-t", str(MAX_SECONDS), "-map", "0:a:0", "-vn",
                             "-af", "ebur128=peak=true", "-f", "null", "-"], capture_output=True, timeout=180)
    if result.returncode:
        raise RuntimeError("FFmpeg EBU R128 响度测量失败。")
    log = result.stderr.decode(errors="replace")
    integrated = re.findall(r"I:\s+(-?\d+(?:\.\d+)?) LUFS", log)
    true_peak = re.findall(r"Peak:\s+(-?(?:\d+(?:\.\d+)?|inf)) dBFS", log)
    distribution = []
    for time, short in re.findall(r"t:\s*([\d.]+).*?S:\s*(-?[\d.]+)", log):
        value = float(short)
        if value > -70:
            distribution.append({"time": round(float(time), 2), "lufs": value})
    value = float(integrated[-1]) if integrated else None
    peak = float(true_peak[-1]) if true_peak else None
    return {"lufs": value if value is not None and value > -70 else None,
            "true_peak_dbfs": peak if peak is not None and np.isfinite(peak) else None,
            "short_term_loudness": distribution[::10]}


def extract(audio: np.ndarray, sample_rate: int, window_seconds: int) -> tuple[dict, list[dict]]:
    duration = len(audio) / sample_rate
    spectrum = np.abs(librosa.stft(audio, n_fft=2048, hop_length=HOP))
    power = spectrum ** 2
    energy = power.sum(axis=0)
    active = energy > 1e-8
    frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=2048)
    centroid = librosa.feature.spectral_centroid(S=spectrum, sr=sample_rate)[0]
    rolloff = librosa.feature.spectral_rolloff(S=spectrum, sr=sample_rate, roll_percent=0.85)[0]
    flatness = librosa.feature.spectral_flatness(S=spectrum)[0]
    high = np.divide(power[frequencies >= 4000].sum(axis=0), energy, out=np.zeros_like(energy), where=active)
    onset = librosa.onset.onset_strength(y=audio, sr=sample_rate, hop_length=HOP)
    times = librosa.frames_to_time(np.arange(spectrum.shape[1]), sr=sample_rate, hop_length=HOP)
    # 归一化频谱变化率避免母带增益成为瞬态强度的主因。
    norm = np.divide(spectrum, spectrum.sum(axis=0), out=np.zeros_like(spectrum), where=spectrum.sum(axis=0) > 1e-9)
    flux = np.r_[0, np.maximum(np.diff(norm, axis=1), 0).sum(axis=0)]
    # 极小的数值波动经峰值归一化会产生假事件；需同时满足绝对谱变化。
    event_frames = librosa.onset.onset_detect(onset_envelope=onset, sr=sample_rate, hop_length=HOP, units="frames") if active.any() else np.array([], dtype=int)
    event_frames = np.array([index for index in event_frames if onset[index] >= 0.5 and
                            np.max(flux[max(0, index - 3):min(len(flux), index + 2)]) >= 0.02], dtype=int)
    events = librosa.frames_to_time(event_frames, sr=sample_rate, hop_length=HOP)
    bands = [(0, 250), (250, 2000), (2000, 4000), (4000, sample_rate / 2 + 1)]

    def summarize(start: float, end: float) -> dict:
        samples = audio[int(start * sample_rate):min(len(audio), int(end * sample_rate))]
        mask = (times >= start) & (times < end)
        valid = mask & active
        rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2))) if len(samples) else 0
        peak = float(np.max(np.abs(samples))) if len(samples) else 0
        total = float(power[:, mask].sum())
        return {"rms": rms, "rms_dbfs": float(20 * np.log10(rms)) if rms > 1e-9 else None,
                "peak_dbfs": float(20 * np.log10(peak)) if peak > 1e-9 else None,
                "crest_db": float(20 * np.log10(peak / rms)) if rms > 1e-9 else None,
                "silent": rms < 1e-7,
                "centroid_hz": float(centroid[valid].mean()) if valid.any() else 0,
                "rolloff_hz": float(rolloff[valid].mean()) if valid.any() else 0,
                "flatness": float(flatness[valid].mean()) if valid.any() else 0,
                "high_frequency_ratio": float(high[valid].mean()) if valid.any() else 0,
                "onset_density": float(np.sum((events >= start) & (events < end)) / max(end - start, 1e-6)),
                "onset_strength": float(flux[valid].mean()) if valid.any() else 0,
                "band_energy": [float(power[(frequencies >= low) & (frequencies < upper)][:, mask].sum() / total) if total else 0 for low, upper in bands]}

    overall = summarize(0, duration)
    tempo = librosa.feature.tempo(onset_envelope=onset, sr=sample_rate, hop_length=HOP)[0] if len(events) >= 3 else None
    overall["bpm"] = float(tempo) if tempo is not None else None
    segments = [{"start": float(start), "end": float(min(start + window_seconds, duration)),
                 **summarize(float(start), float(min(start + window_seconds, duration)))} for start in np.arange(0, duration, window_seconds)]
    bins = np.array_split(audio, min(1200, len(audio)))
    overall["waveform"] = [round(float(np.max(np.abs(chunk))), 5) for chunk in bins]
    return overall, segments


def analyze(path: Path, config: dict, progress) -> tuple[dict, list[dict]]:
    progress(10, "解码音频")
    audio = decode(path)
    progress(25, "提取全曲与分段声学特征")
    features, segments = extract(audio, SR, config["window_seconds"])
    progress(55, "测量 EBU R128 响度")
    features.update(loudness(path))
    for segment in segments:
        values = [x["lufs"] for x in features["short_term_loudness"] if segment["start"] <= x["time"] < segment["end"]]
        if values:
            segment["lufs"] = float(np.median(values))
    progress(70, "运行官方情绪模型")
    features["ai"] = predict(decode(path, 16000), config["ai_enabled"])
    features["provenance"] = {"analyzer": ANALYZER_VERSION, "sample_rate": SR, "fft_size": 2048,
                               "hop_size": HOP, "window_seconds": config["window_seconds"], "librosa": librosa.__version__,
                               "loudness_engine": "FFmpeg ebur128", "analysis_config": config}
    progress(95, "保存分析结果")
    return features, segments
