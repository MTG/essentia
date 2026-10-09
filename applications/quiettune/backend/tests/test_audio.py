import copy
import subprocess

import numpy as np
import pytest
import soundfile as sf

from backend.app.audio import decode, extract, loudness, probe
from backend.app.config import DEFAULTS
from backend.app.models import predict, validate_metadata
from backend.app.scoring import score

SR = 22050


def tone(frequency=220, seconds=6, amplitude=0.1):
    time = np.arange(int(SR * seconds)) / SR
    return (amplitude * np.sin(2 * np.pi * frequency * time)).astype(np.float32)


def test_silence_is_zero():
    """数字静音不能被频谱平坦度误判为噪声。"""
    features, segments = extract(np.zeros(SR * 6, dtype=np.float32), SR, 5)
    result = score(features, segments, DEFAULTS)
    assert result["combined"] == 0
    assert result["peak"] == 0
    assert features["rms_dbfs"] is None


def test_brightness_and_gain_invariance():
    """高频变化影响刺激代理，母带增益只影响较小的响度维度。"""
    low, low_segments = extract(tone(), SR, 5)
    high, high_segments = extract(tone(6000), SR, 5)
    loud, loud_segments = extract(tone(amplitude=0.4), SR, 5)
    assert high["centroid_hz"] > low["centroid_hz"] + 4000
    assert high["high_frequency_ratio"] > 0.99
    assert low["high_frequency_ratio"] < 0.01
    assert score(high, high_segments, DEFAULTS)["combined"] > score(low, low_segments, DEFAULTS)["combined"] + 15
    assert abs(score(loud, loud_segments, DEFAULTS)["combined"] - score(low, low_segments, DEFAULTS)["combined"]) < 8


def test_segments_and_loudest_position():
    """安静前奏后的高频密集片段应被定位，尾段不得丢失。"""
    audio = np.r_[tone(seconds=5), tone(6000, seconds=5), tone(seconds=0.3)]
    features, segments = extract(audio, SR, 5)
    result = score(features, segments, DEFAULTS)
    assert len(segments) == 3
    assert segments[-1]["end"] == pytest.approx(10.3)
    assert result["loudest"]["start"] == 5
    assert result["combined"] >= result["average"]
    assert sum(s["end"] - s["start"] for s in segments) == pytest.approx(10.3)


def test_onset_density_detects_events():
    """密集固定瞬态应比连续纯音包含更多起始事件。"""
    audio = np.zeros(SR * 6, dtype=np.float32)
    burst = tone(2000, seconds=0.025) * np.exp(-np.arange(int(SR * 0.025)) / 90)
    for start in range(0, len(audio) - len(burst), SR // 5):
        audio[start:start + len(burst)] = burst
    features, _ = extract(audio, SR, 5)
    smooth, _ = extract(tone(), SR, 5)
    assert features["onset_density"] > smooth["onset_density"] + 2


def test_real_lufs_gain_difference(tmp_path):
    """FFmpeg 实测振幅翻倍应增加约 6 dB 响度。"""
    first, second = tmp_path / "first.wav", tmp_path / "second.wav"
    sf.write(first, tone(amplitude=0.1), SR)
    sf.write(second, tone(amplitude=0.2), SR)
    a, b = loudness(first), loudness(second)
    assert b["lufs"] - a["lufs"] == pytest.approx(6, abs=0.3)
    assert a["true_peak_dbfs"] is not None
    assert a["short_term_loudness"]


def test_disabled_ai_never_fakes_probabilities():
    result = predict(np.zeros(16000, dtype=np.float32), False)
    assert result["status"] == "disabled"
    assert result["aggressive"] is None and result["relaxed"] is None


def test_model_contract_rejects_wrong_dimensions():
    meta = {"inference": {"sample_rate": 16000}, "classes": ["relaxed", "non_relaxed"], "schema": {"inputs": [{"shape": [200]}]}}
    validate_metadata(meta, "relaxed")
    invalid = copy.deepcopy(meta)
    invalid["schema"]["inputs"][0]["shape"] = [128]
    with pytest.raises(ValueError):
        validate_metadata(invalid, "relaxed")


def test_real_official_model_inference():
    """固定合成输入只验证真实模型管线，不证明真实音乐分类准确率。"""
    time = np.arange(16000 * 4) / 16000
    result = predict((0.1 * np.sin(2 * np.pi * 440 * time)).astype(np.float32), True)
    assert result["status"] == "ready", result.get("reason")
    assert 0 <= result["aggressive"] <= 1 and 0 <= result["relaxed"] <= 1
    assert result["models"]["relaxed"]["classes"] == ["non_relaxed", "relaxed"]
    assert result["embedding_dimension"] == 200


@pytest.mark.parametrize("extension", ["mp3", "flac", "m4a", "ogg"])
def test_supported_formats_decode(tmp_path, extension):
    """通过实际转码的固定音频验证常见容器格式，不使用虚构特征。"""
    source = tmp_path / "source.wav"
    target = tmp_path / f"target.{extension}"
    sf.write(source, tone(), SR)
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", str(source), str(target)], check=True)
    assert probe(target)["duration"] >= 6
    audio = decode(target)
    assert len(audio) >= SR * 5.9
    feature, _ = extract(audio, SR, 5)
    assert feature["centroid_hz"] == pytest.approx(220, abs=50)
