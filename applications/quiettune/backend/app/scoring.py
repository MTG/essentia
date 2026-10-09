import numpy as np

LABELS = {"aggressive": "模型预测激烈程度", "unrelaxed": "模型预测放松程度反向指标",
          "loudness": "响度", "brightness": "高频刺激代理", "density": "瞬态密集程度", "flatness": "噪声状频谱代理"}


def scale(value: float | None, low: float, high: float) -> float:
    return float(np.clip(((value if value is not None else low) - low) / (high - low), 0, 1))


def dimensions(features: dict, ai: dict | None = None) -> dict:
    if features.get("silent"):
        return {key: 0.0 for key in LABELS if key not in {"aggressive", "unrelaxed"}}
    result = {
        "loudness": scale(features.get("lufs", features.get("rms_dbfs")), -35, -8),
        "brightness": 0.55 * scale(features.get("high_frequency_ratio"), 0.02, 0.5)
                      + 0.45 * scale(features.get("centroid_hz"), 600, 5000),
        "density": 0.75 * scale(features.get("onset_density"), 0, 6)
                   + 0.25 * scale(features.get("onset_strength"), 0, 0.2),
        "flatness": scale(features.get("flatness"), 0.01, 0.45),
    }
    if ai and ai.get("status") == "ready":
        result.update(aggressive=ai["aggressive"], unrelaxed=1 - ai["relaxed"])
    return result


def weighted(dims: dict, config: dict) -> float:
    denominator = sum(config["weights"][key] for key in dims)
    return 100 * sum(config["weights"][key] * value for key, value in dims.items()) / denominator if denominator else 0.0


def score(features: dict, segments: list[dict], config: dict) -> dict:
    ai = features.get("ai") if config["ai_enabled"] else None
    dims = dimensions(features, ai)
    local = [{**segment, "score": round(weighted(dimensions(segment), config), 2)} for segment in segments]
    if not local:
        return {"average": 0, "peak": 0, "combined": 0, "segments": [], "dimensions": dims, "reasons": []}
    average = weighted(dims, config)
    peak = max(local, key=lambda s: s["score"])
    quiet = min(local, key=lambda s: s["score"])
    duration = sum(s["end"] - s["start"] for s in local)
    combined = average + config["peak_mix"] * max(0, peak["score"] - average)
    factors = sorted(dims.items(), key=lambda x: config["weights"][x[0]] * x[1], reverse=True)
    reasons = [f"{LABELS[key]}贡献较高（归一化指标 {value * 100:.0f}/100）。" for key, value in factors[:3] if value > 0.25]
    reasons.append(f"声学指数最高片段位于 {peak['start']:.1f}–{peak['end']:.1f} 秒，评分 {peak['score']:.1f}。")
    if peak["score"] - quiet["score"] > 20:
        reasons.append("片段间声学指数差异明显，建议试听峰值片段后再判断舒适度。")
    if features.get("silent"):
        reasons = ["检测到数字静音，声学指数为 0；静音不能代表音乐情绪。"]
    return {"average": round(average, 2), "peak": peak["score"], "combined": round(combined, 2),
            "high_ratio": sum(s["end"] - s["start"] for s in local if s["score"] >= config["threshold"]) / duration,
            "loudest": {"start": peak["start"], "end": peak["end"]},
            "quietest": {"start": quiet["start"], "end": quiet["end"]},
            "dimensions": dims, "reasons": reasons, "segments": local,
            "mode": "AI 与声学综合评分" if ai and ai.get("status") == "ready" and not features.get("silent") else "仅声学特征评分"}


def personal_vector(scored: dict) -> list[float]:
    dims = scored["dimensions"]
    return [dims.get(key, 0) for key in LABELS] + [scored["peak"] / 100, scored.get("high_ratio", 0)]


def personal_score(scored: dict, model: dict) -> float | None:
    if model.get("status") != "ready":
        return None
    vector = np.array(personal_vector(scored))
    prediction = float(np.dot((vector - model["mean"]) / model["scale"], model["coef"]) + model["intercept"])
    return round(float(np.clip(0.65 * prediction + 0.35 * scored["combined"], 0, 100)), 2)
