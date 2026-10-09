import csv
import hashlib
import io
import json
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from .audio import probe
from .config import DATA, DEFAULTS, EXTENSIONS, FFMPEG, MAX_BYTES, MODELS
from .db import (PERSONAL_PROFILE_ID, AnalysisJob, AudioFeatures, PreferenceProfile, ScoringVersion, SegmentFeatures,
                 Session, Track, UserFeedback, init_db, now, settings)
from .jobs import queue
from .models import model_status
from .scoring import LABELS, personal_score, personal_vector, score


@asynccontextmanager
async def lifespan(_):
    init_db()
    queue.start()
    yield
    queue.stop()


app = FastAPI(title="QuietTune 音乐舒适度分析", lifespan=lifespan)


@app.exception_handler(HTTPException)
async def application_error(request: Request, exc: HTTPException):
    from fastapi.responses import JSONResponse
    return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)


def track_or_404(session, identity):
    track = session.get(Track, identity)
    if not track:
        raise HTTPException(404, "歌曲不存在。")
    return track


def invalidate_model(session, reason="反馈或评分配置已更新，请重新训练。"):
    profile = session.get(PreferenceProfile, PERSONAL_PROFILE_ID)
    profile.data = {**profile.data, "status": "stale", "reason": reason}


def records(session) -> list[dict]:
    config = settings(session)
    version = session.scalar(select(func.max(ScoringVersion.id)))
    profile = session.get(PreferenceProfile, PERSONAL_PROFILE_ID).data
    # 先读取任务状态，再读取特征；完成状态和特征由同一事务提交，避免先读旧特征。
    latest = {}
    for job in session.scalars(select(AnalysisJob).order_by(AnalysisJob.created_at)):
        latest[job.track_id] = job
    features = {f.track_id: f for f in session.scalars(select(AudioFeatures))}
    segments = {}
    for segment in session.scalars(select(SegmentFeatures).order_by(SegmentFeatures.start)):
        segments.setdefault(segment.track_id, []).append(segment.data)
    feedback = {f.track_id: f for f in session.scalars(select(UserFeedback))}
    result = []
    for track in session.scalars(select(Track).order_by(Track.created_at.desc())):
        feature = features.get(track.id)
        scored = score(feature.data, segments.get(track.id, []), config) if feature else None
        rating = feedback.get(track.id)
        job = latest.get(track.id)
        result.append({"id": track.id, "title": track.title, "filename": track.filename, "artist": track.artist,
                       "album": track.album, "duration": track.duration, "relative_path": track.relative_path,
                       "created_at": track.created_at, "has_cover": track.cover,
                       "status": job.status if job else "pending", "job": job_data(job) if job else None,
                       "score": scored, "personal_score": personal_score(scored, profile) if scored else None,
                       "scoring_version": version, "features": feature.data if feature else None,
                       "analyzed_at": feature.analyzed_at if feature else None,
                       "feedback": {"rating": rating.rating, "factors": rating.factors, "note": rating.note,
                                    "updated_at": rating.updated_at} if rating else None})
    return result


def job_data(job):
    return {"id": job.id, "track_id": job.track_id, "status": job.status, "progress": job.progress,
            "stage": job.stage, "error": job.error, "created_at": job.created_at, "finished_at": job.finished_at}


@app.get("/api/health")
def health():
    return {"status": "ok", "ffmpeg": bool(shutil.which(FFMPEG)), "models": model_status(),
            "storage_path": str(DATA), "model_path": str(MODELS)}


@app.post("/api/tracks", status_code=201)
def upload(file: UploadFile, relative_path: str = ""):
    original = Path((file.filename or "音频").replace("\\", "/")).name
    extension = Path(original).suffix.lower()
    if extension not in EXTENSIONS:
        raise HTTPException(415, "支持 MP3、FLAC、WAV、M4A、OGG 音频文件。")
    audio_dir = DATA / "audio"
    audio_dir.mkdir(exist_ok=True)
    temporary = audio_dir / f"{uuid.uuid4().hex}.upload"
    digest = hashlib.sha256()
    size = 0
    target = None
    cover_path = None
    committed = False
    try:
        with temporary.open("wb") as output:
            while chunk := file.file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_BYTES:
                    raise HTTPException(413, "单个音频文件不能超过 200 MB。")
                digest.update(chunk)
                output.write(chunk)
        if not size:
            raise HTTPException(400, "不能上传空文件。")
        try:
            info = probe(temporary)
        except (ValueError, OSError) as exc:
            raise HTTPException(422, str(exc)) from exc
        content_hash = digest.hexdigest()
        with queue.lock, Session.begin() as session:
            existing = session.scalar(select(Track).where(Track.sha256 == content_hash))
            if existing:
                return {"id": existing.id, "duplicate": True, "message": "该音频已在音乐库中。"}
            target = audio_dir / f"{content_hash}{extension}"
            temporary.replace(target)
            identity = uuid.uuid4().hex
            track = Track(id=identity, sha256=content_hash, filename=original, path=target.name, duration=info["duration"],
                          title=info["tags"].get("title", Path(original).stem), artist=info["tags"].get("artist", "未知艺术家"),
                          album=info["tags"].get("album", "未标注专辑"), relative_path=relative_path[:1000])
            if info["cover"]:
                import subprocess
                covers = DATA / "covers"
                covers.mkdir(exist_ok=True)
                cover_path = covers / f"{identity}.jpg"
                converted = subprocess.run([FFMPEG, "-v", "error", "-nostdin", "-i", str(target), "-map", "0:v:0",
                                            "-frames:v", "1", "-vf", "scale=400:400:force_original_aspect_ratio=decrease", str(cover_path)],
                                           capture_output=True, timeout=30)
                track.cover = converted.returncode == 0 and cover_path.exists()
            session.add(track)
            try:
                session.flush()
            except IntegrityError as exc:
                raise HTTPException(409, "该音频已存在，请刷新音乐库。") from exc
        committed = True
        job = queue.enqueue(identity)
        return {"id": identity, "duplicate": False, "job_id": job}
    finally:
        temporary.unlink(missing_ok=True)
        if target and not committed:
            target.unlink(missing_ok=True)
            if cover_path:
                cover_path.unlink(missing_ok=True)
        file.file.close()


@app.get("/api/tracks")
def list_tracks(q: str = "", max_score: float = 100, artist: str = "", album: str = "",
                mode: Literal["general", "personal"] = "general", sort: Literal["recent", "score", "score_desc", "title"] = "recent"):
    with Session() as session:
        tracks = records(session)
    def number(track):
        return track["personal_score"] if mode == "personal" and track["personal_score"] is not None else track["score"]["combined"] if track["score"] else None
    tracks = [t for t in tracks if q.casefold() in f"{t['title']} {t['artist']} {t['album']}".casefold()
              and (not artist or t["artist"] == artist) and (not album or t["album"] == album)
              and ((number(t) is None and max_score >= 100) or (number(t) is not None and number(t) <= max_score))]
    if sort in {"score", "score_desc"}:
        tracks.sort(key=lambda t: (number(t) is None, (number(t) or 0) * (-1 if sort == "score_desc" else 1)))
    elif sort == "title":
        tracks.sort(key=lambda t: t["title"].casefold())
    # 音乐库只返回摘要，波形和时间序列由详情接口提供。
    return [{**t, "features": {k: v for k, v in t["features"].items() if k not in {"waveform", "short_term_loudness"}} if t["features"] else None,
             "score": {k: v for k, v in t["score"].items() if k != "segments"} if t["score"] else None} for t in tracks]


@app.get("/api/export.csv")
def export():
    with Session() as session:
        tracks = records(session)
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["歌曲", "艺术家", "专辑", "时长秒", "通用指数", "个人指数", "峰值片段指数", "评分模式", "主观评价"])
    def safe(value):
        text = str(value)
        return "'" + text if text.startswith(("=", "+", "-", "@")) else text
    for track in tracks:
        scored = track["score"] or {}
        writer.writerow([safe(track["title"]), safe(track["artist"]), safe(track["album"]), track["duration"],
                         scored.get("combined", ""), track["personal_score"] if track["personal_score"] is not None else "",
                         scored.get("peak", ""), scored.get("mode", ""), track["feedback"]["rating"] if track["feedback"] else ""])
    return Response("\ufeff" + output.getvalue(), media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": 'attachment; filename="quiettune.csv"'})


@app.get("/api/tracks/{identity}")
def detail(identity: str):
    with Session() as session:
        track_or_404(session, identity)
        return next(t for t in records(session) if t["id"] == identity)


@app.get("/api/tracks/{identity}/audio")
def audio(identity: str):
    with Session() as session:
        track = track_or_404(session, identity)
        path = DATA / "audio" / track.path
    if not path.is_file():
        raise HTTPException(410, "音频文件已丢失，请重新导入。")
    return FileResponse(path)


@app.get("/api/tracks/{identity}/cover")
def cover(identity: str):
    with Session() as session:
        track_or_404(session, identity)
    path = DATA / "covers" / f"{identity}.jpg"
    if not path.is_file():
        raise HTTPException(404, "没有内嵌封面。")
    return FileResponse(path, media_type="image/jpeg")


@app.delete("/api/tracks/{identity}")
def remove(identity: str):
    moved = []
    with queue.lock:
        try:
            with Session.begin() as session:
                track = track_or_404(session, identity)
                active = session.scalar(select(AnalysisJob).where(AnalysisJob.track_id == identity, AnalysisJob.status.in_(["queued", "running"])))
                if active:
                    raise HTTPException(409, "歌曲正在等待或执行分析，请完成后删除。")
                for path in (DATA / "audio" / track.path, DATA / "covers" / f"{identity}.jpg"):
                    if path.exists():
                        trash = path.with_suffix(path.suffix + ".trash")
                        path.replace(trash)
                        moved.append((path, trash))
                session.delete(track)
                invalidate_model(session, "歌曲与标注已删除，请重新训练。")
        except Exception:
            for path, trash in moved:
                trash.replace(path)
            raise
        for _, trash in moved:
            trash.unlink(missing_ok=True)
    return {"message": "歌曲、分析与反馈已删除。"}


@app.post("/api/tracks/{identity}/analyze")
def reanalyze(identity: str):
    with queue.lock, Session() as session:
        track_or_404(session, identity)
        return {"job_id": queue.enqueue(identity)}


class BatchInput(BaseModel):
    track_ids: list[str] = Field(min_length=1, max_length=500)


@app.post("/api/analyze")
def batch_analyze(payload: BatchInput):
    with queue.lock, Session() as session:
        for identity in payload.track_ids:
            track_or_404(session, identity)
        return {"job_ids": [queue.enqueue(identity) for identity in dict.fromkeys(payload.track_ids)]}


@app.get("/api/jobs")
def jobs():
    with Session() as session:
        return [job_data(j) for j in session.scalars(select(AnalysisJob).order_by(AnalysisJob.created_at.desc()).limit(300))]


FACTORS = {"音量过大", "高频刺耳", "鼓点密集", "编曲过于激烈", "人声过于尖锐", "动态变化突然", "其他"}


class FeedbackInput(BaseModel):
    rating: int = Field(ge=1, le=5)
    factors: list[str] = Field(default_factory=list, max_length=7)
    note: str = Field(default="", max_length=2000)

    @model_validator(mode="after")
    def check_factors(self):
        if any(factor not in FACTORS for factor in self.factors):
            raise ValueError("不喜欢的因素包含未知选项。")
        return self


@app.put("/api/tracks/{identity}/feedback")
def feedback(identity: str, payload: FeedbackInput):
    with Session.begin() as session:
        track_or_404(session, identity)
        session.merge(UserFeedback(track_id=identity, **payload.model_dump(), updated_at=now()))
        invalidate_model(session)
    return {"message": "评价已保存，重新训练后更新个人指数。"}


class SettingsInput(BaseModel):
    weights: dict[str, float]
    peak_mix: float = Field(ge=0, le=0.6)
    threshold: int = Field(ge=1, le=100)
    comfort_threshold: int = Field(ge=0, le=99)
    window_seconds: Literal[5, 10, 15, 30]
    concurrency: int = Field(ge=1, le=4)
    ai_enabled: bool

    @model_validator(mode="after")
    def check_weights(self):
        if set(self.weights) != set(LABELS) or any(not np.isfinite(v) or v < 0 or v > 1 for v in self.weights.values()):
            raise ValueError("权重必须包含全部六个维度，且在 0–1 之间。")
        if sum(v for k, v in self.weights.items() if k not in {"aggressive", "unrelaxed"}) <= 0:
            raise ValueError("至少需要一个非零声学维度权重，以支持分段与降级评分。")
        if self.comfort_threshold >= self.threshold:
            raise ValueError("舒适阈值必须低于高吵闹阈值。")
        return self


@app.get("/api/settings")
def get_settings():
    with Session() as session:
        return {**settings(session), "storage_path": str(DATA), "model_path": str(MODELS), "model_status": model_status(),
                "scoring_version": session.scalar(select(func.max(ScoringVersion.id)))}


@app.put("/api/settings")
def update_settings(payload: SettingsInput):
    config = payload.model_dump()
    with Session.begin() as session:
        previous = settings(session)
        session.get(PreferenceProfile, "settings").data = config
        session.add(ScoringVersion(config=config))
        invalidate_model(session)
    return {"message": "评分已使用缓存特征即时更新。", "requires_reanalysis": previous["window_seconds"] != config["window_seconds"] or previous["ai_enabled"] != config["ai_enabled"]}


@app.post("/api/settings/reset")
def reset_settings():
    return update_settings(SettingsInput(**DEFAULTS))


@app.get("/api/profile")
def profile():
    with Session() as session:
        data = session.get(PreferenceProfile, PERSONAL_PROFILE_ID).data
        labels = list(session.scalars(select(UserFeedback)))
    return {**data, "feedback_count": len(labels), "distribution": [sum(f.rating == r for f in labels) for r in range(1, 6)],
            "minimum_samples": 12, "name": "个人偏好"}


@app.post("/api/profile/train")
def train():
    with Session.begin() as session:
        tracks = [t for t in records(session) if t["feedback"] and t["score"]]
        ratings = [t["feedback"]["rating"] for t in tracks]
        if len(tracks) < 12 or len(set(ratings)) < 3:
            data = {"status": "insufficient", "samples": len(tracks), "reason": "至少需要 12 首已分析歌曲和 3 种不同评价，当前继续使用通用评分。"}
        else:
            x = np.array([personal_vector(t["score"]) for t in tracks])
            y = np.array([(rating - 1) * 25 for rating in ratings], dtype=float)
            pipeline = make_pipeline(StandardScaler(), RidgeCV(alphas=[1.0, 10.0, 100.0]))
            folds = KFold(n_splits=4, shuffle=True, random_state=42)
            predicted = cross_val_predict(pipeline, x, y, cv=folds)
            baseline = np.zeros_like(y)
            for train_ids, test_ids in folds.split(x):
                baseline[test_ids] = y[train_ids].mean()
            error, base_error = mean_absolute_error(y, predicted), mean_absolute_error(y, baseline)
            pipeline.fit(x, y)
            scaler, ridge = pipeline.steps[0][1], pipeline.steps[1][1]
            data = {"status": "ready" if error < base_error else "rejected", "samples": len(tracks), "cv_mae": round(error, 2),
                    "baseline_mae": round(base_error, 2), "validation": "四折交叉验证，折内标准化与正则化选择",
                    "reason": "模型验证优于均值基线，个人指数与通用指数混合以降低小样本风险。" if error < base_error else "验证未优于均值基线，继续使用通用评分，请补充多样评价。",
                    "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(), "coef": ridge.coef_.tolist(),
                    "intercept": float(ridge.intercept_), "alpha": float(ridge.alpha_), "trained_at": now(),
                    "feature_names": [*LABELS, "peak", "high_ratio"], "scoring_config": settings(session)}
        session.get(PreferenceProfile, PERSONAL_PROFILE_ID).data = data
    return data


@app.get("/api/stats")
def stats(mode: Literal["general", "personal"] = "general"):
    with Session() as session:
        tracks, config = records(session), settings(session)
    analyzed = [t for t in tracks if t["score"]]
    values = [t["personal_score"] if mode == "personal" and t["personal_score"] is not None else t["score"]["combined"] for t in analyzed]
    recommendations = sorted(analyzed, key=lambda t: t["personal_score"] if mode == "personal" and t["personal_score"] is not None else t["score"]["combined"])[:4]
    return {"total": len(tracks), "analyzed": len(analyzed), "average": round(float(np.mean(values)), 1) if values else None,
            "comfortable": sum(v < config["comfort_threshold"] for v in values), "high": sum(v >= config["threshold"] for v in values),
            "distribution": [{"name": f"{low}–{low + 19 if low < 80 else 100}", "count": sum(low <= v < low + 20 or low == 80 and v == 100 for v in values)} for low in range(0, 100, 20)],
            "recommendation_ids": [t["id"] for t in recommendations], "active_jobs": sum(t["status"] in {"queued", "running"} for t in tracks)}


frontend = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if frontend.exists():
    app.mount("/assets", StaticFiles(directory=frontend / "assets"), name="assets")

    @app.get("/{path:path}")
    def ui(path: str):
        if path.startswith("api/"):
            raise HTTPException(404, "接口不存在。")
        return FileResponse(frontend / "index.html")
