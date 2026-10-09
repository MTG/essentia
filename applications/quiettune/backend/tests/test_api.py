import io
import time

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from backend.app.config import DEFAULTS
from backend.app.db import PERSONAL_PROFILE_ID, AnalysisJob, AudioFeatures, Base, PreferenceProfile, SegmentFeatures, Session, Track, UserFeedback, engine, init_db
from backend.app.main import app


@pytest.fixture
def client():
    Base.metadata.drop_all(engine)
    with TestClient(app) as local:
        local.put("/api/settings", json={**DEFAULTS, "ai_enabled": False})
        yield local


def wav_bytes(frequency=440, seconds=6):
    output = io.BytesIO()
    t = np.arange(int(22050 * seconds)) / 22050
    sf.write(output, 0.1 * np.sin(2 * np.pi * frequency * t), 22050, format="WAV")
    return output.getvalue()


def wait_analysis(client, identity):
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        track = client.get(f"/api/tracks/{identity}").json()
        if track["status"] == "completed":
            return track
        assert track["status"] != "failed", track["job"]
        time.sleep(0.2)
    pytest.fail("后台分析未在限定时间内完成。")


def test_upload_analysis_range_feedback_rescore_delete(client):
    """验证实际音频解码、后台任务、存储、Range 与反馈修改完整路径。"""
    content = wav_bytes()
    uploaded = client.post("/api/tracks", files={"file": ("../../测试.wav", content, "audio/wav")})
    assert uploaded.status_code == 201
    identity = uploaded.json()["id"]
    duplicate = client.post("/api/tracks", files={"file": ("同一首.wav", content, "audio/wav")})
    assert duplicate.json()["duplicate"] is True
    track = wait_analysis(client, identity)
    assert track["features"]["lufs"] is not None
    assert track["score"]["segments"][-1]["end"] == pytest.approx(6)
    assert track["features"]["ai"]["aggressive"] is None
    playback = client.get(f"/api/tracks/{identity}/audio", headers={"Range": "bytes=0-99"})
    assert playback.status_code == 206 and len(playback.content) == 100
    invalid_range = client.get(f"/api/tracks/{identity}/audio", headers={"Range": "bytes=999999999-"})
    assert invalid_range.status_code == 416
    before = track["analyzed_at"]
    feedback = client.put(f"/api/tracks/{identity}/feedback", json={"rating": 2, "factors": ["高频刺耳"], "note": "测试评价"})
    assert feedback.status_code == 200
    assert feedback.json()["message"] == "评价已保存，重新训练后更新个人指数。"
    assert client.put(f"/api/tracks/{identity}/feedback", json={"rating": 4, "factors": []}).status_code == 200
    config = client.get("/api/settings").json()
    config["weights"]["loudness"] = 0.8
    assert client.put("/api/settings", json=config).status_code == 200
    updated = client.get(f"/api/tracks/{identity}").json()
    assert updated["analyzed_at"] == before
    assert updated["feedback"]["rating"] == 4
    assert updated["score"]["combined"] != track["score"]["combined"]
    assert client.post("/api/profile/train").json()["status"] == "insufficient"
    exported = client.get("/api/export.csv").text
    assert "测试" in exported and "个人指数" in exported.splitlines()[0]
    assert client.get("/api/profile").json()["name"] == "个人偏好"
    assert client.get("/api/stats").json()["analyzed"] == 1
    assert client.get("/api/tracks?q=不存在").json() == []
    assert client.delete(f"/api/tracks/{identity}").status_code == 200
    assert client.get(f"/api/tracks/{identity}").status_code == 404
    with Session() as session:
        assert session.get(AudioFeatures, identity) is None
        assert session.get(UserFeedback, identity) is None


def test_invalid_upload_and_settings(client):
    assert client.post("/api/tracks", files={"file": ("x.txt", b"abc")}).status_code == 415
    assert client.post("/api/tracks", files={"file": ("x.wav", b"")}).status_code == 400
    assert client.post("/api/tracks", files={"file": ("x.wav", b"broken")}).status_code == 422
    config = {**DEFAULTS, "weights": {key: 0 for key in DEFAULTS["weights"]}}
    assert client.put("/api/settings", json=config).status_code == 422
    assert client.get("/api/tracks/unknown").status_code == 404
    assert client.post("/api/analyze", json={"track_ids": ["unknown"]}).status_code == 404


def test_personal_training_uses_cached_features(client):
    """固定特征夹具只验证训练、持久化与排序，无须解码或伪造模型预测。"""
    with Session.begin() as session:
        for index in range(20):
            group = index % 5
            features = {"silent": False, "rms_dbfs": -40 + group * 7, "centroid_hz": 300 + group * 1500,
                        "high_frequency_ratio": group / 5, "onset_density": group * 1.5, "onset_strength": group * 0.05,
                        "flatness": group * 0.12, "ai": {"status": "disabled", "aggressive": None, "relaxed": None}}
            identity = f"fixture-{index}"
            session.add(Track(id=identity, sha256=identity, filename="测试夹具.wav", title=identity, path="测试中不读取音频", duration=5))
            session.flush()
            session.add(AudioFeatures(track_id=identity, data=features, analyzed_at="固定分析时间"))
            session.add(SegmentFeatures(track_id=identity, start=0, end=5, data={**features, "start": 0, "end": 5}))
            session.add(UserFeedback(track_id=identity, rating=group + 1, factors=[]))
    trained = client.post("/api/profile/train").json()
    assert trained["status"] == "ready", trained
    assert trained["cv_mae"] < trained["baseline_mae"]
    ordered = client.get("/api/tracks?mode=personal&sort=score").json()
    values = [t["personal_score"] for t in ordered]
    assert all(value is not None for value in values)
    assert values == sorted(values)
    assert all(t["analyzed_at"] == "固定分析时间" for t in ordered)
    assert client.get("/api/jobs").json() == []
    client.put("/api/tracks/fixture-0/feedback", json={"rating": 4, "factors": []})
    assert client.get("/api/profile").json()["status"] == "stale"


def test_background_failure_and_retry(client, monkeypatch):
    """后台错误需要持久化，恢复依赖后可重新排队。"""
    from backend.app import jobs
    original = jobs.analyze
    def fail(*_):
        raise RuntimeError("测试模拟：解码器暂不可用")
    monkeypatch.setattr(jobs, "analyze", fail)
    identity = client.post("/api/tracks", files={"file": ("恢复测试.wav", wav_bytes(220))}).json()["id"]
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        track = client.get(f"/api/tracks/{identity}").json()
        if track["status"] == "failed":
            break
        time.sleep(0.1)
    assert track["status"] == "failed"
    assert "解码器" in track["job"]["error"]
    monkeypatch.setattr(jobs, "analyze", original)
    assert client.post(f"/api/tracks/{identity}/analyze").status_code == 200
    assert wait_analysis(client, identity)["status"] == "completed"


def test_legacy_profile_migration_preserves_model_and_feedback(client):
    """历史偏好仅迁移主键，模型与评价原文保留，重复初始化没有副作用。"""
    model = {"status": "ready", "samples": 20, "coef": [1.2, 3.4], "intercept": 5.6,
             "mean": [7.8, 9.0], "scale": [1.0, 2.0], "cv_mae": 6.5}
    with Session.begin() as session:
        profile = session.get(PreferenceProfile, PERSONAL_PROFILE_ID)
        profile.id, profile.data = "xiaoyi", model
        session.add(Track(id="migration-track", sha256="migration-track", filename="迁移.wav",
                          title="迁移验证", path="不读取音频", duration=5))
        session.flush()
        session.add(UserFeedback(track_id="migration-track", rating=2, factors=["高频刺耳"], note="保留用户原文"))
    init_db()
    init_db()
    with Session() as session:
        assert session.get(PreferenceProfile, PERSONAL_PROFILE_ID).data == model
        assert session.get(PreferenceProfile, "xiaoyi") is None
        feedback = session.get(UserFeedback, "migration-track")
        assert (feedback.rating, feedback.factors, feedback.note) == (2, ["高频刺耳"], "保留用户原文")
    profile = client.get("/api/profile").json()
    assert profile["name"] == "个人偏好" and profile["status"] == "ready" and profile["feedback_count"] == 1


def test_existing_personal_profile_is_not_overwritten(client):
    """通用偏好已存在时不以历史行覆盖它，也不删除历史数据。"""
    current = {"status": "stale", "samples": 15, "reason": "保留当前模型"}
    legacy = {"status": "ready", "samples": 12}
    with Session.begin() as session:
        session.get(PreferenceProfile, PERSONAL_PROFILE_ID).data = current
        session.add(PreferenceProfile(id="xiaoyi", data=legacy))
    init_db()
    with Session() as session:
        assert session.get(PreferenceProfile, PERSONAL_PROFILE_ID).data == current
        assert session.get(PreferenceProfile, "xiaoyi").data == legacy


def test_legacy_system_prompt_changes_without_rewriting_user_note(client):
    """迁移固定提示，歌曲信息和用户备注中的姓名不作批量替换。"""
    with Session.begin() as session:
        profile = session.get(PreferenceProfile, PERSONAL_PROFILE_ID)
        profile.id = "xiaoyi"
        profile.data = {"status": "insufficient", "samples": 0,
                        "reason": "开始记录小怡的真实评价后，再尝试训练个人模型。"}
        session.add(Track(id="original-note", sha256="original-note", filename="用户音乐.wav",
                          title="小怡推荐的音乐", path="不读取音频", duration=5))
        session.flush()
        session.add(UserFeedback(track_id="original-note", rating=2, factors=[], note="小怡推荐，保留原文"))
    init_db()
    init_db()
    profile = client.get("/api/profile").json()
    assert profile["reason"] == "开始记录真实试听评价后，再尝试训练个人模型。"
    with Session() as session:
        assert session.get(Track, "original-note").title == "小怡推荐的音乐"
        assert session.get(UserFeedback, "original-note").note == "小怡推荐，保留原文"
