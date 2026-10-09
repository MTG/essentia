from sqlalchemy import event

from backend.app.db import AnalysisJob, AudioFeatures, Session, Track, engine
from backend.tests.test_api import client


def test_completed_state_has_committed_features(client):
    """强制任务在读取过程中完成，完成状态必须携带已提交的特征。"""
    identity = "snapshot-fixture"
    with Session.begin() as session:
        session.add(Track(id=identity, sha256=identity, filename="并发读取夹具.wav", title="并发读取测试",
                          path="本测试不读取音频", duration=5))
        session.flush()
        session.add(AnalysisJob(id="snapshot-job", track_id=identity, status="running", config={}))
    committed = False

    def complete_during_read(_connection, _cursor, statement, _parameters, _context, _many):
        nonlocal committed
        if not committed and "FROM audio_features" in statement:
            committed = True
            with Session.begin() as session:
                session.add(AudioFeatures(track_id=identity, data={"silent": True}, analyzed_at="本测试固定的提交时间"))
                session.get(AnalysisJob, "snapshot-job").status = "completed"

    event.listen(engine, "after_cursor_execute", complete_during_read)
    try:
        first = client.get(f"/api/tracks/{identity}").json()
        assert committed
        if first["status"] == "completed":
            assert first["features"] is not None and first["analyzed_at"] is not None
        final = client.get(f"/api/tracks/{identity}").json()
        assert final["status"] == "completed"
        assert final["features"]["silent"] is True
        assert final["analyzed_at"] == "本测试固定的提交时间"
    finally:
        event.remove(engine, "after_cursor_execute", complete_during_read)
