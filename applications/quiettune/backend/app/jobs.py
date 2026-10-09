import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from sqlalchemy import delete, select

from .audio import analyze
from .config import DATA
from .db import PERSONAL_PROFILE_ID, AnalysisJob, AudioFeatures, PreferenceProfile, SegmentFeatures, Session, Track, now, settings


class JobQueue:
    def __init__(self):
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.thread = None

    def start(self):
        self.stop_event.clear()
        with Session.begin() as session:
            for job in session.scalars(select(AnalysisJob).where(AnalysisJob.status == "running")):
                job.status, job.progress, job.stage = "queued", 0, "重启后恢复等待"
        self.thread = threading.Thread(target=self.dispatch, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join()

    def enqueue(self, track_id: str) -> str:
        with self.lock, Session.begin() as session:
            existing = session.scalar(select(AnalysisJob).where(AnalysisJob.track_id == track_id, AnalysisJob.status.in_(["queued", "running"])))
            if existing:
                return existing.id
            identity = uuid.uuid4().hex
            session.add(AnalysisJob(id=identity, track_id=track_id, config=settings(session)))
            return identity

    def dispatch(self):
        with ThreadPoolExecutor(max_workers=4, thread_name_prefix="quiettune") as pool:
            active = {}
            while not self.stop_event.wait(0.15):
                active = {key: future for key, future in active.items() if not future.done()}
                with self.lock, Session.begin() as session:
                    slots = max(0, settings(session)["concurrency"] - len(active))
                    jobs = session.scalars(select(AnalysisJob).where(AnalysisJob.status == "queued").order_by(AnalysisJob.created_at).limit(slots)).all()
                    for job in jobs:
                        job.status, job.stage = "running", "准备分析"
                    identities = [job.id for job in jobs]
                for identity in identities:
                    active[identity] = pool.submit(self.run, identity)

    def run(self, identity: str):
        def progress(value: int, stage: str):
            with Session.begin() as session:
                job = session.get(AnalysisJob, identity)
                job.progress, job.stage = value, stage

        try:
            with Session() as session:
                job = session.get(AnalysisJob, identity)
                track = session.get(Track, job.track_id)
                track_id, path, config = track.id, DATA / "audio" / track.path, job.config
            features, segments = analyze(path, config, progress)
            with Session.begin() as session:
                session.merge(AudioFeatures(track_id=track_id, data=features, analyzed_at=now()))
                session.execute(delete(SegmentFeatures).where(SegmentFeatures.track_id == track_id))
                session.add_all([SegmentFeatures(track_id=track_id, start=s["start"], end=s["end"], data=s) for s in segments])
                job = session.get(AnalysisJob, identity)
                job.status, job.progress, job.stage, job.finished_at = "completed", 100, "分析完成", now()
                profile = session.get(PreferenceProfile, PERSONAL_PROFILE_ID)
                if profile.data.get("status") == "ready":
                    profile.data = {**profile.data, "status": "stale", "reason": "声学特征已更新，请重新训练个人模型。"}
        except Exception as exc:
            with Session.begin() as session:
                job = session.get(AnalysisJob, identity)
                if job:
                    job.status, job.stage, job.error, job.finished_at = "failed", "分析失败，可重试", str(exc), now()


queue = JobQueue()
