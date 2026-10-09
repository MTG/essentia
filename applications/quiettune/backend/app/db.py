from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, Float, ForeignKey, Integer, String, Text, create_engine, event
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from .config import DATA, DEFAULTS

PERSONAL_PROFILE_ID = "personal"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Base(DeclarativeBase):
    pass


class Track(Base):
    __tablename__ = "tracks"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    sha256: Mapped[str] = mapped_column(String, unique=True)
    filename: Mapped[str] = mapped_column(String)
    title: Mapped[str] = mapped_column(String)
    artist: Mapped[str] = mapped_column(String, default="未知艺术家")
    album: Mapped[str] = mapped_column(String, default="未标注专辑")
    relative_path: Mapped[str] = mapped_column(String, default="")
    path: Mapped[str] = mapped_column(String)
    duration: Mapped[float] = mapped_column(Float)
    cover: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[str] = mapped_column(String, default=now)


class AudioFeatures(Base):
    __tablename__ = "audio_features"
    track_id: Mapped[str] = mapped_column(ForeignKey("tracks.id", ondelete="CASCADE"), primary_key=True)
    data: Mapped[dict] = mapped_column(JSON)
    analyzed_at: Mapped[str] = mapped_column(String, default=now)


class SegmentFeatures(Base):
    __tablename__ = "segment_features"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    track_id: Mapped[str] = mapped_column(ForeignKey("tracks.id", ondelete="CASCADE"), index=True)
    start: Mapped[float] = mapped_column(Float)
    end: Mapped[float] = mapped_column(Float)
    data: Mapped[dict] = mapped_column(JSON)


class AnalysisJob(Base):
    __tablename__ = "analysis_jobs"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    track_id: Mapped[str] = mapped_column(ForeignKey("tracks.id", ondelete="CASCADE"), index=True)
    status: Mapped[str] = mapped_column(String, default="queued")
    progress: Mapped[int] = mapped_column(Integer, default=0)
    stage: Mapped[str] = mapped_column(String, default="等待分析")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    config: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[str] = mapped_column(String, default=now)
    finished_at: Mapped[str | None] = mapped_column(String, nullable=True)


class UserFeedback(Base):
    __tablename__ = "user_feedback"
    track_id: Mapped[str] = mapped_column(ForeignKey("tracks.id", ondelete="CASCADE"), primary_key=True)
    rating: Mapped[int] = mapped_column(Integer)
    factors: Mapped[list] = mapped_column(JSON)
    note: Mapped[str] = mapped_column(Text, default="")
    updated_at: Mapped[str] = mapped_column(String, default=now)


class PreferenceProfile(Base):
    __tablename__ = "preference_profiles"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    data: Mapped[dict] = mapped_column(JSON)


class ScoringVersion(Base):
    __tablename__ = "scoring_versions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    config: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[str] = mapped_column(String, default=now)


engine = create_engine(f"sqlite:///{DATA / 'quiettune.db'}", connect_args={"check_same_thread": False, "timeout": 30})


@event.listens_for(engine, "connect")
def configure_sqlite(connection, _):
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA journal_mode=WAL")


Session = sessionmaker(engine, expire_on_commit=False)


def init_db():
    Base.metadata.create_all(engine)
    with Session.begin() as session:
        if not session.get(PreferenceProfile, "settings"):
            session.add(PreferenceProfile(id="settings", data=DEFAULTS))
            session.add(ScoringVersion(config=DEFAULTS))
        profile = session.get(PreferenceProfile, PERSONAL_PROFILE_ID)
        if not profile:
            legacy = session.get(PreferenceProfile, "xiaoyi")
            if legacy:
                # 只迁移历史主键，保留已训练模型与评价状态，后续统一使用通用标识。
                legacy.id = PERSONAL_PROFILE_ID
                profile = legacy
            else:
                profile = PreferenceProfile(id=PERSONAL_PROFILE_ID, data={"status": "insufficient", "samples": 0})
                session.add(profile)
        # 仅更新旧版应用生成的固定提示，用户自由输入的评价与备注不参与迁移。
        if profile.data.get("reason") == "开始记录小怡的真实评价后，再尝试训练个人模型。":
            profile.data = {**profile.data, "reason": "开始记录真实试听评价后，再尝试训练个人模型。"}


def settings(session=None) -> dict:
    if session is not None:
        return session.get(PreferenceProfile, "settings").data
    with Session() as local:
        return local.get(PreferenceProfile, "settings").data
