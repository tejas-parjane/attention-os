from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    create_engine,
    event,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    sessionmaker,
)
from sqlalchemy.types import JSON


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    user_id: Mapped[str] = mapped_column(String, primary_key=True)
    archetype: Mapped[str] = mapped_column(String)
    signup_date: Mapped[datetime] = mapped_column(DateTime)
    country: Mapped[str] = mapped_column(String)
    device_type: Mapped[str] = mapped_column(String)
    acquisition_channel: Mapped[str] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Event(Base):
    __tablename__ = "events"
    __table_args__ = (
        Index("ix_events_user_id", "user_id"),
        Index("ix_events_timestamp", "timestamp"),
        Index("ix_events_user_id_timestamp", "user_id", "timestamp"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String, ForeignKey("users.user_id"))
    session_id: Mapped[str] = mapped_column(String)
    timestamp: Mapped[datetime] = mapped_column(DateTime)
    event_type: Mapped[str] = mapped_column(String)
    session_duration: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    screens_viewed: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    actions_completed: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    level: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    purchase_amount: Mapped[float] = mapped_column(Float, default=0.0)
    ad_impressions: Mapped[int] = mapped_column(Integer, default=0)
    ad_revenue: Mapped[float] = mapped_column(Float, default=0.0)
    notification_opened: Mapped[bool] = mapped_column(Boolean, default=False)
    notification_clicked: Mapped[bool] = mapped_column(Boolean, default=False)
    content_category: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    device_type: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    country: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    acquisition_channel: Mapped[Optional[str]] = mapped_column(String, nullable=True)


class UserFeature(Base):
    __tablename__ = "user_features"
    __table_args__ = (Index("ix_user_features_user_id", "user_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String, ForeignKey("users.user_id"))
    feature_name: Mapped[str] = mapped_column(String)
    feature_value: Mapped[float] = mapped_column(Float)
    window: Mapped[str] = mapped_column(String)
    computed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (Index("ix_predictions_user_id", "user_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String, ForeignKey("users.user_id"))
    model_name: Mapped[str] = mapped_column(String)
    prediction_type: Mapped[str] = mapped_column(String)
    prediction_value: Mapped[float] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float)
    predicted_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Recommendation(Base):
    __tablename__ = "recommendations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String, ForeignKey("users.user_id"))
    recommended_action: Mapped[str] = mapped_column(String)
    expected_value: Mapped[float] = mapped_column(Float)
    explanation: Mapped[str] = mapped_column(Text)
    guardrails_applied: Mapped[dict] = mapped_column(JSON)
    recommended_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_name: Mapped[str] = mapped_column(String)
    description: Mapped[str] = mapped_column(Text)
    variants: Mapped[dict] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class ExperimentAssignment(Base):
    __tablename__ = "experiment_assignments"
    __table_args__ = (
        Index("ix_experiment_assignments_user_id_experiment_id", "user_id", "experiment_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[int] = mapped_column(Integer, ForeignKey("experiments.id"))
    user_id: Mapped[str] = mapped_column(String, ForeignKey("users.user_id"))
    variant: Mapped[str] = mapped_column(String)
    assigned_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Outcome(Base):
    __tablename__ = "outcomes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("experiments.id"), nullable=True
    )
    user_id: Mapped[str] = mapped_column(String, ForeignKey("users.user_id"))
    action_taken: Mapped[str] = mapped_column(String)
    metric_name: Mapped[str] = mapped_column(String)
    metric_value: Mapped[float] = mapped_column(Float)
    measured_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


def get_database_url() -> str:
    try:
        from src.config import get_settings

        s = get_settings()
        # Try the primary URL; if it fails at connect time, use the fallback.
        return s.database_url or s.database_url_fallback
    except (ImportError, AttributeError):
        return "sqlite:///attention_os.db"


def get_engine(url: str = None) -> "Engine":
    if url is None:
        url = get_database_url()
    return create_engine(url, echo=False)


def get_session(engine) -> Session:
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()


def get_session_ctx(engine):
    """Context-manager yielding a :class:`Session` from *engine*."""
    from sqlalchemy.orm import Session as _Session
    session = get_session(engine)
    try:
        yield session
    finally:
        session.close()


def init_db(engine) -> None:
    Base.metadata.create_all(engine)
