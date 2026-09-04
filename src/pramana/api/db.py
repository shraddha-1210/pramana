"""Persistence. SQLite via SQLAlchemy -- not Postgres.

Fewer moving parts matters more than scalability here: this is an assurance
testbed, not a service. Every run is persisted **with its seed**, so any number
shown in the dashboard is reproducible from the dashboard.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import JSON, DateTime, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

DEFAULT_DB_PATH = Path("pramana.sqlite")


class Base(DeclarativeBase):
    """Declarative base."""


class Run(Base):
    """One assurance run.

    The seed is not optional and not nullable: a stored result whose seed was lost
    is not reproducible, which defeats the purpose of storing it.
    """

    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    scheme: Mapped[str] = mapped_column(String(128), index=True)
    seed: Mapped[int] = mapped_column()
    rounds: Mapped[int] = mapped_column()
    status: Mapped[str] = mapped_column(String(32), default="pending")
    stage: Mapped[str] = mapped_column(String(64), default="queued")
    completed_rounds: Mapped[int] = mapped_column(default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    report: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(String(1024), nullable=True)

    def to_dict(self) -> dict[str, object]:
        """Serialise, always including the seed."""
        return {
            "id": self.id,
            "scheme": self.scheme,
            "seed": self.seed,
            "rounds": self.rounds,
            "status": self.status,
            "stage": self.stage,
            "completed_rounds": self.completed_rounds,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "report": self.report,
            "error": self.error,
        }


class UploadedSpec(Base):
    """A specification uploaded through the API."""

    __tablename__ = "specs"

    name: Mapped[str] = mapped_column(String(128), primary_key=True)
    source: Mapped[str] = mapped_column(String(65536))
    summary: Mapped[dict] = mapped_column(JSON)


_engine = None
_SessionLocal: sessionmaker[Session] | None = None


def init_db(path: Path | str | None = None, *, echo: bool = False) -> None:
    """Create (or recreate) the engine and schema."""
    global _engine, _SessionLocal
    url = "sqlite://" if path is None else f"sqlite:///{path}"
    _engine = create_engine(url, echo=echo, connect_args={"check_same_thread": False})
    _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)
    Base.metadata.create_all(_engine)


def get_session() -> Iterator[Session]:
    """FastAPI dependency yielding a session.

    Raises:
        RuntimeError: ``init_db`` was never called.
    """
    if _SessionLocal is None:
        raise RuntimeError("database not initialised; call init_db() first")
    session = _SessionLocal()
    try:
        yield session
    finally:
        session.close()


def json_safe(value: object) -> object:
    """Coerce a value into something JSON columns accept."""
    return json.loads(json.dumps(value, default=str))
