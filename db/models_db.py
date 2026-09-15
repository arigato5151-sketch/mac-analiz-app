"""SQLAlchemy ORM models for the football analysis database."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


class League(Base):
    """League/competition information."""

    __tablename__ = "leagues"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    country: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    season: Mapped[int] = mapped_column(Integer, nullable=False)
    logo_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), onupdate=func.now())

    teams: Mapped[list["Team"]] = relationship(back_populates="league")
    matches: Mapped[list["Match"]] = relationship(back_populates="league")

    def __repr__(self) -> str:
        return f"<League(id={self.id}, name='{self.name}', season={self.season})>"


class Team(Base):
    """Team/club information."""

    __tablename__ = "teams"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    short_name: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    league_id: Mapped[int] = mapped_column(Integer, ForeignKey("leagues.id"), nullable=False)
    country: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    founded_year: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    stadium: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    logo_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), onupdate=func.now())

    league: Mapped["League"] = relationship(back_populates="teams")
    home_matches: Mapped[list["Match"]] = relationship(foreign_keys="Match.home_team_id", back_populates="home_team")
    away_matches: Mapped[list["Match"]] = relationship(foreign_keys="Match.away_team_id", back_populates="away_team")
    form_entries: Mapped[list["TeamForm"]] = relationship(back_populates="team")
    availabilities: Mapped[list["PlayerAvailability"]] = relationship(back_populates="team")
    lineups: Mapped[list["FixtureLineup"]] = relationship(back_populates="team")

    __table_args__ = (
        Index("ix_teams_league_id", "league_id"),
        Index("ix_teams_name", "name"),
    )

    def __repr__(self) -> str:
        return f"<Team(id={self.id}, name='{self.name}')>"


class Match(Base):
    """Match/fixture information."""

    __tablename__ = "matches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    league_id: Mapped[int] = mapped_column(Integer, ForeignKey("leagues.id"), nullable=False)
    home_team_id: Mapped[int] = mapped_column(Integer, ForeignKey("teams.id"), nullable=False)
    away_team_id: Mapped[int] = mapped_column(Integer, ForeignKey("teams.id"), nullable=False)
    match_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="scheduled")
    home_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    away_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    home_xg: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    away_xg: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    referee: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    venue: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), onupdate=func.now())

    league: Mapped["League"] = relationship(back_populates="matches")
    home_team: Mapped["Team"] = relationship(foreign_keys=[home_team_id], back_populates="home_matches")
    away_team: Mapped["Team"] = relationship(foreign_keys=[away_team_id], back_populates="away_matches")
    predictions: Mapped[list["Prediction"]] = relationship(back_populates="match")
    odds_history: Mapped[list["OddsQuoteHistory"]] = relationship(back_populates="match")
    lineups: Mapped[list["FixtureLineup"]] = relationship(back_populates="match")
    snapshots: Mapped[list["PredictionSnapshot"]] = relationship(back_populates="match")
    commentary: Mapped[Optional["MatchCommentary"]] = relationship(back_populates="match", uselist=False)
    player_availability: Mapped[list["PlayerAvailability"]] = relationship(back_populates="match")

    __table_args__ = (
        Index("ix_matches_league_date", "league_id", "match_date"),
        Index("ix_matches_status_date", "status", "match_date"),
        Index("ix_matches_teams", "home_team_id", "away_team_id"),
    )

    def __repr__(self) -> str:
        return f"<Match(id={self.id}, home={self.home_team_id} vs away={self.away_team_id})>"


class Prediction(Base):
    """Model prediction for a match."""

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    match_id: Mapped[int] = mapped_column(Integer, ForeignKey("matches.id"), nullable=False)
    model_version: Mapped[str] = mapped_column(String(100), nullable=False)
    prob_home_win: Mapped[float] = mapped_column(Float, nullable=False)
    prob_draw: Mapped[float] = mapped_column(Float, nullable=False)
    prob_away_win: Mapped[float] = mapped_column(Float, nullable=False)
    prob_over_2_5: Mapped[float] = mapped_column(Float, nullable=False)
    prob_btts: Mapped[float] = mapped_column(Float, nullable=False)
    market_probabilities: Mapped[Optional[dict]] = mapped_column(Text, nullable=True)  # JSON
    predicted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    match: Mapped["Match"] = relationship(back_populates="predictions")
    snapshots: Mapped[list["PredictionSnapshot"]] = relationship(back_populates="prediction")
    performance: Mapped[Optional["PredictionPerformance"]] = relationship(back_populates="prediction", uselist=False)

    __table_args__ = (
        Index("ix_predictions_match_id", "match_id"),
        Index("ix_predictions_model_version", "model_version"),
        UniqueConstraint("match_id", "model_version", name="uq_prediction_match_model"),
    )

    def __repr__(self) -> str:
        return f"<Prediction(id={self.id}, match={self.match_id}, model={self.model_version})>"


class PredictionSnapshot(Base):
    """Immutable snapshot of a prediction at a specific point in time."""

    __tablename__ = "prediction_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_prediction_id: Mapped[int] = mapped_column(Integer, ForeignKey("predictions.id"), nullable=False)
    match_id: Mapped[int] = mapped_column(Integer, ForeignKey("matches.id"), nullable=False)
    snapshot_type: Mapped[str] = mapped_column(String(50), nullable=False)
    model_version: Mapped[str] = mapped_column(String(100), nullable=False)
    prob_home_win: Mapped[float] = mapped_column(Float, nullable=False)
    prob_draw: Mapped[float] = mapped_column(Float, nullable=False)
    prob_away_win: Mapped[float] = mapped_column(Float, nullable=False)
    prob_over_2_5: Mapped[float] = mapped_column(Float, nullable=False)
    prob_btts: Mapped[float] = mapped_column(Float, nullable=False)
    market_probabilities: Mapped[Optional[dict]] = mapped_column(Text, nullable=True)
    source_predicted_at: Mapped[str] = mapped_column(String(50), nullable=False)
    context: Mapped[Optional[dict]] = mapped_column(Text, nullable=True)
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    prediction: Mapped["Prediction"] = relationship(back_populates="snapshots")
    match: Mapped["Match"] = relationship(back_populates="snapshots")

    __table_args__ = (
        Index("ix_prediction_snapshots_match_type", "match_id", "snapshot_type"),
        UniqueConstraint("match_id", "snapshot_type", name="uq_snapshot_match_type"),
    )


class PredictionPerformance(Base):
    """Evaluated prediction performance metrics."""

    __tablename__ = "prediction_performance"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    prediction_id: Mapped[int] = mapped_column(Integer, ForeignKey("predictions.id"), nullable=False)
    match_id: Mapped[int] = mapped_column(Integer, ForeignKey("matches.id"), nullable=False)
    was_correct: Mapped[bool] = mapped_column(Boolean, nullable=False)
    over_2_5_was_correct: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    btts_was_correct: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    brier_score: Mapped[float] = mapped_column(Float, nullable=False)
    log_loss: Mapped[float] = mapped_column(Float, nullable=False)
    evaluated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    prediction: Mapped["Prediction"] = relationship(back_populates="performance")
    match: Mapped["Match"] = relationship()

    __table_args__ = (
        Index("ix_prediction_performance_prediction_id", "prediction_id"),
        Index("ix_prediction_performance_match_id", "match_id"),
        UniqueConstraint("prediction_id", name="uq_performance_prediction"),
    )


class OddsQuoteHistory(Base):
    """Historical odds quotes for a match."""

    __tablename__ = "odds_quote_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    match_id: Mapped[int] = mapped_column(Integer, ForeignKey("matches.id"), nullable=False)
    bookmaker: Mapped[str] = mapped_column(String(100), nullable=False)
    bookmaker_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    odds: Mapped[dict] = mapped_column(Text, nullable=False)  # JSON
    source_updated_at: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    is_notification_reference: Mapped[bool] = mapped_column(Boolean, default=False)

    match: Mapped["Match"] = relationship(back_populates="odds_history")

    __table_args__ = (
        Index("ix_odds_quote_history_match_captured", "match_id", "captured_at"),
        Index("ix_odds_quote_history_match_bookmaker", "match_id", "bookmaker"),
    )


class TeamForm(Base):
    """Team form/strength metrics."""

    __tablename__ = "team_form"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    team_id: Mapped[int] = mapped_column(Integer, ForeignKey("teams.id"), nullable=False)
    calculated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    elo_rating: Mapped[float] = mapped_column(Float, nullable=False)
    avg_goals_scored_last5: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    avg_goals_conceded_last5: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    win_rate_last5: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    home_away_split: Mapped[Optional[dict]] = mapped_column(Text, nullable=True)  # JSON

    team: Mapped["Team"] = relationship(back_populates="form_entries")

    __table_args__ = (
        Index("ix_team_form_team_calculated", "team_id", "calculated_at"),
        UniqueConstraint("team_id", "calculated_at", name="uq_team_form_team_calculated"),
    )


class PlayerAvailability(Base):
    """Player availability (injuries, suspensions, doubts)."""

    __tablename__ = "player_availability"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    match_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("matches.id"), nullable=True)
    team_id: Mapped[int] = mapped_column(Integer, ForeignKey("teams.id"), nullable=False)
    player_name: Mapped[str] = mapped_column(String(200), nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False)  # injured, suspended, doubtful
    source: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    refreshed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    team: Mapped["Team"] = relationship(back_populates="availabilities")
    match: Mapped[Optional["Match"]] = relationship(back_populates="player_availability")

    __table_args__ = (
        Index("ix_player_availability_team_refreshed", "team_id", "refreshed_at"),
        Index("ix_player_availability_match", "match_id"),
    )


class FixtureLineup(Base):
    """Confirmed starting lineups for a fixture."""

    __tablename__ = "fixture_lineups"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    match_id: Mapped[int] = mapped_column(Integer, ForeignKey("matches.id"), nullable=False)
    team_id: Mapped[int] = mapped_column(Integer, ForeignKey("teams.id"), nullable=False)
    formation: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    confirmed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    coach_name: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)

    match: Mapped["Match"] = relationship(back_populates="lineups")
    team: Mapped["Team"] = relationship(back_populates="lineups")

    __table_args__ = (
        Index("ix_fixture_lineups_match", "match_id"),
        Index("ix_fixture_lineups_team", "team_id"),
        UniqueConstraint("match_id", "team_id", name="uq_lineup_match_team"),
    )


class MatchCommentary(Base):
    """AI-generated match commentary."""

    __tablename__ = "match_commentary"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    match_id: Mapped[int] = mapped_column(Integer, ForeignKey("matches.id"), nullable=False)
    commentary_text: Mapped[str] = mapped_column(Text, nullable=False)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    match: Mapped["Match"] = relationship(back_populates="commentary")

    __table_args__ = (
        UniqueConstraint("match_id", name="uq_commentary_match"),
    )


class NotificationLog(Base):
    """Log of sent notifications for idempotency."""

    __tablename__ = "notification_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    match_id: Mapped[int] = mapped_column(Integer, ForeignKey("matches.id"), nullable=False)
    notification_type: Mapped[str] = mapped_column(String(50), nullable=False)
    model_version: Mapped[str] = mapped_column(String(100), nullable=False)
    sent_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        Index("ix_notification_log_match_type", "match_id", "notification_type"),
        UniqueConstraint("match_id", "notification_type", name="uq_notification_log"),
    )


class PreMatchTelegramQueue(Base):
    """Queue for pre-match Telegram notifications."""

    __tablename__ = "pre_match_telegram_queue"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    match_id: Mapped[int] = mapped_column(Integer, ForeignKey("matches.id"), nullable=False)
    message_text: Mapped[str] = mapped_column(Text, nullable=False)
    delivered_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_pre_match_queue_delivered", "delivered_at"),
        UniqueConstraint("match_id", name="uq_pre_match_queue_match"),
    )


class ModelArtifact(Base):
    """Track model training artifacts and versions."""

    __tablename__ = "model_artifacts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_version: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    model_type: Mapped[str] = mapped_column(String(50), nullable=False)  # xgboost, poisson, etc.
    training_rows: Mapped[int] = mapped_column(Integer, nullable=False)
    metrics: Mapped[dict] = mapped_column(Text, nullable=False)  # JSON
    hyperparameters: Mapped[dict] = mapped_column(Text, nullable=True)  # JSON
    training_end: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    status: Mapped[str] = mapped_column(String(50), default="trained")  # trained, promoted, rejected
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ShadowPrediction(Base):
    """Shadow model predictions for A/B testing."""

    __tablename__ = "shadow_predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_version: Mapped[str] = mapped_column(String(100), nullable=False)
    match_id: Mapped[int] = mapped_column(Integer, ForeignKey("matches.id"), nullable=False)
    prob_home_win: Mapped[float] = mapped_column(Float, nullable=False)
    prob_draw: Mapped[float] = mapped_column(Float, nullable=False)
    prob_away_win: Mapped[float] = mapped_column(Float, nullable=False)
    prob_over_2_5: Mapped[float] = mapped_column(Float, nullable=False)
    prob_btts: Mapped[float] = mapped_column(Float, nullable=False)
    market_probabilities: Mapped[Optional[dict]] = mapped_column(Text, nullable=True)
    predicted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        Index("ix_shadow_predictions_model_match", "model_version", "match_id"),
        UniqueConstraint("model_version", "match_id", name="uq_shadow_prediction"),
    )


class ShadowPredictionPerformance(Base):
    """Evaluation of shadow model predictions."""

    __tablename__ = "shadow_prediction_performance"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    shadow_prediction_id: Mapped[int] = mapped_column(Integer, ForeignKey("shadow_predictions.id"), nullable=False)
    was_correct: Mapped[bool] = mapped_column(Boolean, nullable=False)
    over_2_5_was_correct: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    btts_was_correct: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    brier_score: Mapped[float] = mapped_column(Float, nullable=False)
    log_loss: Mapped[float] = mapped_column(Float, nullable=False)
    evaluated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_shadow_perf_shadow_pred", "shadow_prediction_id"),
        UniqueConstraint("shadow_prediction_id", name="uq_shadow_perf_shadow"),
    )


class OperationalEvent(Base):
    """Operational events for monitoring and alerting."""

    __tablename__ = "operational_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    component: Mapped[str] = mapped_column(String(100), nullable=False)
    severity: Mapped[str] = mapped_column(String(20), nullable=False)  # info, warning, critical
    event_type: Mapped[str] = mapped_column(String(50), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    context: Mapped[Optional[dict]] = mapped_column(Text, nullable=True)  # JSON
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    resolved_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("ix_operational_events_component_time", "component", "created_at"),
        Index("ix_operational_events_severity", "severity"),
    )