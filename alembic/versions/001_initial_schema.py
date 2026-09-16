"""Initial schema creation

Revision ID: 001
Revises: 
Create Date: 2026-09-15

"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # leagues table
    op.create_table(
        'leagues',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(200), nullable=False),
        sa.Column('country', sa.String(100), nullable=True),
        sa.Column('season', sa.Integer(), nullable=False),
        sa.Column('logo_url', sa.String(500), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), onupdate=sa.func.now(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_leagues_name', 'leagues', ['name'])
    op.create_index('ix_leagues_season', 'leagues', ['season'])

    # teams table
    op.create_table(
        'teams',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(200), nullable=False),
        sa.Column('short_name', sa.String(50), nullable=True),
        sa.Column('league_id', sa.Integer(), nullable=False),
        sa.Column('country', sa.String(100), nullable=True),
        sa.Column('founded_year', sa.Integer(), nullable=True),
        sa.Column('stadium', sa.String(200), nullable=True),
        sa.Column('logo_url', sa.String(500), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), onupdate=sa.func.now(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['league_id'], ['leagues.id'])
    )
    op.create_index('ix_teams_league_id', 'teams', ['league_id'])
    op.create_index('ix_teams_name', 'teams', ['name'])

    # matches table
    op.create_table(
        'matches',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('league_id', sa.Integer(), nullable=False),
        sa.Column('home_team_id', sa.Integer(), nullable=False),
        sa.Column('away_team_id', sa.Integer(), nullable=False),
        sa.Column('match_date', sa.DateTime(timezone=True), nullable=False),
        sa.Column('status', sa.String(50), nullable=False, server_default='scheduled'),
        sa.Column('home_score', sa.Integer(), nullable=True),
        sa.Column('away_score', sa.Integer(), nullable=True),
        sa.Column('home_xg', sa.Float(), nullable=True),
        sa.Column('away_xg', sa.Float(), nullable=True),
        sa.Column('referee', sa.String(200), nullable=True),
        sa.Column('venue', sa.String(200), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), onupdate=sa.func.now(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['league_id'], ['leagues.id']),
        sa.ForeignKeyConstraint(['home_team_id'], ['teams.id']),
        sa.ForeignKeyConstraint(['away_team_id'], ['teams.id'])
    )
    op.create_index('ix_matches_league_date', 'matches', ['league_id', 'match_date'])
    op.create_index('ix_matches_status_date', 'matches', ['status', 'match_date'])
    op.create_index('ix_matches_teams', 'matches', ['home_team_id', 'away_team_id'])

    # predictions table
    op.create_table(
        'predictions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('match_id', sa.Integer(), nullable=False),
        sa.Column('model_version', sa.String(100), nullable=False),
        sa.Column('prob_home_win', sa.Float(), nullable=False),
        sa.Column('prob_draw', sa.Float(), nullable=False),
        sa.Column('prob_away_win', sa.Float(), nullable=False),
        sa.Column('prob_over_2_5', sa.Float(), nullable=False),
        sa.Column('prob_btts', sa.Float(), nullable=False),
        sa.Column('market_probabilities', sa.Text(), nullable=True),
        sa.Column('predicted_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['match_id'], ['matches.id']),
        sa.UniqueConstraint('match_id', 'model_version', name='uq_prediction_match_model')
    )
    op.create_index('ix_predictions_match_id', 'predictions', ['match_id'])
    op.create_index('ix_predictions_model_version', 'predictions', ['model_version'])

    # prediction_snapshots table
    op.create_table(
        'prediction_snapshots',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('source_prediction_id', sa.Integer(), nullable=False),
        sa.Column('match_id', sa.Integer(), nullable=False),
        sa.Column('snapshot_type', sa.String(50), nullable=False),
        sa.Column('model_version', sa.String(100), nullable=False),
        sa.Column('prob_home_win', sa.Float(), nullable=False),
        sa.Column('prob_draw', sa.Float(), nullable=False),
        sa.Column('prob_away_win', sa.Float(), nullable=False),
        sa.Column('prob_over_2_5', sa.Float(), nullable=False),
        sa.Column('prob_btts', sa.Float(), nullable=False),
        sa.Column('market_probabilities', sa.Text(), nullable=True),
        sa.Column('source_predicted_at', sa.String(50), nullable=False),
        sa.Column('context', sa.Text(), nullable=True),
        sa.Column('captured_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['source_prediction_id'], ['predictions.id']),
        sa.ForeignKeyConstraint(['match_id'], ['matches.id']),
        sa.UniqueConstraint('match_id', 'snapshot_type', name='uq_snapshot_match_type')
    )
    op.create_index('ix_prediction_snapshots_match_type', 'prediction_snapshots', ['match_id', 'snapshot_type'])

    # prediction_performance table
    op.create_table(
        'prediction_performance',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('prediction_id', sa.Integer(), nullable=False),
        sa.Column('match_id', sa.Integer(), nullable=False),
        sa.Column('was_correct', sa.Boolean(), nullable=False),
        sa.Column('over_2_5_was_correct', sa.Boolean(), nullable=True),
        sa.Column('btts_was_correct', sa.Boolean(), nullable=True),
        sa.Column('brier_score', sa.Float(), nullable=False),
        sa.Column('log_loss', sa.Float(), nullable=False),
        sa.Column('evaluated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['prediction_id'], ['predictions.id']),
        sa.ForeignKeyConstraint(['match_id'], ['matches.id']),
        sa.UniqueConstraint('prediction_id', name='uq_performance_prediction')
    )
    op.create_index('ix_prediction_performance_prediction_id', 'prediction_performance', ['prediction_id'])
    op.create_index('ix_prediction_performance_match_id', 'prediction_performance', ['match_id'])

    # odds_quote_history table
    op.create_table(
        'odds_quote_history',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('match_id', sa.Integer(), nullable=False),
        sa.Column('bookmaker', sa.String(100), nullable=False),
        sa.Column('bookmaker_id', sa.Integer(), nullable=True),
        sa.Column('odds', sa.Text(), nullable=False),
        sa.Column('source_updated_at', sa.String(50), nullable=True),
        sa.Column('captured_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('is_notification_reference', sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['match_id'], ['matches.id'])
    )
    op.create_index('ix_odds_quote_history_match_captured', 'odds_quote_history', ['match_id', 'captured_at'])
    op.create_index('ix_odds_quote_history_match_bookmaker', 'odds_quote_history', ['match_id', 'bookmaker'])

    # team_form table
    op.create_table(
        'team_form',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('team_id', sa.Integer(), nullable=False),
        sa.Column('calculated_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('elo_rating', sa.Float(), nullable=False),
        sa.Column('avg_goals_scored_last5', sa.Float(), nullable=True),
        sa.Column('avg_goals_conceded_last5', sa.Float(), nullable=True),
        sa.Column('win_rate_last5', sa.Float(), nullable=True),
        sa.Column('home_away_split', sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['team_id'], ['teams.id']),
        sa.UniqueConstraint('team_id', 'calculated_at', name='uq_team_form_team_calculated')
    )
    op.create_index('ix_team_form_team_calculated', 'team_form', ['team_id', 'calculated_at'])

    # player_availability table
    op.create_table(
        'player_availability',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('match_id', sa.Integer(), nullable=True),
        sa.Column('team_id', sa.Integer(), nullable=False),
        sa.Column('player_name', sa.String(200), nullable=False),
        sa.Column('status', sa.String(50), nullable=False),
        sa.Column('source', sa.String(100), nullable=True),
        sa.Column('refreshed_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['match_id'], ['matches.id']),
        sa.ForeignKeyConstraint(['team_id'], ['teams.id'])
    )
    op.create_index('ix_player_availability_team_refreshed', 'player_availability', ['team_id', 'refreshed_at'])
    op.create_index('ix_player_availability_match', 'player_availability', ['match_id'])

    # fixture_lineups table
    op.create_table(
        'fixture_lineups',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('match_id', sa.Integer(), nullable=False),
        sa.Column('team_id', sa.Integer(), nullable=False),
        sa.Column('formation', sa.String(20), nullable=True),
        sa.Column('confirmed_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('coach_name', sa.String(200), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['match_id'], ['matches.id']),
        sa.ForeignKeyConstraint(['team_id'], ['teams.id']),
        sa.UniqueConstraint('match_id', 'team_id', name='uq_lineup_match_team')
    )
    op.create_index('ix_fixture_lineups_match', 'fixture_lineups', ['match_id'])
    op.create_index('ix_fixture_lineups_team', 'fixture_lineups', ['team_id'])

    # match_commentary table
    op.create_table(
        'match_commentary',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('match_id', sa.Integer(), nullable=False),
        sa.Column('commentary_text', sa.Text(), nullable=False),
        sa.Column('model_name', sa.String(100), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['match_id'], ['matches.id']),
        sa.UniqueConstraint('match_id', name='uq_commentary_match')
    )

    # notification_log table
    op.create_table(
        'notification_log',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('match_id', sa.Integer(), nullable=False),
        sa.Column('notification_type', sa.String(50), nullable=False),
        sa.Column('model_version', sa.String(100), nullable=False),
        sa.Column('sent_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['match_id'], ['matches.id']),
        sa.UniqueConstraint('match_id', 'notification_type', name='uq_notification_log')
    )
    op.create_index('ix_notification_log_match_type', 'notification_log', ['match_id', 'notification_type'])

    # pre_match_telegram_queue table
    op.create_table(
        'pre_match_telegram_queue',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('match_id', sa.Integer(), nullable=False),
        sa.Column('message_text', sa.Text(), nullable=False),
        sa.Column('delivered_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['match_id'], ['matches.id']),
        sa.UniqueConstraint('match_id', name='uq_pre_match_queue_match')
    )
    op.create_index('ix_pre_match_queue_delivered', 'pre_match_telegram_queue', ['delivered_at'])

    # model_artifacts table
    op.create_table(
        'model_artifacts',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('model_version', sa.String(100), nullable=False, unique=True),
        sa.Column('model_type', sa.String(50), nullable=False),
        sa.Column('training_rows', sa.Integer(), nullable=False),
        sa.Column('metrics', sa.Text(), nullable=False),
        sa.Column('hyperparameters', sa.Text(), nullable=True),
        sa.Column('training_end', sa.DateTime(timezone=True), nullable=False),
        sa.Column('status', sa.String(50), server_default='trained', nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )

    # shadow_predictions table
    op.create_table(
        'shadow_predictions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('model_version', sa.String(100), nullable=False),
        sa.Column('match_id', sa.Integer(), nullable=False),
        sa.Column('prob_home_win', sa.Float(), nullable=False),
        sa.Column('prob_draw', sa.Float(), nullable=False),
        sa.Column('prob_away_win', sa.Float(), nullable=False),
        sa.Column('prob_over_2_5', sa.Float(), nullable=False),
        sa.Column('prob_btts', sa.Float(), nullable=False),
        sa.Column('market_probabilities', sa.Text(), nullable=True),
        sa.Column('predicted_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['match_id'], ['matches.id']),
        sa.UniqueConstraint('model_version', 'match_id', name='uq_shadow_prediction')
    )
    op.create_index('ix_shadow_predictions_model_match', 'shadow_predictions', ['model_version', 'match_id'])

    # shadow_prediction_performance table
    op.create_table(
        'shadow_prediction_performance',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('shadow_prediction_id', sa.Integer(), nullable=False),
        sa.Column('was_correct', sa.Boolean(), nullable=False),
        sa.Column('over_2_5_was_correct', sa.Boolean(), nullable=True),
        sa.Column('btts_was_correct', sa.Boolean(), nullable=True),
        sa.Column('brier_score', sa.Float(), nullable=False),
        sa.Column('log_loss', sa.Float(), nullable=False),
        sa.Column('evaluated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['shadow_prediction_id'], ['shadow_predictions.id']),
        sa.UniqueConstraint('shadow_prediction_id', name='uq_shadow_perf_shadow')
    )
    op.create_index('ix_shadow_perf_shadow_pred', 'shadow_prediction_performance', ['shadow_prediction_id'])

    # operational_events table
    op.create_table(
        'operational_events',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('component', sa.String(100), nullable=False),
        sa.Column('severity', sa.String(20), nullable=False),
        sa.Column('event_type', sa.String(50), nullable=False),
        sa.Column('message', sa.Text(), nullable=False),
        sa.Column('context', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('resolved_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_operational_events_component_time', 'operational_events', ['component', 'created_at'])
    op.create_index('ix_operational_events_severity', 'operational_events', ['severity'])

def downgrade() -> None:
    op.drop_table('operational_events')
    op.drop_table('shadow_prediction_performance')
    op.drop_table('shadow_predictions')
    op.drop_table('model_artifacts')
    op.drop_table('match_commentary')
    op.drop_table('pre_match_telegram_queue')
    op.drop_table('notification_log')
    op.drop_table('fixture_lineups')
    op.drop_table('player_availability')
    op.drop_table('team_form')
    op.drop_table('odds_quote_history')
    op.drop_table('prediction_performance')
    op.drop_table('prediction_snapshots')
    op.drop_table('predictions')
    op.drop_table('matches')
    op.drop_table('teams')
    op.drop_table('leagues')
