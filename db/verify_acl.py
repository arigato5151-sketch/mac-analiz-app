"""Check the deployed public-role table ACLs through the restricted RPC."""

from __future__ import annotations

import json
from collections.abc import Mapping

from config.settings import get_supabase_admin_settings
from db.db_client import SupabaseRestClient

EXPECTED_COLUMNS = {"grantee", "table_name", "privilege_type"}
EXPECTED_GRANTEES = {"anon", "authenticated"}
EXPECTED_TABLES = {
    "schema_migrations",
    "result_notification_queue",
    "notification_log",
    "model_candidates",
    "shadow_predictions",
    "shadow_prediction_performance",
    "personal_match_workspace",
    "personal_match_decisions",
    "bookmaker_odds_snapshots",
}
EXPECTED_PRIVILEGES = {"TRUNCATE", "REFERENCES", "TRIGGER", "MAINTAIN"}


def assert_no_acl_violations(rows: object) -> None:
    if not isinstance(rows, list):
        raise RuntimeError("Production ACL check returned an invalid response")
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != EXPECTED_COLUMNS:
            raise RuntimeError("Production ACL check returned an invalid response")
        if (
            not isinstance(row["grantee"], str)
            or row["grantee"] not in EXPECTED_GRANTEES
            or not isinstance(row["table_name"], str)
            or row["table_name"] not in EXPECTED_TABLES
            or not isinstance(row["privilege_type"], str)
            or row["privilege_type"] not in EXPECTED_PRIVILEGES
        ):
            raise RuntimeError("Production ACL check returned an invalid response")
    if rows:
        details = json.dumps(rows, ensure_ascii=False, sort_keys=True)
        raise RuntimeError(f"Production public table ACL violations: {details}")


def main() -> None:
    settings = get_supabase_admin_settings()
    client = SupabaseRestClient(
        settings.supabase_url, settings.supabase_service_role_key
    )
    violations = client.rpc("verify_public_table_acl")
    assert_no_acl_violations(violations)
    print("Production public table ACL check passed (0 violations)")


if __name__ == "__main__":
    main()
