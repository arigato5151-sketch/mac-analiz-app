"""Check the deployed public-role table ACLs through the restricted RPC."""

from __future__ import annotations

import json

from config.settings import get_supabase_admin_settings
from db.db_client import SupabaseRestClient


def assert_no_acl_violations(rows: list[dict[str, str]]) -> None:
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
