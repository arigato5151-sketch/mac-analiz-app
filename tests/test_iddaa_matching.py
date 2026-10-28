from data_pipeline.fetch_iddaa_odds import (
    TEAM_ALIASES,
    normalize_team,
    team_names_match,
)


def test_team_alias_groups_do_not_overlap_after_normalization() -> None:
    owners: dict[str, set[str]] = {}
    for canonical, aliases in TEAM_ALIASES.items():
        for name in (canonical, *aliases):
            normalized = normalize_team(name)
            owners.setdefault(normalized, set()).add(canonical)

    collisions = {name: groups for name, groups in owners.items() if len(groups) > 1}
    assert collisions == {}


def test_sporting_names_do_not_cross_match_distinct_clubs() -> None:
    assert team_names_match("Sporting Lisbon", "Sporting CP") is False
    assert team_names_match("Sporting Lisbon", "Sporting Lizbon") is True


def test_exact_and_common_suffix_names_match() -> None:
    assert team_names_match("Bayern München", "Bayern Munich") is True
    assert team_names_match("Queens Park Rangers", "QPR") is True

