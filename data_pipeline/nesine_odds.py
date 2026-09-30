"""Read visible Nesine pre-match odds and persist safe, matched quotes.

The collector deliberately parses rendered rows instead of relying on private
network endpoints. Nesine can change its DOM, so parsing failures are reported
and never create fake odds.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any

from config.settings import get_settings
from db.db_client import SupabaseRestClient

NESINE_BOOKMAKER = "Nesine"
DEFAULT_NESINE_URL = "https://www.nesine.com/iddaa?et=1&le=2"
_ODD = re.compile(r"^(?:[1-9]\d?)(?:[.,]\d{1,3})$")
LOGGER = logging.getLogger(__name__)


def _normalise(value: str) -> str:
    text = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


@dataclass(frozen=True, slots=True)
class NesineMatchQuote:
    home_team: str
    away_team: str
    kickoff_text: str | None
    home_win: str | None
    draw: str | None
    away_win: str | None
    over_2_5: str | None = None
    under_2_5: str | None = None
    btts_yes: str | None = None
    btts_no: str | None = None
    markets: tuple[dict[str, str], ...] = ()

    def snapshot(self) -> dict[str, str | None]:
        return {"home_win": self.home_win, "draw": self.draw, "away_win": self.away_win,
                "over_2_5": self.over_2_5, "under_2_5": self.under_2_5,
                "btts_yes": self.btts_yes, "btts_no": self.btts_no}

    def all_markets(self) -> tuple[dict[str, str], ...]:
        return self.markets

    @property
    def has_result_market(self) -> bool:
        return all(self.snapshot().get(key) for key in ("home_win", "draw", "away_win"))


def _odd(value: str) -> str | None:
    value = value.strip().replace(",", ".")
    return value if _ODD.fullmatch(value) else None


def parse_rendered_text_rows(rows: Iterable[Mapping[str, Any]]) -> list[NesineMatchQuote]:
    """Parse rows extracted from the rendered page.

    Expected row shape: ``home_team``, ``away_team``, optional ``kickoff_text``
    and a mapping/list of market values. This pure function is easy to test and
    keeps browser-specific code out of the data model.
    """
    parsed: list[NesineMatchQuote] = []
    for row in rows:
        home, away = str(row.get("home_team", "")).strip(), str(row.get("away_team", "")).strip()
        if not home or not away:
            continue
        values = row.get("markets") or row.get("odds") or {}
        if isinstance(values, Mapping):
            def get(*keys: str) -> float | None:
                return next(
                    (_odd(str(values[key])) for key in keys if values.get(key) is not None and _odd(str(values[key]))),
                    None,
                )
            result = (get("1", "home_win"), get("X", "draw"), get("2", "away_win"))
            totals = (get("Over 2.5", "over_2_5"), get("Under 2.5", "under_2_5"))
            btts = (get("Var", "Yes", "btts_yes"), get("Yok", "No", "btts_no"))
        else:
            values = [_odd(str(value)) for value in values]
            values = [value for value in values if value]
            result = tuple(values[:3]) + (None,) * max(0, 3 - len(values[:3]))
            totals, btts = (None, None), (None, None)
        raw_markets = row.get("all_markets") or ()
        markets = tuple(
            {"market_code": str(item.get("market_code", "unknown")),
             "market_name": str(item.get("market_name", "Unknown")),
             "selection_code": str(item.get("selection_code", "unknown")),
             "selection_name": str(item.get("selection_name", "Unknown")),
             "odd": str(item.get("odd"))}
            for item in raw_markets
            if isinstance(item, Mapping) and _odd(str(item.get("odd", "")))
        )
        parsed.append(NesineMatchQuote(home, away, row.get("kickoff_text"), *result, *totals, *btts, markets))
    return parsed


def match_score(quote: NesineMatchQuote, match: Mapping[str, Any]) -> float:
    home = _normalise(str(match.get("home_team") or match.get("home_name") or ""))
    away = _normalise(str(match.get("away_team") or match.get("away_name") or ""))
    return (SequenceMatcher(None, _normalise(quote.home_team), home).ratio() +
            SequenceMatcher(None, _normalise(quote.away_team), away).ratio()) / 2


def match_quotes(quotes: Iterable[NesineMatchQuote], matches: Iterable[Mapping[str, Any]], threshold: float = .82) -> list[tuple[int, NesineMatchQuote, float]]:
    """Return only unambiguous fixture matches above the safety threshold."""
    output: list[tuple[int, NesineMatchQuote, float]] = []
    for quote in quotes:
        candidates = sorted(((match_score(quote, match), int(match["id"]), match) for match in matches), reverse=True)
        if not candidates or candidates[0][0] < threshold:
            continue
        if len(candidates) > 1 and candidates[0][0] - candidates[1][0] < .04:
            continue
        output.append((candidates[0][1], quote, candidates[0][0]))
    return output


def store_nesine_quotes(db: SupabaseRestClient, matched: Iterable[tuple[int, NesineMatchQuote, float]], captured_at: str | None = None) -> int:
    captured_at = captured_at or datetime.now(timezone.utc).isoformat()
    inserted = 0
    for match_id, quote, _score in matched:
        if not quote.has_result_market:
            continue
        db.insert("odds_quote_history", [{"match_id": match_id, "bookmaker": NESINE_BOOKMAKER,
            "odds": quote.snapshot(), "source_updated_at": None, "captured_at": captured_at,
            "is_notification_reference": False}])
        if quote.all_markets():
            db.insert("nesine_market_quotes", [{"match_id": match_id, **market,
                "odd": float(market["odd"]), "captured_at": captured_at,
                "source_url": DEFAULT_NESINE_URL} for market in quote.all_markets()])
        inserted += 1
    return inserted


def _extract_rows_from_page(page: Any) -> list[dict[str, Any]]:
    """Extract likely match rows from rendered DOM without depending on CSS classes.

    Nesine's class names are implementation details. This uses visible text and
    numeric odds candidates, then leaves team matching and validation to the
    pure functions above.
    """
    rows = page.evaluate("""() => {
      const odd = /^([1-9]\\d?)[.,]\\d{1,3}$/;
      const nodes = [...document.querySelectorAll('tr, li, [role="row"]')];
      return nodes.map(node => {
        const text = (node.innerText || '').replace(/\\s+/g, ' ').trim();
        const values = [...text.matchAll(/\\b([1-9]\\d?[.,]\\d{1,3})\\b/g)].map(m => m[1]);
        if (values.length < 3) return null;
        const parts = text.split(/\\s{2,}|\\|/).map(s => s.trim()).filter(Boolean);
        const pair = parts.find(p => /\\s[-–—]\\s/.test(p));
        if (!pair) return null;
        const teams = pair.split(/\\s[-–—]\\s/).map(s => s.trim());
        if (teams.length !== 2 || teams.some(t => !t || t.length > 80)) return null;
        return {home_team: teams[0], away_team: teams[1], markets: {"1": values[0], "X": values[1], "2": values[2]}};
      }).filter(Boolean);
    }""")
    if rows:
        return rows
    # Current Nesine layout renders many fixtures as divs rather than rows.
    # Fall back to the stable visible-text pattern: "Home - Away" followed by
    # the three 1/X/2 prices.
    return page.evaluate("""() => {
      const odd = /^([1-9]\\d?)[.,]\\d{1,3}$/;
      const lines = (document.body.innerText || '').split(/\\n+/).map(x => x.trim()).filter(Boolean);
      const result = [];
      for (let i = 0; i < lines.length; i++) {
        if (!/\\s[-–—]\\s/.test(lines[i]) || lines[i].length > 120) continue;
        const teams = lines[i].split(/\\s[-–—]\\s/).map(x => x.trim());
        if (teams.length !== 2 || teams.some(x => !x)) continue;
        const values = [];
        for (let j = i + 1; j < Math.min(i + 10, lines.length) && values.length < 3; j++) {
          if (odd.test(lines[j])) values.push(lines[j].replace(',', '.'));
        }
        if (values.length === 3) result.push({home_team: teams[0], away_team: teams[1], markets: {"1": values[0], "X": values[1], "2": values[2]}});
      }
      return result;
    }""")


def _extract_visible_market_rows(page: Any, home_team: str, away_team: str) -> list[dict[str, str]]:
    """Read visible market/selection/odd triplets from a match detail view."""
    return page.evaluate("""() => {
      const odd = /^([1-9]\\d?)[.,]\\d{1,3}$/;
      const clean = value => (value || '').replace(/\\s+/g, ' ').trim();
      const marketHints = /(maç sonucu|ilk yarı|toplam gol|alt\\s*\\/\\s*üst|kg|karşılıklı|korner|kart|handikap|çifte şans|gol aralığı|oyuncu)/i;
      const nodes = [...document.querySelectorAll('tr, li, [role="row"], [class*="market"], [class*="coupon"]')];
      return nodes.flatMap(node => {
        const cells = [...node.querySelectorAll('th, td, [role="cell"], button, a, span')]
          .map(item => clean(item.innerText)).filter(Boolean);
        const text = clean(node.innerText);
        const values = cells.filter(value => odd.test(value));
        if (!values.length || !marketHints.test(text)) return [];
        const market = cells.find(value => marketHints.test(value)) || text.slice(0, 120);
        return values.map((value, index) => ({
          market_code: market.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_|_$/g, '').slice(0, 80) || 'unknown',
          market_name: market,
          selection_code: String(index + 1),
          selection_name: cells[Math.max(0, cells.indexOf(value) - 1)] || String(index + 1),
          odd: value
        }));
      });
    }""")


def _collect_detail_markets(page: Any, base_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Open match details where possible and attach all visible market rows."""
    collected: list[dict[str, Any]] = []
    for row in base_rows:
        home, away = row["home_team"], row["away_team"]
        try:
            locator = page.locator("tr, li, [role='row']").filter(has_text=home).filter(has_text=away).first
            if locator.count() == 0:
                collected.append(row)
                continue
            locator.click(timeout=3_000)
            page.wait_for_timeout(400)
            detailed = _extract_visible_market_rows(page, home, away)
            collected.append({**row, "all_markets": detailed})
            page.go_back(wait_until="domcontentloaded", timeout=10_000)
            page.wait_for_timeout(300)
        except Exception as exc:
            LOGGER.warning("Nesine market detail skipped for visible row: %s", type(exc).__name__)
            collected.append(row)
    return collected


def collect_from_nesine_page(url: str = DEFAULT_NESINE_URL, *, headless: bool = True) -> list[NesineMatchQuote]:
    """Open Nesine and return rendered, parseable quotes.

    Playwright is imported lazily so the rest of the pipeline remains usable in
    environments where browser binaries are not installed.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError("Playwright is required for Nesine collection") from exc
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=headless)
        try:
            page = browser.new_page(locale="tr-TR")
            page.goto(url, wait_until="domcontentloaded", timeout=45_000)
            page.wait_for_timeout(4_000)
            rows = _collect_detail_markets(page, _extract_rows_from_page(page))
            return parse_rendered_text_rows(rows)
        finally:
            browser.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows-json", help="JSON file containing rendered Nesine rows")
    parser.add_argument("--url", default=DEFAULT_NESINE_URL)
    parser.add_argument("--headed", action="store_true", help="Show Chromium while collecting")
    parser.add_argument("--threshold", type=float, default=.82)
    args = parser.parse_args()
    if args.rows_json:
        with open(args.rows_json, encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        payload = [
            {"home_team": item.home_team, "away_team": item.away_team,
             "kickoff_text": item.kickoff_text, "markets": item.snapshot()}
            for item in collect_from_nesine_page(args.url, headless=not args.headed)
        ]
    settings = get_settings()
    db = SupabaseRestClient(settings.supabase_url, settings.supabase_service_role_key)
    raw_matches = db.select_all("matches", columns="id,home_team_id,away_team_id,match_date",
                                filters={"status": "eq.scheduled"})
    teams = {int(row["id"]): str(row["name"]) for row in db.select_all("teams", columns="id,name")}
    matches = [{**row, "home_team": teams.get(int(row["home_team_id"]), ""),
                "away_team": teams.get(int(row["away_team_id"]), "")} for row in raw_matches]
    matched = match_quotes(parse_rendered_text_rows(payload), matches, args.threshold)
    stored = store_nesine_quotes(db, matched)
    market_count = sum(len(quote.all_markets()) for _match_id, quote, _score in matched)
    report = {"rows": len(payload), "matched": len(matched), "stored": stored, "market_selections": market_count}
    print(json.dumps(report, ensure_ascii=False))
    if not matched:
        raise SystemExit("Nesine collector matched zero fixtures; no odds were stored")


if __name__ == "__main__":
    main()
