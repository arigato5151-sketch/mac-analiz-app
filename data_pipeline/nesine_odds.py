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
from dataclasses import dataclass, replace
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
    # Turkish dotless ı does not decompose under NFKD and was being dropped,
    # turning names such as "Iğdır" into a different string.
    value = unicodedata.normalize("NFKD", value.casefold())
    value = "".join(char for char in value if not unicodedata.combining(char)).replace("ı", "i")
    country_aliases = {
        "guney kore": "south korea", "ozbekistan": "uzbekistan",
        "kazakistan": "kazakhstan", "faroe adalari": "faroe islands",
        "danimarka": "denmark", "hollanda": "netherlands",
        "azerbaycan": "azerbaijan", "cebelitarik": "gibraltar",
        "portekiz": "portugal", "avusturya": "austria", "hindistan": "india",
        "almanya": "germany", "ispanya": "spain", "hirvatistan": "croatia",
        "ingiltere": "england", "cekya": "czechia", "arnavutluk": "albania",
        "finlandiya": "finland", "estonya": "estonia", "izlanda": "iceland",
        "isvicre": "switzerland", "slovakya": "slovakia", "slovenya": "slovenia",
        "luksemburg": "luxembourg", "bulgaristan": "bulgaria", "iskocya": "scotland",
        "fyr macedonia": "north macedonia", "karadag": "montenegro",
        "kuzey makedonya": "north macedonia", "kuzey irlanda": "northern ireland",
    }
    for local_name, canonical_name in country_aliases.items():
        value = re.sub(rf"(?<!\w){re.escape(local_name)}(?!\w)", canonical_name, value)
    text = value.encode("ascii", "ignore").decode()
    tokens = re.sub(r"[^a-z0-9]+", " ", text.lower()).split()
    aliases = {"utd": "united", "munchen": "munich"}
    generic = {"fc", "afc", "cf", "sc", "sk", "fk"}
    return " ".join(aliases.get(token, token) for token in tokens if token not in generic)


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
    event_id: str | None = None

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
        event_id = str(row.get("event_id") or "").strip() or None
        parsed.append(NesineMatchQuote(
            home, away, row.get("kickoff_text"), *result, *totals, *btts, markets, event_id
        ))
    return parsed


def match_score(quote: NesineMatchQuote, match: Mapping[str, Any]) -> float:
    home = _normalise(str(match.get("home_team") or match.get("home_name") or ""))
    away = _normalise(str(match.get("away_team") or match.get("away_name") or ""))

    def similarity(left: str, right: str) -> float:
        left_tokens, right_tokens = left.split(), right.split()
        if not left_tokens or not right_tokens:
            return 0.0
        sequence = SequenceMatcher(None, left, right).ratio()
        common = sum(min(left_tokens.count(token), right_tokens.count(token))
                     for token in set(left_tokens) | set(right_tokens))
        token_overlap = 2 * common / (len(left_tokens) + len(right_tokens))
        return max(sequence, token_overlap)

    return (similarity(_normalise(quote.home_team), home) +
            similarity(_normalise(quote.away_team), away)) / 2


def _same_normalised_team(left: str, right: str) -> bool:
    left_name, right_name = _normalise(left), _normalise(right)
    if left_name == right_name:
        return True
    # Accept common harmless feed abbreviations (for example "Crew" / "Cr.")
    # only when the shorter full token is a prefix and all other tokens agree.
    left_tokens, right_tokens = left_name.split(), right_name.split()
    if len(left_tokens) != len(right_tokens):
        return False
    return all(a == b or (len(a) >= 2 and len(b) >= 2 and (a.startswith(b) or b.startswith(a)))
               for a, b in zip(left_tokens, right_tokens))


def match_quotes(quotes: Iterable[NesineMatchQuote], matches: Iterable[Mapping[str, Any]], threshold: float = .82) -> list[tuple[int, NesineMatchQuote, float]]:
    """Return only unambiguous fixture matches above the safety threshold."""
    output: list[tuple[int, NesineMatchQuote, float]] = []
    match_list = list(matches)
    for quote in quotes:
        candidates = sorted(((match_score(quote, match), int(match["id"]), match) for match in match_list), reverse=True)
        if not candidates or candidates[0][0] < threshold:
            continue
        if len(candidates) > 1 and candidates[0][0] - candidates[1][0] < .04:
            continue
        # Only accept a fixture when both team names identify the app fixture
        # exactly after shared Turkish/English and club-prefix normalization.
        if (not _same_normalised_team(quote.home_team, str(candidates[0][2].get("home_team") or ""))
                or not _same_normalised_team(quote.away_team, str(candidates[0][2].get("away_team") or ""))):
            continue
        output.append((candidates[0][1], quote, candidates[0][0]))
    return output


def store_nesine_quotes(db: SupabaseRestClient, matched: Iterable[tuple[int, NesineMatchQuote, float]], captured_at: str | None = None) -> int:
    captured_at = captured_at or datetime.now(timezone.utc).isoformat()
    inserted = 0
    for match_id, quote, _score in matched:
        markets = quote.all_markets()
        if not quote.has_result_market and not markets:
            continue
        if quote.has_result_market:
            db.insert("odds_quote_history", [{"match_id": match_id, "bookmaker": NESINE_BOOKMAKER,
                "odds": quote.snapshot(), "source_updated_at": None, "captured_at": captured_at,
                "is_notification_reference": False}])
        if markets:
            db.insert("nesine_market_quotes", [{"match_id": match_id, **market,
                "odd": float(market["odd"]), "captured_at": captured_at,
                "source_url": DEFAULT_NESINE_URL} for market in markets])
        inserted += 1
    return inserted


def _extract_rows_from_page(page: Any, target_fixtures: set[tuple[str, str]] | None = None) -> list[dict[str, Any]]:
    """Extract 1/X/2 prices from rendered Nesine event cards and table rows."""
    rows = page.evaluate("""targetFixtures => {
      const odd = /^([1-9]\\d?)[.,]\\d{1,3}$/;
      const clean = value => (value || '').replace(/\\s+/g, ' ').trim();
      const targets = new Set(targetFixtures || []);
      const fixtureKey = (home, away) => `${home}|${away}`;
      const splitPair = value => {
        const parts = clean(value).split(/\\s+[-–—]\\s+/);
        return parts.length === 2 && parts.every(part => part && part.length <= 80) ? parts : null;
      };

      // Current desktop rows expose event and selection IDs as data-test-id
      // attributes. Preserve the event ID so its expanded market panel can be
      // opened after the visible fixture list has been matched to the app.
      const codedRows = [...document.querySelectorAll('[data-test-id^="r_"][data-code]')].map(node => {
        const teamLink = node.querySelector('[data-test-id="matchName"]');
        const teams = teamLink && splitPair(teamLink.innerText);
        if (!teams) return null;
        const result = {};
        for (const item of node.querySelectorAll('[data-testid]')) {
          const testId = item.getAttribute('data-testid') || '';
          if (!testId.startsWith('odd_Maç Sonucu_')) continue;
          const selection = testId.slice('odd_Maç Sonucu_'.length);
          if (['1', 'X', '2'].includes(selection)) result[selection] = clean(item.innerText).replace(',', '.');
        }
        if (!result['1'] || !result.X || !result['2']) return null;
        return {event_id: node.getAttribute('data-code'), home_team: teams[0], away_team: teams[1],
          kickoff_text: clean(node.querySelector('[data-testid^="time-"]')?.innerText),
          markets: {"1": result['1'], "X": result.X, "2": result['2']}};
      });
      const matchesTarget = row => Boolean(row) && (!targets.size || targets.has(fixtureKey(row.home_team, row.away_team)));
      const targetedRows = codedRows.filter(matchesTarget);
      // If the normalized names differ from the browser text, return the
      // catalogue to Python so its canonical team matching can decide safely.
      if (targetedRows.length) return targetedRows;
      if (codedRows.length) return codedRows;

      // Current Nesine football fixtures are OCRow_* event cards identified
      // by id. Read only explicit 1/X/2 labels; positional buttons shift when
      // a selection is unavailable and can pull in the next market's price.
      const eventRows = [...document.querySelectorAll('[id^="OCRow_"]')].map(node => {
        const teamLink = [...node.querySelectorAll('a')]
          .map(link => clean(link.innerText)).find(text => splitPair(text));
        const teams = teamLink && splitPair(teamLink);
        if (!teams) return null;
        const result = {};
        for (const item of node.querySelectorAll('[data-testid]')) {
          const testId = item.getAttribute('data-testid') || '';
          if (!testId.startsWith('odd_Maç Sonucu_')) continue;
          const selection = testId.slice('odd_Maç Sonucu_'.length);
          if (['1', 'X', '2'].includes(selection)) result[selection] = clean(item.innerText).replace(',', '.');
        }
        if (!['1', 'X', '2'].every(selection => result[selection])) return null;
        return {event_id: node.getAttribute('data-code') || null, home_team: teams[0], away_team: teams[1],
          markets: {"1": result['1'], "X": result.X, "2": result['2']}};
      });
      if (eventRows.length) {
        const seen = new Set();
        return eventRows.filter(row => {
          const key = `${row.home_team.toLowerCase()}|${row.away_team.toLowerCase()}|${row.markets['1']}|${row.markets.X}|${row.markets['2']}`;
          if (seen.has(key)) return false;
          seen.add(key);
          return true;
        });
      }

      // Unknown layouts are skipped instead of guessing from text order.
      return [];
    }""", [
      f"{_normalise(str(home))}|{_normalise(str(away))}"
      for home, away in (target_fixtures or set())
    ])
    return rows


def _collect_all_rendered_rows(page: Any, *, max_scrolls: int = 80,
                               target_fixtures: set[tuple[str, str]] | None = None) -> list[dict[str, Any]]:
    """Collect quotes while scrolling Nesine's virtualized event list."""
    # Nesine locks document scrolling while its welcome/help layer is mounted.
    # Removing the scroll lock affects only this temporary browser page and lets
    # the site render the next batch of event cards as we move through the list.
    page.evaluate("""() => {
      document.body.classList.remove('stop-scrolling');
      document.body.style.overflow = 'auto';
      document.documentElement.style.overflow = 'auto';
    }""")
    rows_by_fixture: dict[tuple[str, str], dict[str, Any]] = {}
    previous_state: tuple[int, int] | None = None
    stagnant_bottom_steps = 0
    for _ in range(max_scrolls):
        for row in _extract_rows_from_page(page, target_fixtures):
            key = (_normalise(str(row.get("home_team", ""))),
                   _normalise(str(row.get("away_team", ""))))
            if all(key):
                rows_by_fixture[key] = row

        state = page.evaluate("""() => ({
          y: Math.round(window.scrollY),
          height: Math.round(document.body.scrollHeight),
          viewport: Math.round(window.innerHeight)
        })""")
        current_state = (state["y"], state["height"])
        at_bottom = state["y"] + state["viewport"] >= state["height"] - 5
        if at_bottom and current_state == previous_state:
            stagnant_bottom_steps += 1
        else:
            stagnant_bottom_steps = 0
        if stagnant_bottom_steps >= 3:
            break
        previous_state = current_state
        page.evaluate("window.scrollBy(0, Math.max(400, window.innerHeight * 0.75))")
        page.wait_for_timeout(350)
    return list(rows_by_fixture.values())


def _normalise_market_label(value: str) -> str:
    text = unicodedata.normalize("NFKD", value.casefold())
    text = "".join(char for char in text if not unicodedata.combining(char)).replace("ı", "i")
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _market_slug(value: str) -> str:
    return _normalise_market_label(value).replace(" ", "_")[:80] or "unknown"


def _is_requested_nesine_market(value: str, selection_name: str = "") -> bool:
    """Keep only the match, half, goal, team-total and corner markets requested."""
    name = _normalise_market_label(value)
    selection = _normalise_market_label(selection_name)
    # A half-time/full-time selection such as "1/1" is not a standalone
    # half-result price. Keep it out of the requested 1st-half result market.
    if name.startswith(("ilk yari mac sonucu", "1 yari mac sonucu")) and re.search(
        r"(?:^| )[1x2] [1x2](?: |$)", selection
    ):
        return False
    # Nesine labels the requested market as either "1. Yarı Maç Sonucu" or
    # "1. Yarı Sonucu" depending on the bulletin section.
    if name.startswith(("ilk yari mac sonucu", "1 yari mac sonucu")):
        return True
    if name in {
        "mac sonucu", "cifte sans", "ilk yari sonucu", "ikinci yari sonucu",
        "1 yari sonucu", "2 yari sonucu", "1 yarisi sonucu", "2 yarisi sonucu",
        "1 yari mac sonucu", "2 yari mac sonucu", "ilk yari mac sonucu",
        "karsilikli gol", "mac sonucu ve karsilikli gol",
        "1 yarisi 2 yarisi alt ust", "ev sahibi 1 yarisi 2 yarisi alt ust",
        "deplasman 1 yarisi 2 yarisi alt ust",
    }:
        return True
    if re.fullmatch(r"mac sonucu ve \d+ 5 alt ust", name):
        return True
    if re.fullmatch(r"[12] yari \d+ 5 (?:gol )?alt ust", name):
        return True
    if re.fullmatch(r"\d+ 5 (?:gol )?alt ust", name):
        return True
    if re.fullmatch(r"(ev sahibi|deplasman)(?: [12] (?:y|yari))? \d+ 5 (?:gol )?alt ust", name):
        return True
    if re.fullmatch(r"(?:ev sahibi |deplasman )?[12] yari [12] yari alt ust", name):
        return True
    return "korner" in name and "alt ust" in name


def _extract_requested_market_rows(page: Any, event_id: str) -> list[dict[str, str]]:
    """Read selections from Nesine's expanded all-markets panel."""
    raw = page.evaluate("""eventId => {
      const root = document.querySelector(`[data-test-id="${eventId}-all"]`);
      if (!root) return [];
      const output = [];
      // Read semantic attributes, not ordinal positions, so unavailable
      // selections cannot shift odds across rows. Some bulletin versions
      // virtualize the expanded panel, so harvest as its rendered rows change.
      const marketViewport = root.querySelector('[class*="MarketList"], [class*="marketList"], [class*="scroll"]') || root;
      let previousSignature = '';
      let stagnantPasses = 0;
      for (let pass = 0; pass < 80; pass++) {
        for (const selection of root.querySelectorAll('[data-mid][data-test-title][data-test-value]')) {
          let market = selection.parentElement;
          while (market && !market.querySelector('[data-test-m-id]')) market = market.parentElement;
          const marker = market?.querySelector('[data-test-m-id]');
          const heading = marker?.parentElement?.querySelector('span span') || marker?.parentElement;
          output.push({market_name: (heading?.innerText || '').trim(),
            selection_name: selection.getAttribute('data-test-title') || '',
            odd: selection.getAttribute('data-test-value') || ''});
        }
        const current = [...root.querySelectorAll('[data-mid][data-test-title][data-test-value]')]
          .map(node => `${node.getAttribute('data-mid')}|${node.getAttribute('data-test-title')}|${node.getAttribute('data-test-value')}`)
          .join('~');
        stagnantPasses = current === previousSignature ? stagnantPasses + 1 : 0;
        if (stagnantPasses >= 2) break;
        previousSignature = current;
        const scrollers = [marketViewport, ...root.querySelectorAll('*')]
          .filter(node => node.scrollHeight > node.clientHeight + 20);
        const scroller = scrollers.sort((a, b) => b.scrollHeight - a.scrollHeight)[0];
        if (!scroller) break;
        const before = scroller.scrollTop;
        scroller.scrollTop = before + Math.max(450, scroller.clientHeight * 0.75);
        if (scroller.scrollTop === before) {
          const last = scroller.scrollHeight - scroller.clientHeight;
          if (before <= 0) scroller.scrollTop = last;
          else if (before >= last) scroller.scrollTop = 0;
          else break;
          if (scroller.scrollTop === before) break;
        }
      }
      // Remove duplicates created by overlapping virtualized viewports.
      const seen = new Set();
      return output.filter(row => {
        const key = `${row.market_name}|${row.selection_name}|${row.odd}`;
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
      });
    }""", str(event_id))
    return [
        {"market_code": _market_slug(row["market_name"]), "market_name": row["market_name"],
         "selection_code": _market_slug(row["selection_name"]), "selection_name": row["selection_name"],
         "odd": str(_odd(row["odd"]))}
        for row in raw
        if _is_requested_nesine_market(
            str(row.get("market_name", "")), str(row.get("selection_name", ""))
        )
        and row.get("selection_name") and _odd(str(row.get("odd", "")))
    ]


def collect_requested_markets_from_nesine(url: str, event_ids: Iterable[str]) -> dict[str, tuple[dict[str, str], ...]]:
    """Open each matched event's market-count panel and read the requested odds."""
    wanted_ids = list(dict.fromkeys(str(event_id) for event_id in event_ids if str(event_id).strip()))
    if not wanted_ids:
        return {}
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError("Playwright is required for Nesine collection") from exc

    collected: dict[str, tuple[dict[str, str], ...]] = {}
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            page = browser.new_page(locale="tr-TR")
            page.goto(url, wait_until="domcontentloaded", timeout=45_000)
            page.wait_for_timeout(3_000)
            page.evaluate("""() => {
              document.body.classList.remove('stop-scrolling');
              document.body.style.overflow = 'auto';
              document.documentElement.style.overflow = 'auto';
            }""")
            for index, event_id in enumerate(wanted_ids):
                try:
                    row_selector = f'[data-test-id="r_{event_id}"]'
                    count_selector = f'[data-test-id="{event_id}_m"]'
                    # The home page may initially select another sport. Click
                    # its visible football tab first, just as a user would.
                    football = page.get_by_text("Futbol", exact=True).first
                    if football.count():
                        football.evaluate("element => element.click()")
                        page.wait_for_timeout(800)
                    for _ in range(120):
                        if page.locator(row_selector).count() and page.locator(count_selector).count():
                            break
                        page.evaluate("window.scrollBy(0, Math.max(400, window.innerHeight * 0.75))")
                        page.wait_for_timeout(200)
                    if not page.locator(count_selector).count():
                        LOGGER.warning("Nesine detailed markets unavailable for event %s", event_id)
                        continue
                    counter = page.locator(count_selector).locator("span.a4cc134d0a3a892deaa6")
                    (counter if counter.count() else page.locator(count_selector)).evaluate("element => element.click()")
                    page.wait_for_timeout(1_500)
                    markets = _extract_requested_market_rows(page, event_id)
                    if not markets:
                        # Some fixtures expand directly in the row without an
                        # event-wide wrapper; extraction also checks document.
                        page.wait_for_timeout(800)
                        markets = _extract_requested_market_rows(page, event_id)
                    if markets:
                        collected[event_id] = tuple(markets)
                    LOGGER.info("Nesine detail event %s: %s selected-market odds", event_id, len(markets))
                except Exception as exc:
                    LOGGER.warning("Nesine detail skipped for event %s: %s", event_id, type(exc).__name__)
        finally:
            browser.close()
    return collected


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
            locator = page.locator("[class*='OCRow_'], tr, li, [role='row']").filter(has_text=home).filter(has_text=away).first
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


def collect_from_nesine_page(url: str = DEFAULT_NESINE_URL, *, headless: bool = True,
                             target_fixtures: set[tuple[str, str]] | None = None) -> list[NesineMatchQuote]:
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
            # The page virtualizes its event list, so a single DOM read only
            # sees the first screenful. Scroll through the feed and aggregate
            # every fixture before parsing/storing its 1/X/2 market.
            rows = _collect_all_rendered_rows(page, target_fixtures=target_fixtures)
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
    settings = get_settings()
    db = SupabaseRestClient(settings.supabase_url, settings.supabase_service_role_key)
    now = datetime.now(timezone.utc).isoformat()
    raw_matches = db.select_all("matches", columns="id,home_team_id,away_team_id,match_date",
                                filters={"status": "eq.scheduled", "match_date": f"gte.{now}"},
                                order="match_date.asc")
    teams = {int(row["id"]): str(row["name"]) for row in db.select_all("teams", columns="id,name")}
    matches = [{**row, "home_team": teams.get(int(row["home_team_id"]), ""),
                "away_team": teams.get(int(row["away_team_id"]), "")} for row in raw_matches]
    if args.rows_json:
        with open(args.rows_json, encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        target_fixtures = {
            (_normalise(str(match["home_team"])), _normalise(str(match["away_team"])))
            for match in matches if match["home_team"] and match["away_team"]
        }
        payload = [
            {"home_team": item.home_team, "away_team": item.away_team,
             "kickoff_text": item.kickoff_text, "event_id": item.event_id,
             "markets": item.snapshot()}
            for item in collect_from_nesine_page(args.url, headless=not args.headed,
                                                 target_fixtures=target_fixtures)
        ]
    quotes = parse_rendered_text_rows(payload)
    matched = match_quotes(quotes, matches, args.threshold)
    if not args.rows_json:
        event_ids = [quote.event_id for _match_id, quote, _score in matched if quote.event_id]
        detailed = collect_requested_markets_from_nesine(args.url, event_ids)
        matched = [
            (match_id, replace(quote, markets=detailed.get(quote.event_id or "", quote.all_markets())), score)
            for match_id, quote, score in matched
        ]
    stored = store_nesine_quotes(db, matched)
    market_count = sum(len(quote.all_markets()) for _match_id, quote, _score in matched)
    matched_quotes = {id(quote) for _match_id, quote, _score in matched}
    unmatched_samples = []
    for quote in (quote for quote in quotes if id(quote) not in matched_quotes):
        candidates = sorted(((match_score(quote, match), match) for match in matches),
                            key=lambda item: item[0], reverse=True)
        best = candidates[0] if candidates else None
        unmatched_samples.append({
            "nesine": f"{quote.home_team} - {quote.away_team}",
            "best_candidate": (f"{best[1].get('home_team')} - {best[1].get('away_team')}" if best else None),
            "score": round(best[0], 3) if best else None,
        })
        if len(unmatched_samples) == 5:
            break
    report = {"app_fixtures": len(matches), "rows": len(payload), "matched": len(matched), "stored": stored,
              "market_selections": market_count, "unmatched_samples": unmatched_samples}
    print(json.dumps(report, ensure_ascii=False))
    if not quotes:
        raise SystemExit(
            "Nesine collector extracted zero rendered fixtures; page layout, consent overlay, "
            "or odds availability prevented collection"
        )
    if not matched:
        raise SystemExit(
            f"Nesine collector extracted {len(quotes)} fixtures but none matched the {len(matches)} "
            "scheduled app fixtures; see unmatched_samples"
        )
    if not stored:
        raise SystemExit("Nesine collector matched fixtures but stored zero quotes")


if __name__ == "__main__":
    main()
