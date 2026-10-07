"""Render a structured, evidence-bound match analysis report."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _pct(value: Any) -> str:
    return f"%{float(value) * 100:.0f}"


def build_match_analysis_report(
    *,
    home_team: str,
    away_team: str,
    league_name: str,
    expected_goals: Mapping[str, float],
    probabilities: tuple[float, float, float],
    over_25: float,
    btts: float,
    confidence_threshold: float,
    value_label: str | None = None,
) -> str:
    """Build the requested report without inventing unavailable match data."""
    home_probability, draw_probability, away_probability = probabilities
    labels = ("MS 1", "MS X", "MS 2")
    outcome_values = (home_probability, draw_probability, away_probability)
    best_index = max(range(3), key=outcome_values.__getitem__)
    best_label = labels[best_index]
    best_probability = outcome_values[best_index]
    total_xg = expected_goals["home"] + expected_goals["away"]
    criterion = (
        "Yüksek gol potansiyeli" if total_xg >= 2.6
        else "Sıkı ve düşük üretimli mücadele" if total_xg <= 1.8
        else "Dengeli gol beklentisi"
    )
    main_pick = f"{best_label} ({_pct(best_probability)})"
    if best_probability < confidence_threshold:
        main_pick = f"Pas — {best_label} yalnızca {_pct(best_probability)}"
    value_pick = value_label or "Geçerli oran verisi yok; value tahmini yapılmadı"
    alternative = (
        f"2,5 Gol Alt/Üst: Üst ({_pct(over_25)}) · "
        f"Karşılıklı Gol: Var ({_pct(btts)})"
    )
    surprise = "Model eşiğini aşan sürpriz pazar bulunmuyor"
    return f"""### 📊 1. MAÇ ÖZETİ VE GENEL BAKIŞ

- **Karşılaşma:** {home_team} vs {away_team}
- **Lig:** {league_name}
- **Maç Kriteri:** {criterion}
- **Ana Analiz Özeti:** Modelin beklenen gol üretimi {home_team} için {expected_goals['home']:.2f}, {away_team} için {expected_goals['away']:.2f}. Maç Sonucu olasılıkları 1, X ve 2 seçimleri için sırasıyla {_pct(home_probability)}, {_pct(draw_probability)} ve {_pct(away_probability)}. Sakatlık, hakem, korner veya kart verisi sisteme yüklenmediyse bu alanlar tahmine dahil edilmez.

### 🎯 2. MAÇ İÇİN SEÇİLEN EN YÜKSEK OLASILIKLI TAHMİNLER

- **Ana Tahmin (En Güvenli):** {main_pick}
- **Alternatif / Değerli Tahmin:** {value_pick}
- **Sürpriz / Yüksek Oran Tahmini:** {surprise}

### 🎟️ 3. KUPON ÖNERİLERİ

🟢 **A) BANKO / DÜŞÜK RİSKLİ KUPON**

- Tahmin: {main_pick}
- Mantık: Seçim, Maç Sonucu pazarının 1, X ve 2 olasılıkları içindeki en yüksek değere dayanır; garanti değildir.

🟡 **B) İDEAL / DENGELİ KUPON**

- Tahmin: {alternative}
- Mantık: 2,5 Gol Alt/Üst ve Karşılıklı Gol pazarları model olasılıkları üzerinden üretilir; oran/value desteği yoksa sadece istatistiksel sinyaldir.

🔴 **C) YÜKSEK ORAN / SÜRPRİZ KUPON**

- Tahmin: {surprise}
- Mantık: Model eşiği aşılmadığı için veri olmayan bir yüksek oran seçimi uydurulmadı.
"""
