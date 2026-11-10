# İkincil futbol verisi kaynakları değerlendirmesi

Tarih: 8 Kasım 2026

## Uygulamanın kapsadığı ligler ve Sportmonks xG kapsamı

Sportmonks'un resmi xG kapsam listesiyle uygulamadaki 34 API-Football lig kimliği karşılaştırıldı. Eşleşen ligler aşağıdadır. Kimlikler sağlayıcıya özgüdür; tabloda soldaki uygulama/API-Football kimliği Sportmonks çağrısında kullanılamaz.

| API-Football ID | Uygulamadaki lig | Sportmonks ID | Sonuç |
|---:|---|---:|---|
| 39 | Premier League | 8 | xG kapsam listesinde |
| 140 | La Liga | 564 | xG kapsam listesinde |
| 135 | İtalya Serie A | 384 | xG kapsam listesinde |
| 78 | Bundesliga | 82 | xG kapsam listesinde |
| 61 | Ligue 1 | 301 | xG kapsam listesinde |
| 2 | UEFA Champions League | 2 | xG kapsam listesinde |
| 3 | UEFA Europa League | 5 | xG kapsam listesinde |
| 848 | UEFA Europa Conference League | 2286 | xG kapsam listesinde |
| 203 | Türkiye Süper Lig | 600 | xG kapsam listesinde |
| 94 | Primeira Liga | 462 | xG kapsam listesinde |
| 88 | Eredivisie | 72 | xG kapsam listesinde |
| 144 | Belçika Pro League | 208 | xG kapsam listesinde |
| 40 | İngiltere Championship | 9 | xG kapsam listesinde |
| 307 | Suudi Arabistan Pro League | 944 | xG kapsam listesinde |
| 253 | MLS | 779 | xG kapsam listesinde |
| 71 | Brezilya Serie A | 648 | xG kapsam listesinde |
| 262 | Liga MX | 743 | xG kapsam listesinde |
| 179 | İskoçya Premiership | 501 | xG kapsam listesinde |
| 218 | Avusturya Bundesliga | 181 | xG kapsam listesinde |
| 207 | İsviçre Super League | 591 | xG kapsam listesinde |
| 13 | CONMEBOL Libertadores | 1122 | xG kapsam listesinde |
| 1 | FIFA World Cup | 732 | xG kapsam listesinde |
| 4 | Euro Championship | 1326 | xG kapsam listesinde |
| 5 | UEFA Nations League | 1538 | xG kapsam listesinde |
| 6 | Africa Cup of Nations | 1117 | xG kapsam listesinde |
| 9 | Copa America | 1114 | xG kapsam listesinde |

Resmi listede eşleşme bulunmayan 8 uygulama ligi: Rusya Premier Liga (235), Arjantin Liga Profesional (128), İtalya Serie B (136), İspanya Segunda División (141), Fransa Ligue 2 (62), Yunanistan Super League 1 (197), Asian Cup (7), CONCACAF Gold Cup (22). Eşleşme olmaması bu liglerin API'de kesinlikle bulunmadığını değil, yayımlanmış xG kapsam listesinde olmadığını belirtir.

Sportmonks dokümanı xG tarihçesinin 2024 sezonundan itibaren başladığını, daha eski sezonlar için xG üretilmediğini söylüyor. Kapsam listesi plan türüne bağlı; maç sonu xG Basic paketle, canlı xG Advanced paketle sunuluyor. Bu nedenle 26/34 lig eşleşmesi, satın alınmış abonelikte her fikstürün xG'si kesin gelecek anlamına gelmiyor. Entegrasyondan önce API deneme hesabında farklı sezonlardan ve liglerden örnek fixture çağrıları yapılmalı.

## xA değerlendirmesi

- API-Football'ın fixture istatistikleri takım bazında maç xA alanı sunmuyor; mevcut `home_xa` / `away_xa` sütunları bu yüzden çoğunlukla boş.
- Sportmonks'un resmi Expected değer türleri listesinde xG, xGoT, xPTS ve türevleri var, xA yok.
- FootyStats dokümanı oyuncu-sezon toplamı ve maç başı ortalaması olarak xA (`xa_total_overall`, `xa_per_game_overall`) veriyor. İncelenen dokümanlar xA'yı tek maç ve takım toplamı olarak sağlamayı doğrulamıyor; bu değerler maç xA sütunlarına yazılamaz.
- Ücretsiz StatsBomb Open Data, seçili turnuva/sezonlar için olay verisi sunuyor. Olaylardan kendi xA metriği hesaplanabilir, fakat kapsam uygulamadaki 34 ligin tamamını kapsamaz ve sağlayıcı tanımıyla aynı xA garantisi vermez.

## Önerilen teknik yol

1. API-Football skor, fikstür ve standart maç istatistiklerinin ana kaynağı olarak kalsın.
2. Sportmonks xG için önce ücretsiz deneme/örnek sorgularla 2024+ sezonlarından birkaç eşleşen lig test edilsin. Test; fixture eşleştirme oranını, eksik xG oranını ve kota/maliyet sınırını kaydetsin.
3. Sağlayıcıdan alınacak xG, mevcut API-Football xG'yi ezmesin. Veritabanında kaynak sağlayıcıyı ve model/ölçüm adını saklayıp, her kaynak değeri ayrı tutalım; AI ve istatistik ekranında kaynağı gösterelim.
4. xA'yı maç takım istatistiği olarak doldurma hedefi şimdilik durdurulsun. İleride olay verisi alınabilen sınırlı liglerde ayrı hesaplanan xA'yı açıkça farklı model etiketiyle tutabiliriz.

## Kaynaklar

- [Sportmonks xG lig kapsamı](https://docs.sportmonks.com/v3/tutorials-and-guides/tutorials/expected/coverage)
- [Sportmonks Expected veri türleri](https://docs.sportmonks.com/v3/definitions/types/expected)
- [Sportmonks xG tarihçesi ve FAQ](https://www.sportmonks.com/faq/)
- [FootyStats oyuncu sezon istatistikleri ve xA alanları](https://footystats.org/api/documentations/player-individual)
- [FootyStats maç detayları API'si](https://footystats.org/api/documentations/match-details)
- [StatsBomb Open Data](https://github.com/statsbomb/open-data)
