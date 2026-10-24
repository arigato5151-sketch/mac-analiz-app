# Pazar güven eşiği: kronolojik backtest

## Sonuç

Geçmiş tahminlerde güven eşiği kullanarak düşük güvenli seçimleri elemek, test dönemindeki bütün pazarların toplam isabet oranını artırmadı. Bu nedenle test edilen eşikler canlı tahminlere uygulanmadı.

| Test karşılaştırması | Doğru / toplam | İsabet | Kapsam |
|---|---:|---:|---:|
| Tüm tahminler | 2.131 / 3.454 | %61,7 | %100 |
| Pazar bazında seçilen eşikler | 392 / 665 | %58,9 | %19,3 |

Eşikli yaklaşım toplam isabeti 2,8 yüzde puan düşürdü ve tahmin fırsatlarının yaklaşık beşte birini bıraktı. Bazı pazarlar test döneminde iyileşirken bazıları kötüleşti; küçük örnekli artışlar güvenilir bir canlı değişiklik için yeterli kanıt değil.

## Yöntem

- Kaynak: `evaluated_prediction_results` görünümündeki her maç için son değerlendirme.
- Örnek: 26 Ağustos–18 Ekim 2026 arasında 1.569 maç.
- Veriler tarihe göre sıralandı; ilk 1.255 maç eşik seçimi için, son 314 maç dokunulmamış test için kullanıldı. Test dönemi 11–18 Ekim 2026.
- Her pazar için eşik, eğitim bölümünde 0,30–0,95 aralığında 0,05 adımlarla ve en az 50 seçim koşuluyla seçildi. Sonra eşik değiştirilmeden test döneminde ölçüldü.
- 1X2 için en olası sonuç, ikili pazarlar için seçilen tarafın olasılığı, çifte şans için en yüksek olasılıklı seçenek, doğru skor için ilk üç skorun toplam olasılığı kullanıldı.
- Toplam başarı, her pazar kararını eşit ağırlıkla sayar; bahis getirisi veya kârlılık değildir.

## Pazar bazında test

| Pazar | Eşik yok | Eşikli test |
|---|---:|---:|
| 1X2 | %49,4 (155/314) | %80,0 (8/10) |
| Toplam gol 1,5 | %82,2 (258/314) | %76,6 (95/124) |
| Toplam gol 2,5 | %58,9 (185/314) | %75,0 (12/16) |
| Toplam gol 3,5 | %53,8 (169/314) | %55,3 (21/38) |
| Karşılıklı gol | %63,1 (198/314) | %54,5 (30/55) |
| Çifte şans | %84,1 (264/314) | %90,9 (50/55) |
| Ev sahibi gol 0,5 | %82,8 (260/314) | %88,5 (23/26) |
| Ev sahibi gol 1,5 | %54,8 (172/314) | %56,8 (25/44) |
| Deplasman gol 0,5 | %74,5 (234/314) | %74,5 (82/110) |
| Deplasman gol 1,5 | %55,1 (173/314) | %75,0 (9/12) |
| Doğru skor ilk 3 | %20,1 (63/314) | %21,1 (37/175) |

Test dönemi kısa ve bazı pazarların eşikli örneklemi küçük. Sonuç, pazar bazında birden fazla ileri tarihli test penceresinde doğrulanmadan tahmin veya bildirim kurallarının değiştirilmesi için yeterli değil.
