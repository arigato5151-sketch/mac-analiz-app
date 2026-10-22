# İşletim

## Otomatik işler

- Windows Görev Zamanlayıcısı fikstür ve sonuçları her gün 00:05'te yeniler.
- Windows IdDaa oran işi 00:17'den başlayarak iki saatte bir çalışır (son
  çalışma 22:17). Sunucu kapalı veya internetsizse bu işler çalışamaz.
- GitHub Actions fikstür ve sonuçları her gün 03:17 UTC ve 20:30 UTC'de
  uzlaştırır; her iki çalışmada da IdDaa oranlarını alır. Bu, Windows işine
  ek bir yedek çalışmadır.

GitHub Actions secrets alanında `API_FOOTBALL_KEY`, `SUPABASE_URL` ve
`SUPABASE_SERVICE_ROLE_KEY` tanımlı olmalıdır. Streamlit arayüzünde yalnızca
`SUPABASE_URL` ile `SUPABASE_ANON_KEY` gerekir.

## Oran kapsamı ve hata takibi

Oran çekici IdDaa karşılaşmalarını lig, normalize edilmiş/alias takım adları ve
başlama saatiyle eşleştirir. Bir hedef maç eşleşmezse çekim kısmi başarısız
olarak raporlanır ve sıfırdan farklı çıkış kodu döndürür. Eşleşen maçların
oranları kaydedilmiş olur; aynı çalışmada otomatik tekrar yapılmaz, çünkü bu
oran geçmişine yinelenen anlık kayıtlar ekler. Sonraki zamanlanmış çekim yeniden
eşleşme deneyebilir.

Windows günlükleri `logs/local_scheduler` altındadır. `PIPELINE FAILED` veya
eşleşmeyen maç uyarılarını inceleyin; lig, takım ve saat bilgileri eşleşme
sorununu teşhis etmek için yazılır. GitHub yedeği başarısızsa Actions çalışma
günlüğünü açın. IdDaa kataloğunda bulunmayan ligler için oran gelmemesi kaynak
kapsamı olabilir; bu durum takım adı alias'ı eklemekle çözülmez.

## Veri saklama

`odds_quote_history` her başarılı çekimden yeni bir oran anlık kaydı tutar.
Eski kayıtları otomatik silen bir politika yoktur; geçmiş oran verisini
korumak için bu denetimde üretim verisine silme/temizleme işlemi yapılmadı.
Tablo büyümesini izleyin ve saklama süresi belirlenmeden veri temizliği
eklemeyin.

Tahmin doğruluğunu değerlendirmek için tahmin çıktıları ve maç sonuçlarını
aynı değerlendirme penceresinde karşılaştırın. Yalnızca modelin geçmiş verideki
ayar seçimini görmek, gelecekte daha doğru tahmin verdiğini kanıtlamaz.
