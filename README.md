# Maç Merkezi

Streamlit uygulaması, günlük fikstürleri ve IdDaa oranlarını Supabase'den okur;
geçmiş maç sonuçları ve istatistiklerinden Poisson tabanlı tahminler üretir.
Maç analizlerinde isteğe bağlı Google Gemini yorumu da oluşturulabilir.
İstatistik arşivinde tamamlanmış maçların skor, xG, xA ve korner verileri
gösterilir. Uygulama ekranı veritabanına yazma yapmaz.

## Yerel çalıştırma

`.env` dosyasına aşağıdaki ayarları ekleyin:

```dotenv
API_FOOTBALL_KEY=...
SUPABASE_URL=https://<project>.supabase.co
SUPABASE_ANON_KEY=...
SUPABASE_SERVICE_ROLE_KEY=...
GOOGLE_API_KEY=...       # İsteğe bağlı Gemini maç yorumu
GOOGLE_MODEL=gemini-3.8-flash  # İsteğe bağlı model seçimi
TELEGRAM_BOT_TOKEN=...  # İsteğe bağlı hata bildirimi
TELEGRAM_CHAT_ID=...    # İsteğe bağlı hata bildirimi
ODDS_HISTORY_ALERT_ROWS=...  # İsteğe bağlı satır sayısı uyarı eşiği
```

Ardından:

```powershell
python -m pip install -r requirements.txt
streamlit run app/main.py
```

Arayüz Supabase anon anahtarıyla salt okunur erişim kullanır. Yerel Gemini
yorumları için `GOOGLE_API_KEY` gerekir; Streamlit Cloud dağıtımında bu anahtarı
uygulamanın Secrets ayarına ekleyin. `GEMINI_API_KEY` ve `GEMINI_MODEL` adları da
geriye dönük uyumluluk için okunur. Servis rolü ve API anahtarı değerlerini
herkese açık depoya koymayın.

GitHub Actions gecelik maç önü analizlerini ve yapay zekâ yorumlarını önceden
üretir. Bunun çalışması için aynı `GOOGLE_API_KEY` değerini depo ayarlarında
**Settings → Secrets and variables → Actions** altında repository secret olarak
tanımlayın. Gece önbelleği, takip eden 36 saatte başlayacak maçları hazırlar.

## Veri akışı

- API-Football fikstür ve sonuçları toplar.
- Sonuç yenilemesi tamamlanan maçların saklanmış maç önü tahminlerini de
  değerlendirir; daha önce işlenmiş tahminleri atlayarak son 14 günü uzlaştırır.
- IdDaa günlük bültenindeki maçlar lig, takım adları ve başlama saatiyle
  uygulama fikstürlerine eşleştirilir; oran geçmişine yeni kayıt eklenir.
- Gece iş akışı Poisson tahminlerini, AI destekli maç yorumunu ve risk profili
  seçimlerini maç başlamadan Supabase önbelleğine yazar. Maç kartı açılırken bu
  içerik önceden hazır olduğu için Gemini isteği beklenmez.
- Eşleşmeyen hedef maç varsa çekim kısmi başarısızlık koduyla biter ve görev
  günlüklerinde sayıları gösterir. Kaydedilmiş maçların tekrar yazılmaması
  için rutin çalıştırmada kısmi sonuç yeniden denenmez. Eşleştirme düzeltmesi
  sonrası eksik event'leri tamamlamak için `python -m
  data_pipeline.fetch_iddaa_odds --date YYYY-MM-DD --allow-partial-retry`
  kullanılabilir; mevcut kayıtlar korunur ve yalnızca eşleşmemiş maçlara yeni
  oran anlığı eklenir.
- Windows görev zamanlayıcısı hata bildirimlerinde ASCII mesaj kullanır ve
  Python çıktısını UTF-8'e sabitler; Türkçe Windows kod sayfası kaynaklı bozuk
  karakterleri önler.
- Günlük oran toplaması Windows Görev Zamanlayıcısı'nda; yedek fikstür,
  sonuç ve oran uzlaştırması GitHub Actions'ta çalışır. Ayrıntılar
  [`OPERATIONS.md`](OPERATIONS.md) dosyasındadır.

Oranlar yalnızca IdDaa'nın sunduğu maç ve market kapsamı için gösterilir.
Kaynakta bulunmayan veya güvenli biçimde eşleştirilemeyen bir maçın oranı
uydurulmaz.
