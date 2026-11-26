# Maç Merkezi

Maç Merkezi, Streamlit arayüzünde günlük futbol fikstürünü, IdDaa oranlarını,
geçmiş maç verilerini ve Poisson tabanlı maç tahminlerini gösterir. Tahminler ve
önceden hazırlanmış yapay zekâ yorumları Supabase önbelleğinden yüklenir; maç
kartını açarken AI isteği beklenmez.

Gece iş akışı tüm yakın tarihli maçların tahminlerini önbelleğe alır, ancak AI
yorumlarını yalnızca Türkiye saatine göre o gün oynanacak maçlar için üretir.
Diğer günlerin tahminleri kullanılabilir; bu maçların AI yorumları günlük AI
işine girmez. Sağlayıcı kotası veya geçici servis hatası bazı AI yorumlarını
engelleyebilir; temel model tahmini yine önbelleğe yazılır.

## Yerel çalıştırma

`.env` dosyasına uygulama ve veri erişimi için gereken ayarları ekleyin:

```dotenv
API_FOOTBALL_KEY=...
SUPABASE_URL=https://<project>.supabase.co
SUPABASE_ANON_KEY=...
SUPABASE_SERVICE_ROLE_KEY=...

# AI sağlayıcıları; en az bir anahtar gerekir, tümü isteğe bağlıdır.
GROQ_API_KEY=...
GROQ_MODEL=openai/gpt-oss-20b
NVIDIA_API_KEY=...
NVIDIA_MODEL=nvidia/nemotron-3-ultra-550b-a55b
GOOGLE_API_KEY=...
GOOGLE_MODEL=gemini-3.8-flash

TELEGRAM_BOT_TOKEN=...       # İsteğe bağlı hata bildirimi
TELEGRAM_CHAT_ID=...         # İsteğe bağlı hata bildirimi
ODDS_HISTORY_ALERT_ROWS=... # İsteğe bağlı satır sayısı uyarı eşiği
```

Ardından:

```powershell
python -m pip install -r requirements.txt
streamlit run app/main.py
```

Arayüz Supabase anon anahtarıyla salt okunur maç verisi kullanır. Anonim AI
yorum geri bildirimi gönderildiğinde yalnızca oy ve seçilen sorun etiketleri
kaydedilir. Servis rolü ve AI API anahtarlarını herkese açık depoya koymayın.

## Otomatik AI ve veri güncellemesi

GitHub Actions `Match statistics sync` iş akışı her gün Türkiye saatiyle
**01.18'de** çalışır (GitHub cron: `18 22 * * *`, UTC). İş akışı fikstürleri,
oranları ve sonuçları günceller; bugünün tahminlerini ve AI yorumlarını
Supabase'e yazar. GitHub Actions'ta ayrıca `workflow_dispatch` ile elle
çalıştırma seçeneği vardır.

İş akışının çalışması için repository secrets altında şu değerleri tanımlayın:

- `API_FOOTBALL_KEY`
- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`
- AI sağlayıcılarından en az birinin anahtarı: `GROQ_API_KEY`, `NVIDIA_API_KEY`
  veya `GOOGLE_API_KEY`

Sağlayıcı sırası Groq, NVIDIA NIM ve Google Gemini şeklindedir. İsteklerde
sağlayıcıya özel kota ve zaman aşımı sınırları uygulanır. Bir sağlayıcı
başarısız olursa sıradaki denenir. Tümü başarısız olursa temel tahminler
korunur, eksik AI yorumları Actions çalışma günlüğünde raporlanır.

## Veri akışı

- API-Football fikstürleri, tamamlanan maç sonuçlarını ve desteklenen maç
  istatistiklerini sağlar.
- IdDaa günlük bültenindeki maçlar lig, takım adları ve başlama saatiyle
  uygulama fikstürlerine eşleştirilir; bulunan oranlar geçmişe eklenir.
- Gece iş akışı yaklaşan maçların Poisson tahminlerini önbelleğe yazar; AI
  yorumlarını yalnızca İstanbul takvim günündeki maçlar için üretir.
- Oranlar AI yorumu yazıldıktan sonra değişmişse yorum görünmeye devam eder.
  İlgili AI seçimleri, eski oranlara dayandığı uyarısıyla gösterilir.
- Eşleşmeyen hedef maç varsa çekim kısmi başarısızlık koduyla biter ve görev
  günlüklerinde sayıları gösterir. Kaydedilmiş maçların tekrar yazılmaması
  için rutin çalıştırmada kısmi sonuç yeniden denenmez. Eşleştirme düzeltmesi
  sonrası eksik event'leri tamamlamak için `python -m
  data_pipeline.fetch_iddaa_odds --date YYYY-MM-DD --allow-partial-retry`
  kullanılabilir.
- Oranlar yalnızca IdDaa'nın sunduğu maç ve market kapsamı için gösterilir.
  Kaynakta bulunmayan veya güvenli biçimde eşleştirilemeyen bir maçın oranı
  uydurulmaz.

Windows tarafındaki zamanlanmış görevler, hata takibi ve veri saklama ayrıntıları
[`OPERATIONS.md`](OPERATIONS.md) dosyasındadır.
