# Maç Analiz ve Tahmin

25 futbol ligi ve turnuvası için API-Football verisini Supabase'e aktaran,
Poisson ve XGBoost modelleriyle olasılık üreten kişisel Streamlit uygulaması.

> Tahminler bilgilendirme amaçlı istatistiksel olasılıklardır; kesin sonuç veya
> bahis tavsiyesi değildir. Uygulama bahis işlemi yapmaz.

## Yerel kurulum

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
streamlit run app/main.py
```

Yerelde tüm veri hattını çalıştıracaksanız `.env` içindeki dört alanı da
doldurun: `API_FOOTBALL_KEY`, `SUPABASE_URL`, `SUPABASE_ANON_KEY` ve
`SUPABASE_SERVICE_ROLE_KEY`. Sadece Streamlit arayüzünü çalıştırmak için ilk
iki Supabase alanı yeterlidir. Gizli dosyalar Git tarafından yok sayılır.

## Temel komutlar

```powershell
# Önümüzdeki üç günün fikstürü
python -m data_pipeline.fetch_fixtures --days 3

# Üç sezonluk tarihsel veri
python -m data_pipeline.backfill --seasons 2023 2024 2025

# Kronolojik doğrulamayla model eğitimi
python -m models.train_model

# Gelecek maç tahminleri
python -m models.predict --days 3

# Testler
python -m pytest -q
```

## Mimari

```text
API-Football -> Python veri hattı -> Supabase PostgreSQL
                                      |
                                      +-> XGBoost / Poisson
                                      |
                                      +-> Streamlit
```

API çağrıları zamanlanmış işler tarafından yapılır; Streamlit sayfaları normal
kullanımda Supabase'deki önbelleklenmiş veriyi okur. Veri hattındaki tüm yazma
işlemleri idempotent `upsert` kullanır.

## Veritabanı kurulumu ve güncellemeler

Tek doğruluk kaynağı `db/migrations/` klasöründeki numaralanmış, salt-okunur
migration dosyalarıdır (ledger). Her dosya bir işleme (transaction) sarılıdır ve
SHA-256 özeti `db.migrations` aracıyla `schema_migrations` ledger tablosunda
denetlenir. Üst düzey `db/*.sql` dosyaları yoktur; şemayı elle uygulamayın.

Yeni bir Supabase projesinde SQL Editor üzerinden migration'ları **sırayla**
(000 ile başlayıp en yeniye doğru) uygulayın:

```text
db/migrations/000_migration_ledger.sql
db/migrations/001_prediction_snapshots.sql
db/migrations/002_binary_market_performance.sql
db/migrations/003_shadow_model_evaluation.sql
db/migrations/004_confirmed_lineups.sql
db/migrations/005_odds_quote_history.sql
db/migrations/006_database_telegram_scheduler.sql
db/migrations/007_twenty_minute_telegram_window.sql
db/migrations/008_public_evaluated_results_view.sql
db/migrations/009_availability_history.sql
db/migrations/010_reliable_telegram_delivery.sql
db/migrations/011_simplify_telegram_pre_match_message.sql
db/migrations/012_result_telegram_delivery_queue.sql
db/migrations/013_expected_assists_metrics.sql
db/migrations/014_prediction_quality_metrics.sql
db/migrations/015_operational_event_log.sql
db/migrations/016_fix_availability_count_semantics.sql
db/migrations/017_restrict_raw_prediction_tables.sql
db/migrations/018_rename_pre_match_notification_type.sql
db/migrations/019_add_automatic_pre_match_commentary.sql
db/migrations/020_fix_public_performance_view_permissions.sql
```

Migration'lar idempotent olacak şekilde tasarlanmıştır; yine de üretim
veritabanında uygulamadan önce her birini gözden geçirin. Deployed ledger'ın
repo ile birebir eşleştiğini doğrulamak için (CI'da da çalışır):

```powershell
python -m db.migrations --verify-production
```

Güvenlik notu: `020` public performans görünümünü `security_invoker = false`
(owner-executed) olarak sabitler ve yalnızca dar sütun seçimini anon'a açar —
`prediction_performance`, `prediction_snapshots` ve `team_form` gibi ham tablolar
anon için kapalıdır. Görünüme yeni sütun eklerken anon'a ifşa olmamasına dikkat
edin (RLS owner-executed görünümlerde bypass edilir; sütun seçimi tek savunmadır).

## Dağıtım ve operasyon

| Ortam | Gerekli değişkenler | Yetki |
| --- | --- | --- |
| Streamlit Cloud | `SUPABASE_URL`, `SUPABASE_ANON_KEY` | Salt-okunur kullanıcı arayüzü |
| GitHub Actions | `API_FOOTBALL_KEY`, `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY` | Veri alma, yazma ve model üretimi |
| Yerel tam çalışma | Dört değişkenin tamamı | Geliştirme ve bakım |

`Daily data update` iş akışı sabah fikstür/form/kadro/tahmin güncellemesini,
gece ise sonuç ve performans değerlendirmesini çalıştırır. Sabah çalışmasının
sonunda veri kalite denetimi; eksik tahmin ve 36 saati aşmış form/kadro
bağlamını çalışma özetine yazar. API-Football kota uyarıları ilgili adımın
günlüklerinde görünür. Eksik veya eski bağlam bulunduğunda iş akışı başarısız
olur; böylece sorun sessizce canlıya taşınmaz.

Model artefaktları (`.joblib` + metadata `.json`) git'e değil Supabase Storage'a
saklanır; `Weekly retrain` iş akışı yeni candidate artefaktları
`python -m models.artifact_store --push models/saved_models` ile `models`
bucket'ına yükler ve doğrulama sonrası promote eder. Tahmin/deneme çalışmaları
(`models/predict.py`, `models/shadow.py`) artefaktı önce `models/saved_models/`
altında arar, yoksa `SUPABASE_URL` ve `SUPABASE_SERVICE_ROLE_KEY` ile Storage'dan
indirir (`--pull` ile elle de indirilebilir). Depodaki `.joblib` dosyaları CI'da
güncellenmez; eski track'li artefaktlar artık kaynak olarak kabul edilmez.

## Telegram bildirimleri

Bildirimler isteğe bağlıdır. Yapılandırıldığında sabah iş akışı her yaklaşan
maç için ayrı bir mesaj gönderir; mesaj 1-X-2, Üst/Alt 2.5 ve KG Var/Yok
olasılıklarıyla birlikte API-Football üzerinden gelen güncel Bet365 oranlarını
da içerir. Oran verisi o maçta mevcut değilse tahmin mesajı yine gönderilir.
Gece iş akışı son 30 değerlendirmedeki 1-X-2
performansını gönderir. `Result Telegram notifications` iş akışı her 15 dakikada
bir sonuçlanan maçları değerlendirip her maç için ayrı sonuç kartı yollar; kartta
1-X-2, Üst 2.5 ve KG Var tahminlerinin doğru/yanlış durumu `✓`/`✗` ile gösterilir.
`Pre-match Telegram notifications` iş akışı her 5 dakikada bir kontrol yapar;
maça 25 dakika veya daha az kaldığında ilgili iki takımın form ve kadro bağlamını
yeniler, tahmini günceller ve tek mesaj gönderir. Mesaja, Gemini ile üretilen
bağlama dayalı kısa bir `🧠 Maç yorumu` bölümü de eklenir. Gemini erişilemezse
zaman duyarlı tahmin bildirimi yorum bölümü olmadan yine gönderilir. GitHub'ın
zamanlanmış pre-match çalışmasını geciktirmesine karşı aynı idempotent kontrol,
otomatik sonuç workflow'unda da çalışır. Kurulum:

1. Telegram'da `@BotFather` ile bir bot oluşturun ve bot tokenını alın.
2. Botunuza Telegram'dan `/start` gönderin.
3. Kişisel mesaj için `@userinfobot` ile sayısal chat ID'nizi alın.
4. GitHub deposunda **Settings → Secrets and variables → Actions** alanına
   `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` ve `GEMINI_API_KEY` secrets
   değerlerini ekleyin. `GEMINI_API_KEY`, Google AI Studio'da oluşturulan
   güncel `AQ.` Auth Key olmalıdır.

Tokenı veya chat ID'yi kaynak koda, issue'ya ya da sohbete yazmayın. Secrets
yoksa iş akışı bildirim adımını güvenle atlar; Telegram hatası veri
güncellemesini durdurmaz.

## Güvenlik

- `.env` ve `.streamlit/secrets.toml` repoya alınmaz.
- Streamlit yalnızca `SUPABASE_ANON_KEY` kullanır; bu anahtar yazma yetkisi
  vermez ve istemci kodu mutasyon çağrılarını engeller.
- Anonim erişim, RLS ile yalnızca uygulamanın okuduğu sınırlı veri kümelerine
  (`leagues`, `teams`, `matches`, `team_form`, `predictions`,
  `prediction_performance`, kadro bağlamı ve sonuç görünümü) `SELECT` olarak
  tanımlıdır. `INSERT`, `UPDATE` ve `DELETE` yoktur.
- `SUPABASE_SERVICE_ROLE_KEY` ve `API_FOOTBALL_KEY` yalnızca GitHub Actions
  veya güvenli yerel veri hattında tutulur; Streamlit Cloud secrets alanına
  kesinlikle eklenmez.
- Anahtarlar kaynak kodda, commit geçmişinde veya uygulama ekranlarında
  tutulmaz/gösterilmez. Bir anahtar sızarsa ilgili sağlayıcıdan hemen
  yenileyin.
