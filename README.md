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

Yeni bir Supabase projesinde SQL Editor üzerinden şu dosyaları sırasıyla
uygulayın:

```text
db/schema.sql
db/public_read_policies.sql
db/evaluated_prediction_results.sql
db/availability_context.sql
db/pre_match_notifications.sql
```

Mevcut bir kurulumda son iki dosya sonuç ekranının tek-sorgu görünümünü ve
kadro verisinin yenilik işaretçisini ekler. SQL dosyaları tekrarlanabilir
olarak tasarlanmıştır; yine de üretim veritabanında uygulamadan önce gözden
geçirin.

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

## Telegram bildirimleri

Bildirimler isteğe bağlıdır. Yapılandırıldığında sabah iş akışı her yaklaşan
maç için ayrı bir mesaj gönderir; mesaj 1-X-2, Üst/Alt 2.5 ve KG Var/Yok
olasılıklarıyla birlikte API-Football üzerinden gelen güncel Bet365 oranlarını
da içerir. Oran verisi o maçta mevcut değilse tahmin mesajı yine gönderilir.
Gece iş akışı son 30 değerlendirmedeki 1-X-2
performansını gönderir ve yeni sonuçlanan her maç için ayrı sonuç kartı yollar;
kartta her piyasanın doğru/yanlış durumu `✓`/`✗` ile gösterilir. `Pre-match Telegram notifications` iş akışı her 15
dakikada bir kontrol yapar; maçtan 45–75 dakika önce ilgili iki takımın form
ve kadro bağlamını yeniler, tahmini günceller ve tek mesaj gönderir. Kurulum:

1. Telegram'da `@BotFather` ile bir bot oluşturun ve bot tokenını alın.
2. Botunuza Telegram'dan `/start` gönderin.
3. Kişisel mesaj için `@userinfobot` ile sayısal chat ID'nizi alın.
4. GitHub deposunda **Settings → Secrets and variables → Actions** alanına
   `TELEGRAM_BOT_TOKEN` ve `TELEGRAM_CHAT_ID` secrets değerlerini ekleyin.

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
