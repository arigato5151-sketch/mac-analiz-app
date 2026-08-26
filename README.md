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

`.env` içindeki `API_FOOTBALL_KEY`, `SUPABASE_URL` ve
`SUPABASE_SERVICE_ROLE_KEY` alanlarını kendi değerlerinizle doldurun. Gizli
dosyalar Git tarafından yok sayılır.

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

## Güvenlik

- `.env` ve `.streamlit/secrets.toml` repoya alınmaz.
- Supabase tabloları anonim erişime açık değildir.
- Backend işlemleri yalnızca `service_role`/secret key ile çalışır.
- Anahtarlar kaynak kodda tutulmaz.
