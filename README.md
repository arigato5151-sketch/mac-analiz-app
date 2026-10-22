# Maç Merkezi

Streamlit uygulaması, günlük fikstürleri ve IdDaa oranlarını Supabase'den okur;
geçmiş maç sonuçları ve istatistiklerinden Poisson tabanlı tahminler üretir.
İstatistik arşivinde tamamlanmış maçların skor, xG, xA ve korner verileri
gösterilir. Uygulama ekranı veritabanına yazma yapmaz.

## Yerel çalıştırma

`.env` dosyasına aşağıdaki ayarları ekleyin:

```dotenv
API_FOOTBALL_KEY=...
SUPABASE_URL=https://<project>.supabase.co
SUPABASE_ANON_KEY=...
SUPABASE_SERVICE_ROLE_KEY=...
```

Ardından:

```powershell
python -m pip install -r requirements.txt
streamlit run app/main.py
```

Arayüz Supabase anon anahtarıyla salt okunur erişim kullanır. Servis rolü ve
API-Football anahtarı yalnızca veri toplama işleri içindir; bunları herkese
açık depoya veya Streamlit arayüz ayarlarına koymayın.

## Veri akışı

- API-Football fikstür ve sonuçları toplar.
- IdDaa günlük bültenindeki maçlar lig, takım adları ve başlama saatiyle
  uygulama fikstürlerine eşleştirilir; oran geçmişine yeni kayıt eklenir.
- Eşleşmeyen hedef maç varsa çekim kısmi başarısızlık koduyla biter ve görev
  günlüklerinde sayıları gösterir. Kaydedilmiş maçların tekrar yazılmaması
  için kısmi sonuç yeniden denenmez.
- Günlük oran toplaması Windows Görev Zamanlayıcısı'nda; yedek fikstür,
  sonuç ve oran uzlaştırması GitHub Actions'ta çalışır. Ayrıntılar
  [`OPERATIONS.md`](OPERATIONS.md) dosyasındadır.

Oranlar yalnızca IdDaa'nın sunduğu maç ve market kapsamı için gösterilir.
Kaynakta bulunmayan veya güvenli biçimde eşleştirilemeyen bir maçın oranı
uydurulmaz.
