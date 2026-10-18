# Maç İstatistikleri

Uygulama, Supabase'de kayıtlı tamamlanmış maçların sonuç ve saha içi
istatistiklerini salt okunur biçimde gösterir. Eski analiz arayüzü sıfırlanmıştır.

## Çalıştırma

`.env` dosyasında `SUPABASE_URL` ve `SUPABASE_ANON_KEY` değerlerini tanımlayın,
ardından:

```powershell
python -m pip install -r requirements.txt
streamlit run app/main.py
```

Maç kayıtları Supabase'de korunur. Uygulama bu kayıtları değiştirmez veya silmez.
