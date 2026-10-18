# İşletim

Uygulama `SUPABASE_URL` ve `SUPABASE_ANON_KEY` ile yalnızca kayıtlı maç
istatistiklerini okur. Günlük GitHub Actions işi fikstür ve tamamlanmış maç
sonuçlarını API-Football'dan alıp Supabase'e yazar.

İş akışı için GitHub Actions secrets alanlarında `API_FOOTBALL_KEY`,
`SUPABASE_URL` ve `SUPABASE_SERVICE_ROLE_KEY` tanımlı olmalıdır. Yenileme işi
başarısız olursa Actions çalışma günlüğünü inceleyin; mevcut maç kayıtları
Supabase'de kalır.
