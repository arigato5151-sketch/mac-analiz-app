# İşletim

Uygulama `SUPABASE_URL` ve `SUPABASE_ANON_KEY` ile yalnızca kayıtlı maç
istatistiklerini okur. Windows zamanlayıcısı her gün 00:05'te fikstürleri ve
sonuçları yeniler; IdDaa oranları 00:17'den başlayarak gün boyunca iki saatte
bir çekilir. Günlük ilk çekimin erken başlayan maçların oranlarını da
yakalayabilmesi için sunucunun gece açık ve internete bağlı olması gerekir.
GitHub Actions ayrıca fikstür/sonuçları günde iki kez uzlaştırır.

İş akışı için GitHub Actions secrets alanlarında `API_FOOTBALL_KEY`,
`SUPABASE_URL` ve `SUPABASE_SERVICE_ROLE_KEY` tanımlı olmalıdır. Yenileme işi
başarısız olursa Actions çalışma günlüğünü inceleyin; mevcut maç kayıtları
Supabase'de kalır.
