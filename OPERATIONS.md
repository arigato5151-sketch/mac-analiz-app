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

Hata bildirimi için isteğe bağlı `TELEGRAM_BOT_TOKEN` ve `TELEGRAM_CHAT_ID`
değerlerini Windows sunucusunun `.env` dosyasına ve GitHub Actions secrets
alanına ekleyin. Ayarlar yoksa işler yine çalışır; Windows günlükleri ve
GitHub Actions çalışma özeti hata bilgisini göstermeye devam eder.
`ODDS_HISTORY_ALERT_ROWS` varsayılan olarak 100.000 satırdır; Windows `.env`
dosyasına veya GitHub repository variable alanına farklı eşik yazılabilir. Eşik
aşılınca Telegram uyarısı gönderilir, her çalışmada toplam satır sayısı da
raporlanır.

Günlük iş ayrıca şu andan sonraki 7 günün planlanmış maçlarını, IdDaa oranı
kapsamını ve son oran kaydının yaşını raporlar. Windows günlüklerinde ve GitHub
Actions çalışma özetinde bu değerler görünür; kapsam raporu veri değiştirmez.

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
günlüğünü veya çalışmanın özetindeki eşleşme tablosunu açın. Arayüz seçili gün
için oranı olmayan maç sayısını ve en son kayıt zamanını gösterir; gerekirse
“Verileri şimdi yenile” ile önbelleği temizleyip veriyi yeniden alabilirsiniz.
IdDaa kataloğunda bulunmayan ligler için oran gelmemesi kaynak kapsamı olabilir;
bu durum takım adı alias'ı eklemekle çözülmez.

## Veri saklama

`odds_quote_history` her başarılı çekimden yeni bir oran anlık kaydı tutar.
Her veri işinde satır sayısı tabloyu indirmeden okunur ve Windows günlüğü / GitHub
çalışma özeti içine yazılır. Eski kayıtları otomatik silen bir politika yoktur;
geçmiş oran verisini korumak için otomatik temizlik yapılmaz. Saklama süresi
belirlenmeden veri silme eklemeyin.

Tahmin ekranı lig geçmişinin erken bölümünde seçilen form ağırlığını daha sonraki
ayrı maç diliminde 1X2, 2.5 üst ve karşılıklı gol var için log-loss, Brier skoru
ve 10 dilimli kalibrasyon hatasıyla ölçer; sabit ağırlık karşılaştırmasını ve
örnek sayısını da gösterir. Bu, geçmişteki zaman sıralı değerlendirmedir; canlı
tahmin performansının kaydı değildir ve gelecekte başarıyı garanti etmez.
