# Colab'da Çalıştırma Talimatları

Bu belge Colab'a girdikten sonra donanım seçimini, projenin test edilmesini ve deneylerin
çalıştırılmasını adım adım anlatır. Notebook: [`notebooks/colab_train.ipynb`](notebooks/colab_train.ipynb).

## 1. Notebook'u aç

1. [colab.research.google.com](https://colab.research.google.com) → **Dosya → Not defteri aç → GitHub**.
2. `GuneyTunc/recursive-collapse-filter` yaz, dal olarak `main` seç ve `notebooks/colab_train.ipynb`'yi aç.
   (README'deki **Colab'da aç** rozeti de aynı yere götürür.)
3. Değişiklik yapacaksan **Dosya → Drive'a bir kopya kaydet** de. Yoksa düzenlemelerin kaybolur.

## 2. Donanımı seç

**Çalışma zamanı → Çalışma zamanı türünü değiştir → Donanım hızlandırıcı:**

| Seçenek | Ne zaman |
|---------|----------|
| **L4 GPU** | ✅ Varsayılan seçim: ucuz, bf16 destekli, 24 GB. |
| T4 GPU | L4 yoksa ya da compute unit azaldıysa. fp16 kullanır, daha yavaştır. |
| G4 / A100 | Gerek yok: bu küçük model o gücü kullanamaz, saatlik maliyet yüksek. |
| v5e / v6e (TPU) | ❌ Seçme: kod CUDA gerektirir, TPU'da çalışmaz. |

Sağ üstteki **RAM/Disk → Kaynakları görüntüle** panelinden kalan compute unit'leri ve kartın saatlik
tüketimini görebilirsin.

## 3. Projeyi test et (ilk oturum, ~15 dk)

Hücreleri sırayla çalıştır:

1. **GPU kontrolü:** Kart adını ve `precision` satırını kontrol et. L4'te `bf16`, T4'te `fp16 (T4)` yazmalı.
   "GPU yok" hatası alırsan 2. adıma dön.
2. **Ayarlar:** İlk test için şöyle bırak:
   ```python
   LINES = ["raw"]
   SEEDS = [0]
   GEN_BATCH = "auto"   # VRAM'e göre: T4 256, L4 512, A100/G4 1024
   ```
3. **Drive ve kod:** Drive erişimine izin ver. Son satırda en güncel commit görünmeli.
4. **Kurulum ve testler:** Sonunda `40 passed` görmelisin.
   Colab "oturumu yeniden başlat" uyarısı verirse yeniden başlat ve 1. hücreden devam et.
5. **Duman testi:** Birkaç dakika sürer. `gen 0 done`, `gen 1 done`, `gen 2 done` satırları görünmeli,
   `Traceback` görünmemeli.

## 4. Hız ve maliyeti ölç (~1 saat)

1. **Eğitim** hücresini (6) çalıştır.
2. Çıktıda `gen 0 done` satırı göründükten sonra hücreyi **durdur** (■). Çalışma yarıda kalmaz; sonra
   aynı yerden devam eder.
3. **Süre kontrolü** hücresini (8) çalıştır ve `dk` değerini not et. Bu, bir neslin süresidir.
4. Bir neslin maliyeti = süre (saat) × kartın saatlik compute unit tüketimi.
   Bir hat ve bir tohum 5 nesil + nesil-0 sürer.

Karşılaştırma istersen aynı adımları diğer kartta tekrarla ve ucuz olanı seç.

**Ölçülen süre (T4, `generate.batch_size=128`, 2026-11-26):** nesil başına ~31 dk (eğitim 6 dk,
üretim 24 dk ≈ 10.5 doküman/sn), son nesil ~8 dk. `raw` hattı, tek tohum: toplam **2 saat 42 dk**.
Aynı tohumun sonraki hatları nesil-0'ı paylaştığı için ~31 dk daha kısa sürer. Batch 256 ile üretimin
hızlanması beklenir; ilk çalıştırmada 8. hücreden kontrol et.

## 5. Asıl deneyleri çalıştır

1. Hatları şu sırayla çalıştır: `raw` → `dual_filter` → `human_mix` (H1, H2, H4), sonra ablasyonlar:
   `density_only`, `ppl_only`, `semdedup`, `dual_human`, `accumulate`.
2. Her hat için tohumlar `0, 1, 2`. Bir oturumda **bir tohum** önerilir; T4'te bir hat ve tohum ~2–2.7 saat sürer.
3. Örnek ayar:
   ```python
   LINES = ["raw", "dual_filter", "human_mix"]
   SEEDS = [0]
   GEN_BATCH = "auto"
   ```
4. Hepsi bitince **Rapor** hücresini (7) çalıştır. Tablo ve grafikler Drive'da
   `MyDrive/Model_Collapse_Data_Mining/results/` altına yazılır.

## 5.1 Sonuçları yerel bilgisayarda görmek

Colab tüm çıktıları Drive'daki `MyDrive/Model_Collapse_Data_Mining/` klasörüne yazar
(`outputs/`, `data/real/`, `results/`). Bu klasörü yerelde görmek için:

1. [Google Drive masaüstü uygulamasını](https://www.google.com/drive/download/) kur ve aynı Google
   hesabıyla giriş yap.
2. **Ayarlar → Google Drive → "Dosyaları akış olarak aktar" (Stream files)** seçili kalsın.
   "Dosyaları yansıt" (Mirror) seçme: model ağırlıkları ve havuzlar diski doldurur. Akış modunda
   yalnızca açılan dosyalar indirilir.
3. Drive, `G:\My Drive\Model_Collapse_Data_Mining\` (Türkçe arayüzde `G:\Drive'ım\...`) olarak görünür.
   Önemli dosyalar: `results/all_metrics.csv`, `outputs/<hat>/seed<k>/gen<g>/{metrics,selection}.json`,
   `samples.jsonl`.

Drive'daki klasör yalnızca çıktılar içindir. Yerel geliştirme kopyası `Desktop\Projeler\` altında kalır;
Drive'daki kod kopyası kullanılmaz (Colab kodu GitHub'dan klonlar).

## 6. Kurallar ve ipuçları

- **Oturum koparsa:** 1–4. hücreleri, ardından 6. hücreyi çalıştır. Tamamlanan nesiller atlanır.
  Yarıda kalan nesil baştan yapılır.
- **İşin bitince** **Çalışma zamanı → Çalışma zamanının bağlantısını kes ve sil** de. Yoksa compute unit
  harcanmaya devam eder.
- **Deneyi tanımlayan ayarları hatlar arasında değiştirme.** `data.n_train` gibi ayarlar tüm hatlarda
  aynı olmalı. `generate.batch_size` yalnızca hız ayarıdır, karttan karta değiştirilebilir.
- Aynı Drive klasörüne **iki oturumdan aynı anda** yazma: nesil-0 modeli hatlar arasında paylaşılır.
- Tarayıcı sekmesini açık tut. Ücretsiz/Pro oturumların süre ve boşta kalma sınırları vardır.

## 7. Sorun giderme

| Belirti | Çözüm |
|---------|-------|
| `... was produced with a different configuration` | Deneyi tanımlayan bir ayarı değiştirdin. Eski ayara dön ya da `EXTRA`'ya farklı bir `experiment.line=...` ekle. |
| `CUDA out of memory` (hata) | Ayarlar hücresinde `GEN_BATCH`'i otomatik değerin yarısına indir (ör. T4'te `128`). |
| Logda `[W ... CUDACachingAllocator] memory allocation failed with OOM` uyarısı | Çalışma durmadıysa zararsızdır: PyTorch önbelleği boşaltıp yeniden dener. Notebook bunu azaltmak için `expandable_segments` açar. |
| `GPU yok` hatası | Çalışma zamanı türünü GPU olarak değiştir. |
| Kurulumdan sonra garip içe aktarma hataları | **Çalışma zamanı → Oturumu yeniden başlat**, 1. hücreden başla. |
