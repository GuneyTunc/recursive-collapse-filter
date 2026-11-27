# Proje Planı — Yinelemeli LLM Eğitiminde Model Çökmesinin Veri Madenciliği ile Hafifletilmesi

Kaynak fikir: [raporlar/Model Collapse.pdf](raporlar/Model%20Collapse.pdf)

## 0. Fikrin Özeti (benimsenen çekirdek)

Bir dil modeli kendi sentetik çıktılarıyla nesiller boyu eğitildiğinde dağılımın kuyrukları
kaybolur (erken çöküş) ve sonunda anlamsızlaşır (geç çöküş). Öneri, pahalı LLM hakemleri yerine
**çift eksenli, deterministik ve hafif bir filtre (Dual-Filter Framework)**:

1. **Geometrik yoğunluk madenciliği** — embedding uzayında aşırı yoğun "merkez" örnekleri seyrelt.
2. **Perplexity sınırı** — seyrek "kuyruk" örneklerinde gürültüyü (gibberish) bilgilendirici nadir
   veriden ayır.
3. **Dengelenmiş yeniden besleme** — seyreltilmiş merkez + filtrelenmiş kuyruk karışımıyla bir
   sonraki nesli eğit.

Deney: TinyStories/Cosmopedia tohum verisi, SmolLM-135M / Qwen2.5-0.5B, 5 nesil, kontrol ve deney
hatları.

---

## 1. Geliştirmeye Açık Noktalar

### 1.1 Literatür ve özgünlük konumlandırması
| # | Sorun | Öneri |
|---|-------|-------|
| L1 | **Özgünlük riski:** Embedding tabanlı seyreltme ve PPL filtresi literatürde ayrı ayrı mevcut: SemDeDup (Abbas vd., 2023), D4 (Tirumala vd., 2023), PPL tabanlı budama (Wenzek vd., 2020 – CCNet; Marion vd., 2023; Ankner vd., 2024). | Katkıyı "yeni filtre" değil, **bu iki eksenin yinelemeli (recursive) eğitimde kuyruk korunumu için birleştirilmesi ve nesiller boyu etkisinin ölçülmesi** olarak konumla. Bu çalışmalar ilgili çalışmalar bölümüne girmeli. |
| L2 | Model çökmesinin "kaçınılmaz" olmadığını gösteren kritik çalışma eksik: Gerstgrasser vd. (2024), *Is Model Collapse Inevitable?* — veri **biriktirilirse (accumulate)** çöküş büyük ölçüde engelleniyor; **yerine koyma (replace)** rejiminde oluyor. | Deney rejimi açıkça tanımlanmalı (bkz. M6). Biriktirme rejimi ayrı bir baseline olmalı. |
| L3 | Diğer temel kaynaklar eksik: Dohmatob vd. (2024) *A Tale of Tails* (kuyruk kaybı ↔ ölçekleme yasaları), Alemohammad vd. (2023) *Self-Consuming Generative Models Go MAD*, Bertrand vd. (2023) yinelemeli eğitimin kararlılığı, Feng vd. (2024) doğrulama ile çöküşün önlenmesi. | Literatür taramasına ekle, `literature/` klasörüne PDF'leri koy. |
| L4 | **Atıf hataları:** "Active Learning and Coreset Selection in Deep Learning" başlığı yanlış → doğrusu Sener & Savarese (ICLR 2018), *Active Learning for Convolutional Neural Networks: A Core-Set Approach*. "Textbooks Are All You Need" ilk yazarı Gunasekar vd. (2023, phi-1); Li vd. (2023) devam çalışması phi-1.5'tir. Shumailov vd.'nin Nature sürümünün başlığı farklıdır: *AI models collapse when trained on recursively generated data* (2024). | Atıfları düzelt. (Tüm künyeler son halde tek tek doğrulanmalı.) |
| L5 | Başlıktaki "Önlenmesi" iddiası güçlü. | "Hafifletilmesi / Geciktirilmesi" daha savunulabilir; sonuçlar güçlüyse "önleme" kullanılabilir. |

### 1.2 Yöntem
| # | Sorun | Öneri |
|---|-------|-------|
| M1 | **PPL paradoksu:** Nadir/bilgilendirici kuyruk verisi doğası gereği yüksek PPL'e sahiptir; tek bir üst eşik tam da korunmak istenen kuyruğu eleyebilir. | (a) PPL'i **gerçek (nesil-0) veriyle eğitilmiş/dondurulmuş referans modelle** hesapla, üreten modelle değil. (b) Global değil **küme-içi/yerel (göreli) eşik** kullan. (c) Gibberish için ucuz sinyallerle destekle: n-gram tekrar oranı, gzip sıkıştırma oranı, alfabe dışı karakter oranı, ortalama kelime uzunluğu. (d) Aşırı **düşük** PPL (döngüsel tekrar) için alt eşik de düşün. |
| M2 | Yüksek boyutlu (384–768-d) uzayda KDE güvenilmez ve pahalı; DBSCAN `eps`'e aşırı duyarlı. | Yoğunluğu **k-NN mesafesi** ile tahmin et (FAISS ile hızlı), ya da PCA/UMAP ile 10–50 boyuta indirip KDE uygula. Kümeleme gerekiyorsa **HDBSCAN** kullan. Bu seçenekler ablasyonla karşılaştırılmalı. |
| M3 | "Merkez" ve "kuyruk" tanımı ve seyreltme oranı belirsiz. | Yoğunluk yüzdelikleriyle tanımla (ör. üst %X = merkez). Sert kesme yerine **ters-yoğunluk ağırlıklı örnekleme** (p ∝ density^-α) alternatifi dene; α hiperparametre. |
| M4 | %70/%30 karışım oranı keyfi. | Hiperparametre yap; 90/10, 70/30, 50/50 ablasyonu. Oranın nesiller boyu sabit mi uyarlanabilir mi olacağını belirle. |
| M5 | Embedding modeli belirtilmemiş; anlamsal embedding sözcüksel/sözdizimsel çöküşü kaçırabilir. | `all-MiniLM-L6-v2` veya `bge-small-en-v1.5`. Yoğunluğu isteğe bağlı olarak TF-IDF/n-gram özellikleriyle de hesaplayıp karşılaştır. |
| M6 | **Veri rejimi tanımsız:** her nesil yalnızca bir önceki neslin verisiyle mi (replace), yoksa birikmiş veriyle mi (accumulate) eğitiliyor? Her nesil önceden eğitilmiş baz modelden mi başlıyor, önceki nesilden mi? | Ana deney: **replace** rejimi (çöküşün gözlendiği zor senaryo), her nesil **aynı baz ağırlıklardan** ince ayar (Shumailov vd. protokolü). Her nesilde veri boyutu sabit (N örnek). Filtre veri kaybettiriyorsa fazla üretip (oversampling) N'e tamamla. |
| M7 | Örnekleme ayarları belirtilmemiş; sıcaklık/top-p kesmesi çöküşü doğrudan hızlandırır. | Varsayılan: temperature=1.0, top-p=1.0 (saf örnekleme). Ek analiz: top-p=0.9 ile çöküşün hızlanması. Prompt'lar: gerçek verideki hikâye başlangıçları. |
| M8 | "Kuramsal köprü" iddiası deneysel olarak desteklenmiyor. | Ucuz bir **oyuncak deney** ekle: 1-B/2-B Gauss karışımında yinelemeli tahmin → varyans çöküşü, ardından aynı yoğunluk filtresinin etkisi. Hem sezgi hem şekil sağlar, GPU gerektirmez. |

### 1.3 Deney tasarımı ve değerlendirme
| # | Sorun | Öneri |
|---|-------|-------|
| E1 | **Değerlendirme metrikleri hiç tanımlanmamış.** | Bkz. Bölüm 3. |
| E2 | **Tek baseline (ham veri) yetersiz:** filtre veri miktarını da azalttığı için iyileşmenin "filtreden" mi yoksa "daha az/farklı veriden" mi geldiği ayrılamaz. | Bkz. Bölüm 2 deney hatları: **rastgele alt küme** (aynı boyut), yalnız-yoğunluk, yalnız-PPL, SemDeDup, insan verisi karıştırma (%30 gerçek veri), biriktirme rejimi. |
| E3 | İstatistiksel geçerlilik yok. | Her hat için ≥3 tohum (seed); ortalama ± standart sapma / %95 GA raporla. |
| E4 | TinyStories zaten GPT-3.5/4 üretimi (sentetik). | Bunu açıkça belirt: nesil-0 = "gerçek dağılım vekili". Birincil veri: TinyStories (küçük model ve bütçe için en uygun); Cosmopedia ikincil/opsiyonel genelleme deneyi. |
| E5 | Qwen2.5-0.5B devasa ön eğitimli; ince ayarda çöküş dinamiği farklı ve pahalı. | Birincil: **SmolLM-135M** ince ayarı. Qwen2.5-0.5B yalnızca ölçek kontrolü olarak (bütçe kalırsa, tek tohum). |
| E6 | 5 nesil kısa olabilir. | Küçük modelde 5'ten başla; bütçe yeterse 10 nesle uzat. Hangi nesilde metriklerin ayrıştığını raporla. |
| E7 | Hesaplama bütçesi tahmini yok. | Faz 1'de tek nesil süresini ölç, toplam bütçeyi (hat × tohum × nesil) buna göre küçült. |

---

## 2. Deney Hatları

| Hat | Açıklama | Amaç |
|-----|----------|------|
| `raw` | Ham sentetik veri (kontrol, çöken hat) | Çöküşü göstermek |
| `random` | Filtreyle aynı boyutta rastgele alt küme | Veri miktarı etkisini ayırmak (**zorunlu**) |
| `density_only` | Yalnız yoğunluk seyreltme | Ablasyon |
| `ppl_only` | Yalnız PPL/gürültü filtresi | Ablasyon |
| `semdedup` | SemDeDup tarzı anlamsal tekilleştirme | Literatür baseline'ı |
| `dual_filter` | **Önerilen yöntem** | Deney grubu |
| `human_mix` | %30 sabit gerçek veri + %70 sentetik | Endüstri yaklaşımı (üst referans) |
| `accumulate` | Tüm nesillerin verisi biriktirilir | Gerstgrasser vd. referansı (opsiyonel) |

Ek analiz: `dual_filter + human_mix` birleşimi (yöntem endüstri yaklaşımına ek fayda sağlıyor mu?).

## 3. Değerlendirme Metrikleri (her nesil için)

**Kalite / sadakat**
- Gerçek (nesil-0) held-out test kümesinde perplexity — çöküşün birincil göstergesi.
- MAUVE (üretilen vs gerçek metin dağılımı).

**Çeşitlilik / kuyruk korunumu**
- Distinct-n (n=1..4), Self-BLEU.
- Embedding uzayında: ortalama ikili mesafe, kovaryans izi, **Vendi Score**.
- Unigram/bigram dağılımının nesil-0'a KL/JS ıraksaması; kelime dağarcığı kapsamı.
- **Nadir token geri çağırma:** nesil-0'da frekansı alt %10'daki token'ların nesil-k çıktılarında görülme oranı (kuyruk kaybının doğrudan ölçüsü).
- Zipf eğimi değişimi.

**Gürültü / geç çöküş**
- Tekrar oranı (repeated n-gram), gzip sıkıştırma oranı, ortalama uzunluk.

**Maliyet**
- Filtre çalışma süresi ve bellek (yöntemin "hafiflik" iddiası için).

Opsiyonel: küçük bir örneklemde insan veya LLM-hakem değerlendirmesi (yalnızca doğrulama amaçlı; yöntemin parçası değil).

## 4. Hipotezler
- **H1:** `raw` hattında nesil arttıkça test PPL artar, çeşitlilik metrikleri ve nadir token geri çağırma düşer.
- **H2:** `dual_filter`, aynı veri boyutundaki `random` hattına göre çeşitlilik ve kuyruk metriklerinde anlamlı olarak daha iyidir.
- **H3:** İki eksenin birleşimi, tek eksenli ablasyonların her birinden daha iyidir.
- **H4:** `dual_filter`, LLM hakemine gerek olmadan `human_mix`'e yakın sonuç verir (veya ona ek fayda sağlar).

## 5. Fazlar ve Teslimatlar

| Faz | İçerik | Teslimat |
|-----|--------|----------|
| **F0 – Hazırlık** | Literatür PDF'leri, atıf düzeltmeleri, git, ortam (Python sanal ortamı, PyTorch, transformers, sentence-transformers, faiss, hdbscan) | `literature/`, `requirements.txt`, düzeltilmiş literatür özeti |
| **F1 – Oyuncak deney** | Gauss karışımında yinelemeli tahmin + yoğunluk filtresi | `notebooks/01_toy_gaussian.ipynb`, şekiller |
| **F2 – Altyapı** | Veri yükleme, nesil-0 ince ayarı, üretim, metrik modülü, config sistemi; tek nesil süre ölçümü | `src/` modülleri, nesil-0 modeli, bütçe tahmini |
| **F3 – Baseline çöküş** | `raw` ve `random` hatları, 5 nesil | Çöküşün yeniden üretildiğini gösteren eğriler (H1) |
| **F4 – Filtre geliştirme** | Yoğunluk (k-NN/KDE/HDBSCAN) ve PPL/gürültü bileşenleri; eşik ve oran ayarı **yalnız nesil-1 doğrulama verisinde** | `src/filters/`, filtre analiz notebook'u (elenen/korunan örnek incelemesi) |
| **F5 – Ana deneyler** | Tüm hatlar × ≥3 tohum × 5 nesil | `results/` altında metrik tabloları |
| **F6 – Ablasyon & analiz** | Oran, α, embedding modeli, örnekleme sıcaklığı ablasyonları; istatistiksel testler | Şekiller, tablolar |
| **F7 – Raporlama** | Makale/rapor yazımı, sonuçların tartışılması, sınırlılıklar | `raporlar/` altında final rapor |

## 6. Riskler

| Risk | Önlem |
|------|-------|
| Küçük modelde 5 nesilde belirgin çöküş görülmemesi | Saf örnekleme + replace rejimi; gerekirse veri boyutunu küçült veya nesil sayısını artır |
| Filtrenin yalnızca rastgele seyreltme kadar iyi olması | Bu da raporlanabilir bir sonuç; ablasyonlarla nedenini analiz et |
| GPU bütçesinin yetmemesi | Qwen deneylerini çıkar, tohum sayısını 3'te tut, test kümesini sabit ve küçük tut |
| PPL filtresinin kuyruğu da elemesi | Referans modeli nesil-0 verisinden kur, yerel eşik kullan, elenen örnekleri elle incele |

---

## 7. Uygulama Durumu ve Uygulamada Alınan Kararlar (2026-10-15)

**Tamamlanan:** F0 (ortam, git, bağımlılıklar), F1 (oyuncak deney), F2 altyapısı (veri, ince ayar,
üretim, filtreler, metrikler, raporlama, birim testleri). Komutlar için `CLAUDE.md`'ye bak.

**Plandan sapmalar / netleşen tasarım kararları**
- **`random` hattı kaldırıldı:** Her hat aynı boyutta havuzdan tam N örnek seçtiği için `raw`
  (havuzdan rastgele N) zaten boyut eşlenmiş rastgele baseline'dır (E2'nin amacı tasarımla karşılanır).
- **Gürültü sınırları gerçek veriye kalibre:** PPL ve sezgisel özellik eşikleri el ile seçilmez;
  gerçek doğrulama verisinin %1/%99 kuantilleridir (M1'in uygulanışı). Alt PPL sınırı döngüsel
  tekrarları, üst sınır anlamsız metni yakalar.
- **Dejenere kuantil toleransı (2026-10-25):** Smoke testte alfabe dışı karakter oranının %99
  kuantili 0 çıktı (1000 gerçek hikâyeden yalnızca 1'inde bu tür karakter var), bu yüzden tek bir
  "$" içeren metin eleniyor ve nesil-1 havuzunun %37'si bu sınıra takılıyordu.
  `filter.bounds.heuristic_tolerance.nonalpha_ratio: 0.005` eklendi: sınır bundan daha katı olamaz.
  Değer, gerçek veride görülen en yüksek değerin (0.0022) yaklaşık iki katı; smoke havuzlarındaki
  kod, markup ve Latin dışı alfabe içeren çıktılar bunun üstünde. Nesil-1 havuzunda sezgisel
  sınırların elediği metin sayısı 99'dan 45'e indi.
- **Yoğunluk sıraya dönüştürülür:** k-NN / KDE / HDBSCAN skorları karşılaştırılabilir olmadığından
  seçim, yoğunluk sırası (0, 1] üzerinden yapılır; merkez ağırlığı `rank^-alpha` (M2, M3).
- **`tail_quantile: auto`:** Oyuncak deneyde sabit 0.3 değeri kuyruğu merkezle aynı oranda seyreltti
  ve nadir mod raw'dan daha hızlı kayboldu. `auto = (1-center_frac)·N/M` ile temiz kuyruğun tamamı
  tutulur, yalnızca merkez seyreltilir.
- **Koşulsuz üretim (`prompt_mode: bos`):** Gerçek veriden prompt almak her nesle gerçek bilgi
  sızdırıp çöküşü yapay olarak yavaşlatır; `real_prefix` yalnızca ablasyon.
- **Ek hat `dual_human`:** Dual filtre + %30 gerçek veri (H4'ün ikinci yarısı).
- **Disk:** Ara nesil ağırlıkları saklanmaz; çalıştırma havuz dosyalarından kaldığı yerden devam eder.

**Oyuncak deney (F1) ilk bulguları** (N=200, 30 nesil, 5 tohum, 4 bileşenli 2-B GMM, son nesil):

| Hat | KL(gerçek‖model), %2 gürültü | KL, gürültüsüz | Nadir mod kütlesi (gerçek 0.04) |
|-----|------|------|------|
| raw | 2.39 ± 0.89 | 9.74 ± 5.56 | 0.010 / 0.029 |
| density_only | 6.15 ± 0.25 | 3.74 ± 0.39 | 0.001 / 0.012 |
| dual | 0.74 ± 0.61 | 0.70 ± 0.56 | 0.087 / 0.096 |
| human_mix | 0.71 ± 0.29 | 0.69 ± 0.26 | 0.010 / 0.030 |

- Dual filtre, gerçek veri kullanmadan human_mix düzeyinde KL'ye ulaşıyor.
- **Yalnız-yoğunluk tehlikeli:** Ters-yoğunluk ağırlıklandırma gürültüyü ve uç noktaları kayırıp
  varyansı patlatıyor; gürültü ekseni hem gürültüyü eliyor hem de bu pozitif geri beslemeyi frenliyor.
  Bu, iki eksenin birlikte gerekli olduğu iddiasını (H3) destekliyor.
- **Sınırlılık:** Dual filtre kuyruğu fazla temsil ediyor (nadir mod ≈0.09 > 0.04, varyans ≈14 > 9.7).
  `center_frac` / `center_alpha` ablasyonu (F6) bu dengeyi ölçmeli.

**Uçtan uca doğrulama:** 8 hattın tamamı küçük ölçekte (N=150, 2 nesil, 1 tohum) hatasız çalıştı ve
rapor modülü tablolarla grafikleri üretti. Bu ölçekteki sayılar bilimsel sonuç değildir.

**Mühendislik bulguları (RTX 4060 Laptop, 8 GB):**
- Windows/WDDM, VRAM dolunca hata vermeden paylaşımlı RAM'e taşıyor ve 10–30 kat yavaşlıyor.
  Mikro-batch 16 ile eğitim ~11 GB'a çıkıyordu → 8×2 biriktirme + gradient checkpointing (~3.6 GB).
- Perplexity hesabı tam logit tensörü yerine parçalı LM head ile yapılıyor (1.5 GB, ~15 kat hızlı).
- Üretimde KV önbelleği açıkça açılmalı (`use_cache=True`); kapalıyken 17 kat yavaş.

**Bütçe tahmini (N=10k, havuz 15k, ölçümlerden):** eğitim ~7 dk, üretim ~38 dk (6.5 doküman/sn),
özellikler + değerlendirme ~4 dk → **nesil başına ~50 dk**, hat×tohum başına (5 nesil) ~4 saat.
7 hat × 3 tohum ≈ 85 saat. Öneri: önce `raw`, `dual_filter`, `human_mix` × 3 tohum (~40 saat; H1, H2,
H4'ün ilk yarısı), sonra ablasyonlar. Bütçe yetmezse `data.n_train=5000` süreyi yarıya indirir.

**İlk gerçek çalıştırma (Colab T4, `raw`, tohum 0, N=10k, 2026-11-26):**

| Nesil | test PPL | rare_recall | distinct-2 | Vendi |
|------:|---------:|------------:|-----------:|------:|
| 0 | 5.19 | 0.301 | 0.332 | 27.0 |
| 1 | 5.50 | 0.278 | 0.332 | 27.0 |
| 2 | 5.66 | 0.258 | 0.340 | 28.0 |
| 3 | 5.76 | 0.247 | 0.336 | 28.3 |
| 4 | 5.85 | 0.230 | 0.340 | 28.6 |
| 5 | 5.93 | 0.201 | 0.341 | 29.3 |

- **H1'in çöküş kısmı tek tohumda gözleniyor:** gerçek test PPL'i 5 nesilde %14 arttı. Nadir kelime
  geri çağırma %33 düştü; dağılımın kuyruğu kayboluyor.
- **Çeşitlilik metrikleri ters yönde:** distinct-2 ve Vendi hafifçe arttı. T=1 saf örneklemede model
  hatalarını biriktirip gerçek dağılımdan uzaklaşan, daha "gürültülü" metin üretiyor (Shumailov vd.'nin
  dil modeli deneylerinde de görülen olasılıksız dizilerin artışı). Bu yüzden çeşitlilik tek başına
  "sağlıklı" yorumlanmamalı; `js_real_gen`, `frechet_real` ve `rare_recall` ile birlikte okunmalı.
  H1'in "çeşitlilik düşer" kısmı bu ayarda tutmayabilir; hipotez metni sonuçlar netleşince güncellenmeli.
- Eğitim kaybı nesil-0'da 1.63 iken sentetik veride ~2.0'a çıkıp sabit kalıyor: sentetik veri gerçek
  veriden daha yüksek entropili.
- **Süre:** nesil başına ~31 dk (üretim 24 dk, 10.5 doküman/sn, batch 128), hat×tohum ~2.7 saat.
  Yerel tahminin (~4 saat) altında. 8 hat × 3 tohum ≈ 55–60 saat (T4).

**İlk karşılaştırma (Colab T4, tohum 0, nesil 5; 2026-11-27):**

| Hat | test PPL | rare_recall | js_real_gen | frechet_real | Vendi |
|-----|---------:|------------:|------------:|-------------:|------:|
| nesil-0 (ortak) | 5.19 | 0.301 | 0.0228 | 0.0445 | 27.0 |
| `raw` | 5.93 | 0.201 | 0.0318 | 0.0648 | 29.3 |
| `dual_filter` | 5.92 | 0.206 | 0.0308 | 0.0581 | 29.5 |
| `human_mix` | **5.51** | **0.255** | **0.0259** | **0.0516** | 30.1 |

- **`dual_filter` ≈ `raw`:** test PPL, nadir kelime ve JS farkları tek tohum gürültüsünün altında.
  Yalnızca `frechet_real` nesil 3'ten itibaren açılan bir fark gösteriyor (0.058 vs 0.065, %10). H2 şu an
  **desteklenmiyor**; ≥3 tohum olmadan karar verilemez.
- **`human_mix` açıkça daha iyi:** PPL nesil 2'den sonra ~5.5'te düzleşiyor, nadir kelime kaybı yarı
  yarıya. %30 gerçek veri çapası beklenen etkiyi veriyor ve hedeflenen üst referans olarak çalışıyor.
- **Olası neden — çöküşün türü:** Bu ayarda çöküş çeşitlilik kaybı olarak değil, gerçek dağılımdan
  uzaklaşma (gürültü birikimi) olarak görünüyor: Vendi tüm hatlarda artıyor, JS/Fréchet artıyor.
  Yoğunluk ekseni çeşitlilik kaybına karşı tasarlandı; uzaklaşmaya karşı asıl araç gürültü ekseni.
  Ancak varsayılan ayarda perplexity sınırı yalnızca kuyruğa uygulanıyor (`ppl_scope: tail`) ve
  seçilen setin %70'ini oluşturan merkez PPL kontrolünden geçmiyor.
- **Süre:** `generate.batch_size=256` ile üretim 1426 → 1117 sn (%22 hızlı). Dual-Filter'ın özellik
  çıkarımı (embedding + referans PPL, 15k doküman) nesil başına ~2.5 dk ekliyor. Nesil başına ~29 dk.

**Seçim istatistikleri (`dual_filter`, tohum 0, havuz 15.000, kuyruk 3.000):**

| Nesil | Elenen (PPL) | Elenen (sezgisel) | Elenen kuyruk | Elenen merkez | Seçilen kuyruk | Seçilen merkez |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 1196 | 425 | 1333 | 206 | 1667 | 8333 |
| 3 | 904 | 416 | 1020 | 250 | 1980 | 8020 |
| 5 | 763 | 433 | 885 | 256 | 2115 | 7885 |

- Kuyruğun %30–44'ü elenir, neredeyse tamamı PPL sınırıyla. Kuyruk gerçekten gürültü yoğun.
- Hedeflenen %30 kuyruk payına ulaşılmaz: temiz kuyruk 1.667–2.115 örnek, açık merkezden doldurulur.
  Gerçekleşen karışım ~%17–21 kuyruk / ~%79–83 merkez.
- Merkezde yalnızca sezgisel filtre çalışır (~%2 elenir). Seçilen setin ~%80'i PPL kontrolünden geçmez.
  Bu, `dual_filter ≈ raw` sonucunu ve `ppl_scope=all` deneyinin gerekliliğini destekler.
- Uzunluk kısalması çöküşün belirgin bir işareti: ortalama kelime sayısı 161 → `raw` 125,
  `dual_filter` 134, `human_mix` 142. Dual filtre kısalmayı yavaşlatıyor (n_words alt sınırı etkisi).

**Sıradaki adım:**
1. ~~Seçim istatistiklerine bak~~ (yukarıda).
2. Tohum 0 ile `ppl_only` (gürültü ekseni havuzun tamamında) ve `dual_filter` + `filter.dual.ppl_scope=all`
   varyantını (ayrı bir `experiment.line` adıyla) çalıştır. Hipotez: uzaklaşma rejiminde kazanç gürültü
   ekseninden gelir.
3. Ardından `raw`, `dual_filter`, `human_mix` için tohum 1 ve 2.
