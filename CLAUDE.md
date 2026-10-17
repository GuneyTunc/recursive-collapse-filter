# CLAUDE.md

Bu dosya, bu depoda çalışırken Claude Code'a yol gösterir.

## Proje

**Yinelemeli LLM Eğitiminde Model Çökmesinin Veri Madenciliği Yöntemleriyle Hafifletilmesi.**
Küçük bir dil modeli (SmolLM-135M) kendi sentetik çıktılarıyla nesiller boyu eğitilir. Çöküşü
(varyans/kuyruk kaybı) LLM hakemi kullanmadan, iki eksenli deterministik bir filtreyle
(**Dual-Filter**) hafifletmek hedeflenir:

1. **Yoğunluk ekseni:** Embedding uzayında yoğunluk sırası (k-NN / KDE / HDBSCAN) çıkarılır; en seyrek kısım "kuyruk", kalanı "merkez". Merkez ters-yoğunluk ağırlıklı örneklemeyle seyreltilir.
2. **Gürültü ekseni:** Gerçek doğrulama verisine kalibre edilmiş sınırlar: dondurulmuş referans modelin log-perplexity'si ve ucuz metin sinyalleri (3-gram tekrar oranı, gzip oranı, alfabe dışı karakter, uzunluk).
3. **Karışım:** Seçilen setin `center_frac` kadarı merkezden, kalanı temiz kuyruktan gelir (varsayılan %70/%30).

- Orijinal fikir: `raporlar/Model Collapse.pdf`
- Plan, geliştirme noktaları, hipotezler, fazlar: `PROJE_PLANI.md` — kapsam veya deney tasarımı değiştiğinde güncelle.
- Literatür PDF'leri: `literature/`

## Klasör Yapısı

```
configs/base.yaml        # tüm parametreler ve açıklamaları; tek doğruluk kaynağı
configs/lines/*.yaml     # deney hatları (base'den miras alır, yalnızca farkı yazar)
configs/smoke.yaml       # dakikalar süren uçtan uca kontrol
src/pipeline.py          # nesil döngüsü ve çıktı düzeni (docstring'de ayrıntılı)
src/lm.py                # model/tokenizer yükleme, doküman kodlama, perplexity
src/train/finetune.py    # saf PyTorch ince ayar döngüsü
src/generate/sampler.py  # sentetik üretim
src/features.py          # önbellekli embedding ve referans PPL
src/filters/             # heuristics, perplexity, density, semdedup, selection (dual_filter)
src/metrics/             # text.py, embedding.py, report.py (toplama + grafikler)
scripts/toy_gaussian.py  # F1 oyuncak deneyi (GPU gerekmez)
tests/                   # birim testleri (model gerektirmez)
results/                 # küçük tablolar ve şekiller (git'e girer)
outputs/, data/          # ağırlıklar, havuzlar, embedding'ler (git'e GİRMEZ)
```

## Deney Kuralları (bozulmaması gereken değişmezler)

- **Veri rejimi:** Ana deney *replace*: nesil g yalnızca nesil g-1 modelinin ürettiği havuzdan seçilen veriyle eğitilir. Her nesil **aynı baz ağırlıklardan** (`model.base`) başlar.
- **Sabit N:** Her nesilde eğitim kümesi tam N örnektir; havuz `oversample * N` boyutundadır ve her hat aynı havuz boyutunu kullanır. Bu yüzden `raw` hattı (havuzdan rastgele N) aynı zamanda boyut eşlenmiş rastgele baseline'dır; ayrı bir `random` hattı yoktur.
- **Paylaşılan nesil-0:** Nesil-0 modeli ve nesil-1 havuzu tohum başına `_shared/` altında tüm hatlarca paylaşılır. Aynı tohum için hatları **paralel değil sırayla** çalıştır (ilk çalıştırma `_shared`'ı oluşturur).
- **Test kümesi:** Held-out gerçek test kümesi hiçbir zaman eğitimde veya filtre ayarında kullanılmaz; tüm tohum ve nesillerde aynıdır (`data.split_seed`).
- **Kalibrasyon:** Filtre sınırları el ile seçilmez; gerçek doğrulama verisinin kuantilleridir (`filter.bounds`). Eşik/oran ayarı yalnızca doğrulama verisiyle yapılır, test kümesine bakarak yapılmaz.
- **Referans PPL modeli:** Perplexity, o neslin üreten modeliyle değil, dondurulmuş nesil-0 modeliyle hesaplanır.
- **Kuyruk payı:** `filter.dual.tail_quantile: auto` = `(1-center_frac)*N/M`; temiz kuyruğun tamamı tutulur, yalnızca merkez seyreltilir. Sabit ve daha büyük bir değer kuyruğu da merkezle aynı oranda seyreltir (oyuncak deneyde nadir modu kaybettirdi).
- **Örnekleme:** Varsayılan temperature=1.0, top-p=1.0, `prompt_mode: bos` (koşulsuz; gerçek veriden prompt sızıntısı yok). Diğer ayarlar yalnızca ablasyon olarak kullanılır.
- **Tekrarlanabilirlik:** Her çalıştırma config + tohum ile belirlenir; config ve git hash'i `<line>/seed<k>/config.json` ve her `metrics.json` içine yazılır. Her hat ≥3 tohumla çalıştırılır.
- Yöntemin bir parçası olarak LLM hakemi veya kapalı API kullanılmaz (projenin temel iddiası).

## Çıktı Düzeni

`src/pipeline.py` docstring'ine bak. Özet: `outputs/<line>/seed<k>/gen<g>/{generated,filtered,samples}.jsonl, selection.json, metrics.json`. `metrics.json` en son yazılır ve neslin bittiğini işaretler; yarıda kalan çalıştırma aynı komutla kaldığı yerden devam eder.
Hat adları: `raw`, `density_only`, `ppl_only`, `semdedup`, `dual_filter`, `human_mix`, `dual_human`, `accumulate`.

## Ortam

- Windows 11, RTX 4060 Laptop (8 GB), Python 3.13 (global kurulum, CUDA'lı torch 2.6). transformers 5.x: `from_pretrained(..., dtype=...)` kullanılır, `torch_dtype` değil.
- **Disk dar (C: ~30 GB boş):** Ara model ağırlıkları kaydedilmez (`keep_checkpoints: false`); yalnızca tohum başına nesil-0 modeli saklanır.
- **Proje OneDrive içinde:** Büyük çıktılar senkronize olmasın diye `MC_OUTPUT_DIR` ortam değişkeniyle OneDrive dışına yönlendir.
- Konsolda Türkçe/Unicode için `PYTHONIOENCODING=utf-8` gerekebilir. Dosya yollarında `pathlib` kullan (kullanıcı dizini `Güney`).
- Uzun eğitim/üretim komutlarını arka planda çalıştır. **Darboğaz üretimdir:** batch 128'de ~6.5 doküman/sn (300 token), yani N=10k ve oversample=1.5 ile havuz başına ~40 dk. Batch >128 VRAM'i aşıp WDDM paylaşımlı belleğe taşar ve çok yavaşlar. Özel örnekleme döngüsü hız kazandırmadı (ölçüldü). Dizüstünü prize takıp yüksek performans modunda çalıştır.
- `finetune()` eğitimde KV önbelleğini kapatır; `generate` çağrısında `use_cache=True` açıkça verilmeli (olmazsa ~17 kat yavaş).
- **VRAM 8 GB, taşma sessizdir:** Windows (WDDM) CUDA belleğini hata vermeden paylaşımlı RAM'e taşırır; belirti yalnızca 10–30 kat yavaşlamadır. Mikro-batch 16 eğitimde ~11 GB'a çıkıyordu; bu yüzden 8×2 biriktirme + gradient checkpointing (~3.6 GB). Perplexity (`src/lm.py: token_nll`) LM head'i token parçalarına uygular; tam (batch, uzunluk, 49k sözlük) logit tensörünü oluşturma. Eğitim/PPL sonrasında `free_gpu()` çağrılır. Bellek parametrelerini değiştirirsen `torch.cuda.max_memory_reserved()` ile ölç.

## Kod Kuralları

- Kod, değişken adları ve docstring'ler İngilizce; raporlar ve plan dokümanları Türkçe.
- Sihirli sayı yok: eşikler, oranlar, model adları `configs/base.yaml`'dan gelir; yeni parametreyi oraya açıklamasıyla ekle.
- `src/filters/selection.py` ve `src/metrics/` saf fonksiyonlardır (önceden hesaplanmış dizilerle çalışır); model gerektirmeyen birim testleri `tests/` altına.
- Pahalı adımlar (embedding, referans PPL, üretim) diske önbelleğe alınır.

## Komutlar

```bash
python -m pytest -q tests
python scripts/toy_gaussian.py --out results/toy
python -m src.pipeline --config configs/smoke.yaml --set selection.method=dual --set experiment.line=smoke_dual
python -m src.pipeline --config configs/lines/raw.yaml --seeds 0
python -m src.pipeline --config configs/lines/dual_filter.yaml --set data.n_train=5000
python -m src.metrics.report --results outputs --out results
```
