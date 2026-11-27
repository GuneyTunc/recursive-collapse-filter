"""Build notebooks/colab_train.ipynb (keeps the notebook reviewable as plain Python).

    python scripts/make_colab_notebook.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


CELLS = [
    md("""
# recursive-collapse-filter — Colab'da eğitim

Bu notebook deney hatlarını Colab GPU'sunda çalıştırır. **Tüm çıktılar Google Drive'a yazılır**, böylece
oturum kapansa bile kaybolmaz. Hücreleri yeniden çalıştırınca pipeline kaldığı nesilden devam eder.

Adım adım talimatlar: [COLAB.md](https://github.com/GuneyTunc/recursive-collapse-filter/blob/main/COLAB.md)

1. **Çalışma zamanı → Çalışma zamanı türünü değiştir → L4 GPU** (yoksa T4). TPU seçme.
2. Hücreleri sırayla çalıştır. Yalnızca **Ayarlar** hücresini düzenlemen gerekir.
3. Aynı tohum için hatları **sırayla** çalıştır: nesil-0 modeli tüm hatlarca paylaşılır.
   Aynı Drive klasörüne iki Colab oturumundan aynı anda yazma.
"""),
    md("## 1. GPU kontrolü"),
    code("""
!nvidia-smi --query-gpu=name,memory.total --format=csv
import torch
print("torch", torch.__version__, "| CUDA:", torch.cuda.is_available())
assert torch.cuda.is_available(), "GPU yok: Çalışma zamanı → Çalışma zamanı türünü değiştir → GPU"
major, minor = torch.cuda.get_device_capability()
print("compute capability", f"{major}.{minor}", "→ precision:", "bf16" if major >= 8 else "fp16 (T4)")

# Sampling throughput grows with batch size; ~2.8 GB per 128 documents of 300 tokens was measured.
vram_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
AUTO_GEN_BATCH = 1024 if vram_gb >= 40 else 512 if vram_gb >= 20 else 256 if vram_gb >= 14 else 128
print(f"VRAM {vram_gb:.1f} GB → generate.batch_size={AUTO_GEN_BATCH}")
"""),
    md("## 2. Ayarlar"),
    code("""
REPO_URL = "https://github.com/GuneyTunc/recursive-collapse-filter.git"
BRANCH = "main"

# Drive'daki proje klasörü: outputs/, data/real ve results/ buraya yazılır. Kod buradan değil,
# GitHub'dan klonlanan kopyadan çalışır. Drive masaüstü uygulamasıyla yerelde de görünür.
DRIVE_ROOT = "/content/drive/MyDrive/Model_Collapse_Data_Mining"
LINES = ["raw"]          # sırayla çalışır: raw, dual_filter, human_mix, density_only, ppl_only, semdedup, dual_human, accumulate
SEEDS = [0]              # bir oturumda 1 tohum önerilir (~4 saat / hat)
GEN_BATCH = "auto"       # üretim batch'i: "auto" = VRAM'e göre (T4 256, L4 512, A100/G4 1024); yalnızca hız ayarı
EXTRA = [                # isteğe bağlı config değişiklikleri: "anahtar=değer"
    # "data.n_train=5000",       # süreyi yarıya indirir (tüm hatlarda aynı kullan!)
]
"""),
    md("## 3. Drive ve kod"),
    code("""
from google.colab import drive
drive.mount("/content/drive")

import os, pathlib
REPO_DIR = "/content/recursive-collapse-filter"
if pathlib.Path(REPO_DIR).exists():
    !git -C {REPO_DIR} fetch -q origin && git -C {REPO_DIR} checkout -q {BRANCH} && git -C {REPO_DIR} pull -q --ff-only
else:
    !git clone -q -b {BRANCH} {REPO_URL} {REPO_DIR}
%cd {REPO_DIR}
!git log --oneline -1

os.environ["MC_OUTPUT_DIR"] = f"{DRIVE_ROOT}/outputs"
os.environ["PYTHONIOENCODING"] = "utf-8"
# Variable-length batches fragment the CUDA cache; on a T4 this caused an OOM retry during training.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
DATA_DIR = f"{DRIVE_ROOT}/data/real"  # gerçek veri bir kez indirilir, sonraki oturumlar Drive'dan okur
pathlib.Path(DRIVE_ROOT).mkdir(parents=True, exist_ok=True)
"""),
    md("## 4. Kurulum ve testler\nColab'ın kendi PyTorch'u kullanılır; yalnızca eksik paketler kurulur (~1 dk)."),
    code("""
!pip install -q -r requirements.txt
!python -m pytest -q tests
"""),
    md("""
## 5. (İsteğe bağlı) Duman testi
Birkaç dakikada uçtan uca çalışır. Yeni bir GPU türünde ilk kez çalıştırıyorsan önerilir.
Çıktıları `/content` altına yazar, Drive'a değil.
"""),
    code("""
!MC_OUTPUT_DIR=/content/outputs_smoke python -m src.pipeline --config configs/smoke.yaml \\
    --set selection.method=dual --set experiment.line=smoke_dual
"""),
    md("""
## 6. Eğitim
Uzun sürer. Oturum koparsa 3. hücreden itibaren yeniden çalıştır: tamamlanan nesiller atlanır.
İlerleme Drive'da `outputs/<hat>/run.log` dosyasına da yazılır.
"""),
    code("""
import shlex
gen_batch = AUTO_GEN_BATCH if GEN_BATCH == "auto" else GEN_BATCH
overrides = [f"data.data_dir={DATA_DIR}", f"generate.batch_size={gen_batch}", *EXTRA]  # EXTRA wins
sets = " ".join(f"--set {shlex.quote(s)}" for s in overrides)
seeds = " ".join(map(str, SEEDS))
for line in LINES:
    print(f"===== {line} | seeds {seeds} =====")
    !python -m src.pipeline --config configs/lines/{line}.yaml --seeds {seeds} {sets}
"""),
    md("## 7. Rapor\nTüm hatların `metrics.json` dosyalarını toplar; tablo ve grafikler Drive'daki `results/` klasörüne yazılır."),
    code("""
!python -m src.metrics.report --results {os.environ['MC_OUTPUT_DIR']} --out {DRIVE_ROOT}/results

import pandas as pd
from IPython.display import Image, display
summary = pd.read_csv(f"{DRIVE_ROOT}/results/summary.csv")
cols = ["test_ppl", "rare_recall", "vendi", "distinct_2", "js_real_gen", "frechet_real", "rep3_mean"]
display(summary[["line", "generation", "n_seeds", *[f"{c}_mean" for c in cols]]].round(4))
# Rising vendi/distinct alone is not "healthy diversity": read it together with js_real_gen and
# frechet_real (distance to real data) and rare_recall (tail preservation).
for metric in ["test_ppl", "rare_recall", "vendi", "js_real_gen", "frechet_real"]:
    display(Image(f"{DRIVE_ROOT}/results/plots/{metric}.png"))
"""),
    md("""
## 8. Süre ve kaynak kontrolü
Her neslin süreleri `metrics.json` içindeki `time_*` alanlarındadır. İlk nesilden sonra buna bakıp
oturum başına kaç nesil sığacağını hesapla.
"""),
    code("""
import json, glob
for path in sorted(glob.glob(f"{os.environ['MC_OUTPUT_DIR']}/**/metrics.json", recursive=True)):
    m = json.load(open(path))
    times = {k: v for k, v in m.items() if k.startswith("time_")}
    print(path.split("outputs/")[-1], m.get("gpu"), m.get("precision"), f"{sum(times.values())/60:.1f} dk", times)
"""),
]


def main() -> None:
    nb = {
        "nbformat": 4, "nbformat_minor": 0,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        "cells": CELLS,
    }
    out = ROOT / "notebooks" / "colab_train.ipynb"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
