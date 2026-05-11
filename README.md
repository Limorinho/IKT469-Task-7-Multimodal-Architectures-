# Multimodal Architectures on MM-IMDB

Semester project for **IKT469 — Deep Neural Networks** (UiA), Option 7:
multi-label genre classification on MM-IMDB combining text (plot outlines) and
images (movie posters). Covers unimodal baselines, early / late / GMU fusion,
semi-supervised learning (pseudo-labels, Mean Teacher), and cross-modal
self-supervised pretraining (InfoNCE).

## Prerequisites

- **Python 3.13** (pinned in `.python-version`)
- **[uv](https://docs.astral.sh/uv/)** for environment + dependency management
- ~16 GB free disk for the full MM-IMDB HDF5; a CUDA-capable GPU is recommended
  but not required (everything works on CPU since features are pre-extracted)

Install uv if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Setup

```bash
git clone <repo-url>
cd project
uv sync                # creates .venv and installs deps from pyproject.toml
```

## Dataset

Two dataset variants are used:

### 1. Full MM-IMDB (HDF5) — used by `main.py`

Expected at `dataset/multimodal_imdb.hdf5` (~16 GB). The file ships with
pre-extracted features so no encoders need to run:

- `features`        — 300-d averaged word2vec plot embeddings
- `vgg_features`    — 4096-d VGG19 poster embeddings
- `genres`          — 23-class multi-hot labels
- `attrs["split"]`  — train / dev / test row ranges

It originates from the [MM-IMDb release by Arevalo et al.
(2017)](https://github.com/johnarevalo/gmu-mmimdb). Place the file at:

```
dataset/multimodal_imdb.hdf5
```

Verify the layout once placed:

```bash
uv run python main.py inspect-data
```

This prints every dataset key with shape/dtype and the train-split genre
distribution.

### 2. Tiny MM-IMDB (raw images + CSV) — for smoke tests

A small Kaggle mirror containing posters + plot text, used by the alternate
loader at `src/data/loader.py` (BERT tokenizer + VGG-style transforms). Fetch
with:

```bash
uv run python -m src.data.get
```

It lands at `dataset/tiny-mm-imdb/tinymmimdb/{data.csv,images/}`. This path is
**not** used by `main.py` — it's there for the test suite and for any
end-to-end pipeline experiments.

## Running experiments

Every experiment is a `main.py` subcommand. Results (history + final test
metrics) are dumped to `results/<experiment>.json`.

### Baselines and fusion

```bash
uv run python main.py text-only             # text features → MLP
uv run python main.py image-only            # image features → MLP
uv run python main.py fusion-early          # concat → MLP
uv run python main.py fusion-late           # per-modality logits, prob average
uv run python main.py fusion-gmu            # Gated Multimodal Unit
```

### Semi-supervised (defaults to 20% labels)

```bash
uv run python main.py semi-baseline         # supervised on labeled subset
uv run python main.py semi-pseudo           # + pseudo-labeling
uv run python main.py semi-meanteacher      # + Mean Teacher consistency
```

Tune with `--pseudo-threshold`, `--consistency-weight`, `--ema-alpha`.

### Self-supervised (cross-modal contrastive)

```bash
uv run python main.py selfsup-contrastive   # InfoNCE pretrain + linear probe
```

Tune temperature with `--temperature`.

### Common flags (all subcommands)

| Flag                | Default | Notes                                       |
|---------------------|---------|---------------------------------------------|
| `--epochs`          | 20      |                                             |
| `--batch-size`      | 128     |                                             |
| `--lr`              | 1e-3    | Adam, weight decay 1e-5                     |
| `--device`          | `auto`  | `auto` / `cpu` / `cuda`                     |
| `--seed`            | 0       |                                             |
| `--label-fraction`  | 1.0     | semi-* defaults to 0.2                      |

### Experimenting tips

- Sweep label fractions: `for f in 0.05 0.10 0.20 0.50; do uv run python main.py semi-pseudo --label-fraction $f; done`
- Override the run name by reading the printed `Results written to ...` line.
- Hack on architectures in `src/fusion/`, then re-run a subcommand — the
  classifier wiring is in `src/combiner/classifier.py`.
- Plotting and ad-hoc analysis live in `experiments.ipynb`.

## Project layout

```
src/
  data/        HDF5 dataset, tiny CSV loader, kagglehub download script
  fusion/      unimodal, early (concat+MLP), late (prob-avg), gmu
  combiner/    MultimodalClassifier (fusion → head)
  ssl/         pseudo-labeling, Mean Teacher, InfoNCE + projection head
  training/    supervised + semi-supervised loops, evaluate
tests/         pytest tests per module
docs/          Typst report (unified-uia-thesis template)
results/       per-experiment JSON metric dumps
experiments.ipynb  analysis + plots
main.py        CLI entrypoint for every experiment
```

## Tests

```bash
uv run pytest -v
```

## Report

Built with Typst. From `docs/`:

```bash
typst compile main.typ
```

Live editing also available at:
[Typst project](https://typst.app/project/ra2kU2K5bMHrj9o9C3bu8W).

## Course

IKT469 — Deep Neural Networks, University of Agder. Group: Erlend Tregde,
Sander Wesstøl, Linor Ujkani. Supervisor: Morten Goodwin.
