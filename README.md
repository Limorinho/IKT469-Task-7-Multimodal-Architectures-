# Multimodal Architectures on MM-IMDB

Semester project for IKT469 (Deep Neural Networks, UiA). Option 7 — multimodal
classification on the MM-IMDB dataset.

## Setup

```bash
uv sync                # install deps from pyproject.toml
```

The dataset `dataset/multimodal_imdb.hdf5` (~16 GB) must already be present.
It ships with pre-extracted 300-d averaged word2vec text features
(`features`) and 4096-d VGG19 image features (`vgg_features`) — both are used
directly as the multimodal embeddings.

## Running experiments

Every experiment is a subcommand of `main.py`. Results are written to
`results/<experiment>.json`.

```bash
uv run python main.py inspect-data          # print HDF5 schema + label counts
uv run python main.py text-only             # 1. text-only baseline
uv run python main.py image-only            # 2. image-only baseline
uv run python main.py fusion-early          # 3. concat + MLP
uv run python main.py fusion-late           # 4. per-modality, prob avg
uv run python main.py fusion-gmu            # 5. Gated Multimodal Unit
uv run python main.py semi-baseline         # 6. supervised at 20% labels
uv run python main.py semi-pseudo           # 7. + pseudo-labeling
uv run python main.py semi-meanteacher      # 8. + Mean Teacher
uv run python main.py selfsup-contrastive   # 9. cross-modal InfoNCE + linear probe
```

Common flags for every subcommand: `--epochs`, `--batch-size`, `--lr`,
`--device {auto,cpu,cuda}`, `--seed`, `--label-fraction`.

## Project layout

```
src/
  data/        HDF5 dataset, tiny CSV loader (smoke tests)
  fusion/      unimodal, early (concat+MLP), late (prob-avg), gmu
  combiner/    MultimodalClassifier
  ssl/         pseudo-labeling, Mean Teacher, InfoNCE
  training/    supervised + semi-supervised loops, evaluate
tests/         pytest tests per module
docs/          Typst report (uses unified-uia-thesis template)
results/       per-experiment JSON metric dumps
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
