# LAFA and MgHiSal reproduction code

Notebooks used for the course report on LAFA (AAAI 2024) and MgHiSal
(Knowledge-Based Systems 2025). This file only explains how to run them.
Results and method descriptions are in the report.

Neither paper has usable official code, so everything here is an independent
implementation.

## Disclaimer

The code in this repository was written with the help of AI tools.
The author directed the work, ran the experiments and checked the results.

## Environment used

| Item | Value |
|---|---|
| GPU | 1 x NVIDIA RTX PRO 6000 Blackwell Server Edition |
| CPU / RAM | 20 cores / 160 GB |
| OS | Linux (cloud runtime) |
| Python | 3.13.11 |
| PyTorch | 2.11.0 (CUDA 13.0 runtime) |
| torchvision | 0.26.0 |
| marimo | 0.25.2 |

A CUDA GPU is required for training and evaluation.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lafa.txt      # LAFA
# pip install -r requirements-mghisal.txt  # MgHiSal (use a separate venv)
cp .env.example .env      # then set WANDB_API_KEY
set -a; source .env; set +a
```

## Run

Each notebook is a [marimo](https://marimo.io) file. Open it, then run the
cells from top to bottom. Feature extraction, training and evaluation each start from their own
button, so a long run is never triggered by editing a cell. MgHiSal has separate
train and evaluate buttons for DB15K and MKG-W.

```bash
marimo edit --no-sandbox LAFA.py
marimo edit --no-sandbox MgHiSal.py
```

| Method | Notebook | W&B project |
|---|---|---|
| LAFA | `LAFA.py` | `lafa-reproduction` |
| MgHiSal | `MgHiSal.py` | `MgHiSal-reproduction` |

Every run logs its metrics and artifacts to the W&B project of its method.

## Data

Datasets are not stored in this repository.

**LAFA.** The notebook downloads the
FB15k-237 and WN18 triples from
[DeepGraphLearning/KnowledgeGraphEmbedding](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding/tree/master/data) and reads the entity image archive
`fb15k_wn18_images.zip` from Google Drive. Set `DRIVE_ROOT` near the top of the notebook to
the Drive folder that holds the archive. The first Drive access asks for a
Google login in the cell output.

**MgHiSal.** The notebook reads the DB15K and MKG-W folders from Google Drive
under `DRIVE_ROOT` (near the top of the notebook). The two source folders are:

- DB15K: <https://drive.google.com/drive/folders/1eWW4QquPkZU2hEVONkJfLGpsBY_IpwR7>
- MKG-W: <https://drive.google.com/drive/folders/1MoEJmhkwqD_CPAlfkD012pP4pBOH2Wrk>

Copy both folders to your Drive and adjust `DRIVE_ROOT`.
Entity captions come from the Hugging Face dataset
`pengyu3/beyond-images-enriched`, and text features from
`sentence-transformers/all-MiniLM-L6-v2`.
