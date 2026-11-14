# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "gdown==6.5.1",
#     "wandb==0.31.0",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    import os
    from pathlib import Path
    import marimo as mo
    import numpy as np
    import pandas as pd
    import torch
    import json
    import wandb

    ROOT = Path.cwd() / "mghisal"
    ROOT.mkdir(parents=True, exist_ok=True)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    return DEVICE, ROOT, json, mo, os, torch, wandb


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Setup and Dataset Sources
    """)
    return


@app.cell
def _(mo):
    from gdrive_fsspec import GoogleDriveFileSystem

    fs = GoogleDriveFileSystem(use_listings_cache=False, skip_instance_cache=True, auth_kwargs={"use_local_webserver": False})
    return (fs,)


@app.cell
def _():
    DRIVE_ROOT = "Master/Graph/Mid/MgHiSal/datasets"

    DATASETS = {
        "MKG-W": f"{DRIVE_ROOT}/MKG-W",
        "MMKB-DB15K": f"{DRIVE_ROOT}/MMKB-DB15K",
    }

    return DATASETS, DRIVE_ROOT


@app.cell
def _(DATASETS, fs):
    for _name, _path in DATASETS.items():

        for _file in fs.ls(_path, detail=False):
            pass
    return


@app.cell
def _(os, wandb):
    wandb.login(key=os.environ["WANDB_API_KEY"])

    api = wandb.Api()

    WANDB_ENTITY = api.default_entity
    WANDB_PROJECT = "mghisal-reproduction"

    return (WANDB_PROJECT,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Preprocessing MKG-DB15K
    """)
    return


@app.cell
def _(DRIVE_ROOT, ROOT, fs):
    DB15K_SOURCE = f"{DRIVE_ROOT}/MMKB-DB15K"

    DB15K_OUTPUT = ROOT / "processed" / "DB15K"
    DB15K_OUTPUT.mkdir(parents=True, exist_ok=True)

    return DB15K_OUTPUT, DB15K_SOURCE


@app.cell
def _(DB15K_SOURCE, fs):
    def read_triples(source, split_name):
        _path = f"{source}/{split_name}.txt"

        with fs.open(_path, "rt", encoding="utf-8") as _file:
            return [
                tuple(_line.strip().split("\t"))
                for _line in _file
                if _line.strip()
            ]


    DB15K_RAW_SPLITS = {
        _split: read_triples(DB15K_SOURCE, _split)
        for _split in ("train", "valid", "test")
    }

    for _split, _triples in DB15K_RAW_SPLITS.items():
        pass
    return DB15K_RAW_SPLITS, read_triples


@app.cell
def _(DB15K_RAW_SPLITS):
    DB15K_ENTITIES = sorted({
        _entity
        for _triples in DB15K_RAW_SPLITS.values()
        for _h, _r, _t in _triples
        for _entity in (_h, _t)
    })

    DB15K_RELATIONS = sorted({
        _r
        for _triples in DB15K_RAW_SPLITS.values()
        for _h, _r, _t in _triples
    })

    DB15K_ENTITY2ID = {
        _entity: _idx
        for _idx, _entity in enumerate(DB15K_ENTITIES)
    }

    DB15K_RELATION2ID = {
        _relation: _idx
        for _idx, _relation in enumerate(DB15K_RELATIONS)
    }

    return DB15K_ENTITY2ID, DB15K_RELATION2ID


@app.cell
def _(
    DB15K_ENTITY2ID,
    DB15K_OUTPUT,
    DB15K_RAW_SPLITS,
    DB15K_RELATION2ID,
    json,
):
    DB15K_NUMERIC_SPLITS = {
        _split: [
            (
                DB15K_ENTITY2ID[_h],
                DB15K_RELATION2ID[_r],
                DB15K_ENTITY2ID[_t],
            )
            for _h, _r, _t in _triples
        ]
        for _split, _triples in DB15K_RAW_SPLITS.items()
    }

    for _split, _triples in DB15K_NUMERIC_SPLITS.items():
        with open(
            DB15K_OUTPUT / f"{_split}2id.txt",
            "w",
            encoding="utf-8",
        ) as _file:
            for _h, _r, _t in _triples:
                _file.write(f"{_h}\t{_r}\t{_t}\n")

    for _filename, _mapping in {
        "entity2id.json": DB15K_ENTITY2ID,
        "relation2id.json": DB15K_RELATION2ID,
    }.items():
        with open(
            DB15K_OUTPUT / _filename,
            "w",
            encoding="utf-8",
        ) as _file:
            json.dump(_mapping, _file, ensure_ascii=False, indent=2)

    return (DB15K_NUMERIC_SPLITS,)


@app.cell
def _(DB15K_ENTITY2ID, DB15K_OUTPUT, DB15K_SOURCE, fs, json):
    DB15K_DESCRIPTIONS = {}

    with fs.open(
        f"{DB15K_SOURCE}/DB15K_description.txt",
        "rt",
        encoding="utf-8",
    ) as _file:
        for _line in _file:
            _parts = _line.rstrip("\n").split("\t", 1)

            if len(_parts) == 2:
                _entity = _parts[0].strip().strip("<>")
                _description = _parts[1].strip()

                if _entity in DB15K_ENTITY2ID:
                    DB15K_DESCRIPTIONS[
                        str(DB15K_ENTITY2ID[_entity])
                    ] = _description

    DB15K_MISSING_DESCRIPTIONS = [
        _idx
        for _idx in DB15K_ENTITY2ID.values()
        if str(_idx) not in DB15K_DESCRIPTIONS
    ]

    for _filename, _content in {
        "entity_descriptions.json": DB15K_DESCRIPTIONS,
        "missing_descriptions.json": DB15K_MISSING_DESCRIPTIONS,
    }.items():
        with open(
            DB15K_OUTPUT / _filename,
            "w",
            encoding="utf-8",
        ) as _file:
            json.dump(_content, _file, ensure_ascii=False)

    return (DB15K_DESCRIPTIONS,)


@app.cell
def _(DB15K_OUTPUT, DB15K_SOURCE, fs, json):
    DB15K_SAMEAS_LINKS = {}

    with fs.open(
        f"{DB15K_SOURCE}/DB15K_SameAsLink.txt",
        "rt",
        encoding="utf-8",
    ) as _file:
        for _line in _file:
            _parts = _line.strip().split()

            if len(_parts) >= 3:
                _freebase = _parts[0].strip("<>")
                _dbpedia = _parts[2].strip("<>")

                DB15K_SAMEAS_LINKS[_freebase] = _dbpedia

    with open(
        DB15K_OUTPUT / "sameas_links.json",
        "w",
        encoding="utf-8",
    ) as _file:
        json.dump(DB15K_SAMEAS_LINKS, _file, ensure_ascii=False)

    return (DB15K_SAMEAS_LINKS,)


@app.cell
def _(
    DB15K_DESCRIPTIONS,
    DB15K_ENTITY2ID,
    DB15K_NUMERIC_SPLITS,
    DB15K_OUTPUT,
    DB15K_RELATION2ID,
    DB15K_SAMEAS_LINKS,
    DB15K_SOURCE,
    json,
):
    DB15K_METADATA = {
        "dataset": "MMKB_DB15K",
        "source_path": DB15K_SOURCE,
        "source_url": "https://drive.google.com/drive/folders/1eWW4QquPkZU2hEVONkJfLGpsBY_IpwR7",
        "triple_order": "hrt",
        "num_entities": len(DB15K_ENTITY2ID),
        "num_relations": len(DB15K_RELATION2ID),
        "splits": {
            _name: len(_rows)
            for _name, _rows in DB15K_NUMERIC_SPLITS.items()
        },
        "num_descriptions": len(DB15K_DESCRIPTIONS),
        "num_sameas_links": len(DB15K_SAMEAS_LINKS),
        "split_policy": "preserve_original",
    }

    with open(
        DB15K_OUTPUT / "metadata.json",
        "w",
        encoding="utf-8",
    ) as _file:
        json.dump(DB15K_METADATA, _file, indent=2)

    return (DB15K_METADATA,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Preprocessing MKG-W
    """)
    return


@app.cell
def _(DRIVE_ROOT, ROOT):
    W_SOURCE = f"{DRIVE_ROOT}/MKG-W"

    W_OUTPUT = ROOT / "processed" / "MKG-W"
    W_OUTPUT.mkdir(parents=True, exist_ok=True)

    return W_OUTPUT, W_SOURCE


@app.cell
def _(W_SOURCE, read_triples):
    W_RAW_SPLITS = {
        _split: read_triples(W_SOURCE, _split)
        for _split in ("train", "valid", "test")
    }

    for _split, _rows in W_RAW_SPLITS.items():
        pass
    return (W_RAW_SPLITS,)


@app.cell
def _(W_RAW_SPLITS):
    W_ENTITIES = sorted({
        _entity
        for _triples in W_RAW_SPLITS.values()
        for _h, _r, _t in _triples
        for _entity in (_h, _t)
    })

    W_RELATIONS = sorted({
        _r
        for _triples in W_RAW_SPLITS.values()
        for _h, _r, _t in _triples
    })

    W_ENTITY2ID = {
        _entity: _idx
        for _idx, _entity in enumerate(W_ENTITIES)
    }

    W_RELATION2ID = {
        _relation: _idx
        for _idx, _relation in enumerate(W_RELATIONS)
    }

    return W_ENTITY2ID, W_RELATION2ID


@app.cell
def _(W_ENTITY2ID, W_OUTPUT, W_RAW_SPLITS, W_RELATION2ID, json):
    W_NUMERIC_SPLITS = {
        _split: [
            (
                W_ENTITY2ID[_h],
                W_RELATION2ID[_r],
                W_ENTITY2ID[_t],
            )
            for _h, _r, _t in _triples
        ]
        for _split, _triples in W_RAW_SPLITS.items()
    }

    for _split, _triples in W_NUMERIC_SPLITS.items():
        with open(
            W_OUTPUT / f"{_split}2id.txt",
            "w",
            encoding="utf-8",
        ) as _file:
            for _h, _r, _t in _triples:
                _file.write(f"{_h}\t{_r}\t{_t}\n")

    for _filename, _mapping in {
        "entity2id.json": W_ENTITY2ID,
        "relation2id.json": W_RELATION2ID,
    }.items():
        with open(
            W_OUTPUT / _filename,
            "w",
            encoding="utf-8",
        ) as _file:
            json.dump(
                _mapping,
                _file,
                ensure_ascii=False,
                indent=2,
            )

    return (W_NUMERIC_SPLITS,)


@app.cell
def _(W_ENTITY2ID, W_OUTPUT, W_SOURCE, fs, json):
    W_ENT_LINKS = []

    with fs.open(
        f"{W_SOURCE}/ent_links",
        "rt",
        encoding="utf-8",
    ) as _file:
        for _line in _file:
            _parts = _line.rstrip("\n").split("\t", 1)

            if len(_parts) == 2:
                W_ENT_LINKS.append(
                    (_parts[0].strip(), _parts[1].strip())
                )

    W_LINKED_IDS = {
        _dbpedia: W_ENTITY2ID[_wikidata]
        for _dbpedia, _wikidata in W_ENT_LINKS
        if _wikidata in W_ENTITY2ID
    }

    with open(
        W_OUTPUT / "ent_links.tsv",
        "w",
        encoding="utf-8",
    ) as _file:
        for _dbpedia, _wikidata in W_ENT_LINKS:
            _file.write(f"{_dbpedia}\t{_wikidata}\n")

    with open(
        W_OUTPUT / "dbpedia_to_wikidata_id.json",
        "w",
        encoding="utf-8",
    ) as _file:
        json.dump(
            W_LINKED_IDS,
            _file,
            ensure_ascii=False,
            indent=2,
        )

    return W_ENT_LINKS, W_LINKED_IDS


@app.cell
def _(
    W_ENTITY2ID,
    W_ENT_LINKS,
    W_LINKED_IDS,
    W_NUMERIC_SPLITS,
    W_OUTPUT,
    W_RELATION2ID,
    W_SOURCE,
    json,
):
    W_METADATA = {
        "dataset": "MKG_W",
        "source_path": W_SOURCE,
        "source_url": (
            "https://drive.google.com/drive/folders/"
            "1MoEJmhkwqD_CPAlfkD012pP4pBOH2Wrk"
        ),
        "triple_order": "hrt",
        "num_entities": len(W_ENTITY2ID),
        "num_relations": len(W_RELATION2ID),
        "splits": {
            _name: len(_rows)
            for _name, _rows in W_NUMERIC_SPLITS.items()
        },
        "num_entity_links": len(W_ENT_LINKS),
        "num_mapped_links": len(W_LINKED_IDS),
        "split_policy": "preserve_original",
        "mapping_policy": "sorted_lexicographically",
        "preprocessing_version": "1.0",
    }

    with open(
        W_OUTPUT / "metadata.json",
        "w",
        encoding="utf-8",
    ) as _file:
        json.dump(
            W_METADATA,
            _file,
            indent=2,
        )

    return (W_METADATA,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Upload Processed Datasets to W&B
    """)
    return


@app.cell
def _(DB15K_METADATA, DB15K_OUTPUT, W_METADATA, W_OUTPUT):
    ARTIFACTS = {
        "db15k-processed": {
            "path": DB15K_OUTPUT,
            "metadata": DB15K_METADATA,
        },
        "mkg-w-processed": {
            "path": W_OUTPUT,
            "metadata": W_METADATA,
        },
    }

    ARTIFACT_REFS = {}
    return ARTIFACTS, ARTIFACT_REFS


@app.cell
def _(ARTIFACTS, ARTIFACT_REFS, WANDB_PROJECT, wandb):
    with wandb.init(
        project=WANDB_PROJECT,
        job_type="preprocessing",
        name="mghisal-preprocessing-from-raw",
        config={
            "pipeline_version": "1.0",
            "source": "google_drive",
            "preprocess_from_scratch": True,
            "split_policy": "preserve_original",
        },
    ) as _run:

        for _name, _config in ARTIFACTS.items():
            _artifact = wandb.Artifact(
                name=_name,
                type="dataset",
                description="MgHiSal dataset preprocessed from raw data",
                metadata=_config["metadata"],
            )

            _artifact.add_dir(
                local_path=str(_config["path"])
            )

            _logged = _run.log_artifact(
                _artifact,
                aliases=["latest", "raw-preprocessed"],
            )

            _logged.wait()

            ARTIFACT_REFS[_name] = (
                f"{_run.entity}/{_run.project}/{_name}:latest"
            )

    return


@app.cell
def _(ARTIFACT_REFS, ROOT, json):
    REFS_PATH = ROOT / "processed" / "artifact_refs.json"

    with open(
        REFS_PATH,
        "w",
        encoding="utf-8",
    ) as _file:
        json.dump(
            ARTIFACT_REFS,
            _file,
            indent=2,
        )

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. MgHiSal Model Setup
    """)
    return


@app.cell
def _(DB15K_OUTPUT, W_OUTPUT, json, torch):
    graph_data = {}

    for _name, _path in {
        "DB15K": DB15K_OUTPUT,
        "MKG_W": W_OUTPUT,
    }.items():
        with open(_path / "entity2id.json") as _file:
            _entities = json.load(_file)

        with open(_path / "relation2id.json") as _file:
            _relations = json.load(_file)

        _splits = {}

        for _split in ("train", "valid", "test"):
            with open(
                _path / f"{_split}2id.txt",
                encoding="utf-8",
            ) as _file:
                _triples = [
                    [int(_v) for _v in _line.split()]
                    for _line in _file
                    if _line.strip()
                ]

            _splits[_split] = torch.tensor(
                _triples,
                dtype=torch.long,
            ).reshape(-1, 3)

        graph_data[_name] = {
            "num_entities": len(_entities),
            "num_relations": len(_relations),
            **_splits,
        }

    return (graph_data,)


@app.cell
def _(torch):
    import torch.nn as nn


    class MgHiSal(nn.Module):
        def __init__(
            self,
            num_entities,
            num_relations,
            dim=256,
            heads=4,
            dropout=0.1,
        ):
            super().__init__()

            self.entities = nn.Embedding(num_entities, dim)
            self.relations = nn.Embedding(num_relations, dim)

            # Modality projections
            self.text_proj = nn.LazyLinear(dim)
            self.visual_text_proj = nn.LazyLinear(dim)

            # Level 1: text-visual semantic alignment
            self.cross_attention = nn.MultiheadAttention(
                dim,
                heads,
                dropout=dropout,
                batch_first=True,
            )

            self.local_gate = nn.Linear(dim * 3, dim)
            self.norm_local = nn.LayerNorm(dim)

            # Level 2: structural-semantic alignment
            self.hier_attention = nn.MultiheadAttention(
                dim,
                heads,
                dropout=dropout,
                batch_first=True,
            )

            self.global_gate = nn.Linear(dim * 3, dim)
            self.norm_global = nn.LayerNorm(dim)

            # Neighbor-aware attention
            self.neighbor_query = nn.Linear(
                dim, dim, bias=False
            )
            self.neighbor_key = nn.Linear(
                dim, dim, bias=False
            )
            self.neighbor_value = nn.Linear(
                dim, dim, bias=False
            )

            self.neighbor_gate = nn.Linear(dim * 2, dim)
            self.norm_neighbor = nn.LayerNorm(dim)

            self.dropout = nn.Dropout(dropout)

            nn.init.xavier_uniform_(self.entities.weight)
            nn.init.xavier_uniform_(self.relations.weight)

        def forward(
            self,
            text_features,
            visual_text_features,
            edge_index,
            visual_mask=None,
            return_aux=False,
        ):
            _struct = self.entities.weight
            _text = self.text_proj(text_features)
            _visual = self.visual_text_proj(
                visual_text_features
            )

            # Missing visual descriptions
            if visual_mask is None:
                _mask = (
                    visual_text_features.abs().sum(dim=-1) > 0
                )
            else:
                _mask = visual_mask.bool()

            _visual = _visual.masked_fill(
                ~_mask.unsqueeze(1),
                0.0,
            )

            # Level 1: cross-modal attention
            _tokens = torch.stack(
                [_text, _visual],
                dim=1,
            )

            _key_mask = torch.stack(
                [
                    torch.zeros_like(_mask),
                    ~_mask,
                ],
                dim=1,
            )

            _attended, _ = self.cross_attention(
                _text.unsqueeze(1),
                _tokens,
                _tokens,
                key_padding_mask=_key_mask,
                need_weights=False,
            )

            _attended = _attended.squeeze(1)

            _local_gate = torch.sigmoid(
                self.local_gate(
                    torch.cat(
                        [_text, _visual, _attended],
                        dim=-1,
                    )
                )
            )

            _local_gate = (
                _local_gate * _mask.unsqueeze(1)
            )

            _local = self.norm_local(
                _text
                + _local_gate * (_attended - _text)
            )

            # Level 2: hierarchical alignment
            _hier_tokens = torch.stack(
                [_struct, _local, _visual],
                dim=1,
            )

            _hier_mask = torch.stack(
                [
                    torch.zeros_like(_mask),
                    torch.zeros_like(_mask),
                    ~_mask,
                ],
                dim=1,
            )

            _aligned, _ = self.hier_attention(
                _struct.unsqueeze(1),
                _hier_tokens,
                _hier_tokens,
                key_padding_mask=_hier_mask,
                need_weights=False,
            )

            _aligned = _aligned.squeeze(1)

            _global_gate = torch.sigmoid(
                self.global_gate(
                    torch.cat(
                        [_struct, _local, _aligned],
                        dim=-1,
                    )
                )
            )

            _target = 0.5 * (_local + _aligned)

            _fused = self.norm_global(
                _struct
                + _global_gate * (_target - _struct)
            )

            # Neighbor-aware attention
            _src, _dst = edge_index

            _query = self.neighbor_query(_fused)
            _key = self.neighbor_key(_fused)
            _value = self.neighbor_value(_fused)

            _logits = (
                _query[_dst] * _key[_src]
            ).sum(dim=-1) / (_fused.shape[1] ** 0.5)

            # Stable per-destination softmax
            _max = torch.full(
                (_fused.shape[0],),
                -torch.inf,
                device=_fused.device,
                dtype=_fused.dtype,
            )

            _max.scatter_reduce_(
                0,
                _dst,
                _logits.detach(),
                reduce="amax",
                include_self=True,
            )

            _weights = torch.exp(
                (_logits - _max[_dst]).clamp(
                    min=-30,
                    max=0,
                )
            )

            _denom = torch.zeros_like(_max)

            _denom.index_add_(
                0,
                _dst,
                _weights,
            )

            _denom = _denom.clamp_min(1e-8)

            _attention = (
                _weights / _denom[_dst]
            )

            _aggregated = torch.zeros_like(_value)

            _aggregated.index_add_(
                0,
                _dst,
                _value[_src]
                * _attention.unsqueeze(1),
            )

            _neighbor_gate = torch.sigmoid(
                self.neighbor_gate(
                    torch.cat(
                        [_fused, _aggregated],
                        dim=-1,
                    )
                )
            )

            _output = self.norm_neighbor(
                _fused
                + self.dropout(
                    _neighbor_gate * _aggregated
                )
            )

            if not return_aux:
                return _output

            # Gate regularization for training
            _gate_reg = (
                (_local_gate * (1 - _local_gate)).mean()
                + (_global_gate * (1 - _global_gate)).mean()
            )

            return _output, {
                "text": _text,
                "visual": _visual,
                "local": _local,
                "fused": _fused,
                "visual_mask": _mask,
                "gate_reg": _gate_reg,
            }

        # Reciprocal ComplEx scoring
        def score(self, entity_embeddings, triples):
            _h, _r, _t = triples.unbind(dim=1)

            _hr, _hi = entity_embeddings[_h].chunk(
                2, dim=-1
            )
            _rr, _ri = self.relations(_r).chunk(
                2, dim=-1
            )
            _tr, _ti = entity_embeddings[_t].chunk(
                2, dim=-1
            )

            return (
                (_hr * _rr - _hi * _ri) * _tr
                + (_hr * _ri + _hi * _rr) * _ti
            ).sum(dim=-1)

        def score_candidates(
            self,
            entity_embeddings,
            triples,
            candidate_ids,
            direction="tail",
        ):
            _h, _r, _t = triples.unbind(dim=1)
            _rr, _ri = self.relations(_r).chunk(
                2, dim=-1
            )

            if direction == "tail":
                _hr, _hi = entity_embeddings[_h].chunk(
                    2, dim=-1
                )

                _qr = _hr * _rr - _hi * _ri
                _qi = _hr * _ri + _hi * _rr

            elif direction == "head":
                _tr, _ti = entity_embeddings[_t].chunk(
                    2, dim=-1
                )

                _qr = _rr * _tr + _ri * _ti
                _qi = _rr * _ti - _ri * _tr

            else:
                raise ValueError(direction)

            _er, _ei = entity_embeddings[
                candidate_ids
            ].chunk(2, dim=-1)

            if candidate_ids.ndim == 1:
                return (
                    _qr @ _er.T
                    + _qi @ _ei.T
                )

            return (
                _qr.unsqueeze(1) * _er
                + _qi.unsqueeze(1) * _ei
            ).sum(dim=-1)

    return (MgHiSal,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Multimodal Feature Preparation
    """)
    return


@app.cell
def _(DB15K_OUTPUT, ROOT, W_OUTPUT):
    import re
    import hashlib
    from urllib.parse import unquote
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer

    feature_sources = {
        "DB15K": DB15K_OUTPUT,
        "MKG_W": W_OUTPUT,
    }

    features_root = ROOT / "features"
    features_root.mkdir(parents=True, exist_ok=True)

    caption_repo = "pengyu3/beyond-images-enriched"
    encoder_name = "sentence-transformers/all-MiniLM-L6-v2"

    return (
        SentenceTransformer,
        caption_repo,
        encoder_name,
        feature_sources,
        hashlib,
        load_dataset,
        re,
        unquote,
    )


@app.cell
def _(json, load_dataset, re, unquote):
    def cache_visual_captions(sources, repo):
        _pending = {
            _name: _path
            for _name, _path in sources.items()
            if not (_path / "image_captions.json").exists()
        }

        if not _pending:
            return

        _rows = load_dataset(
            repo,
            "entities",
            split="train",
        )

        def _keys(value):
            _value = unquote(
                str(value or "").strip().strip("<>")
            ).rstrip("/")

            if not _value:
                return []

            _leaf = (
                _value.rsplit("/", 1)[-1]
                .rsplit("#", 1)[-1]
            )

            _result = [
                "raw:" + _value.casefold()
            ]

            if re.fullmatch(r"Q\d+", _leaf, re.I):
                _result.append(
                    "qid:" + _leaf.upper()
                )

            if "dbpedia.org/resource/" in _value.lower():
                _result.append(
                    "db:" + _leaf.casefold()
                )

            _result.append(
                "name:" + _leaf.casefold()
            )

            return _result

        def _register(lookup, key, idx):
            if key not in lookup:
                lookup[key] = idx
            elif lookup[key] != idx:
                lookup[key] = None

        _lookups = {}
        _captions = {}
        _sources = {}
        _sizes = {}

        for _name, _path in _pending.items():
            with open(
                _path / "entity2id.json",
                encoding="utf-8",
            ) as _file:
                _entity2id = json.load(_file)

            _lookup = {}

            for _entity, _idx in _entity2id.items():
                for _key in _keys(_entity):
                    _register(_lookup, _key, int(_idx))

            if _name == "DB15K":
                _link_path = _path / "sameas_links.json"

                if _link_path.exists():
                    with open(
                        _link_path,
                        encoding="utf-8",
                    ) as _file:
                        _links = json.load(_file)

                    for _freebase, _dbpedia in _links.items():
                        _idx = next(
                            (
                                _lookup[_key]
                                for _key in _keys(_freebase)
                                if _lookup.get(_key) is not None
                            ),
                            None,
                        )

                        if _idx is not None:
                            for _key in _keys(_dbpedia):
                                _register(_lookup, _key, _idx)

            else:
                _link_path = (
                    _path / "dbpedia_to_wikidata_id.json"
                )

                if _link_path.exists():
                    with open(
                        _link_path,
                        encoding="utf-8",
                    ) as _file:
                        _links = json.load(_file)

                    for _dbpedia, _idx in _links.items():
                        for _key in _keys(_dbpedia):
                            _register(
                                _lookup, _key, int(_idx)
                            )

            _lookups[_name] = _lookup
            _captions[_name] = {}
            _sources[_name] = {}
            _sizes[_name] = len(_entity2id)

        _priority = {
            "original": 2,
            "new": 1,
        }

        for _row in _rows:
            _name = (
                "MKG_W"
                if _row["dataset"] == "MKG-W"
                else _row["dataset"]
            )

            if _name not in _pending:
                continue

            _caption = " ".join(
                str(
                    _row.get("merged_descriptions") or ""
                ).split()
            )

            if not _caption or not _row.get("num_images"):
                continue

            _lookup = _lookups[_name]

            _idx = None

            for _field in (
                "dbpedia_url",
                "wikidata_url",
                "entity_qid",
                "entity_name",
            ):
                for _key in _keys(_row.get(_field)):
                    if _lookup.get(_key) is not None:
                        _idx = _lookup[_key]
                        break

                if _idx is not None:
                    break

            if _idx is None:
                continue

            _idx = str(_idx)
            _source = _row["source"]

            _previous = _sources[_name].get(_idx)

            if (
                _previous is None
                or _priority.get(_source, 0)
                > _priority.get(_previous, 0)
            ):
                _captions[_name][_idx] = _caption[:1000]
                _sources[_name][_idx] = _source

        for _name, _path in _pending.items():
            with open(
                _path / "image_captions.json",
                "w",
                encoding="utf-8",
            ) as _file:
                json.dump(
                    _captions[_name],
                    _file,
                    ensure_ascii=False,
                )

            _count = len(_captions[_name])


    return (cache_visual_captions,)


@app.cell
def _(cache_visual_captions, caption_repo, feature_sources):
    cache_visual_captions(
        sources=feature_sources,
        repo=caption_repo,
    )
    return


@app.cell
def _(graph_data, json, re, torch, unquote):
    def prepare_entity_texts(dataset, path):
        def _label(value):
            _value = unquote(
                str(value).strip().strip("<>")
            )

            _name = (
                _value.rsplit("/", 1)[-1]
                .rsplit("#", 1)[-1]
            )

            return re.sub(
                r"[_\s]+",
                " ",
                _name,
            ).strip() or str(value)

        with open(
            path / "entity2id.json",
            encoding="utf-8",
        ) as _file:
            _entity2id = json.load(_file)

        _count = len(_entity2id)
        _labels = [""] * _count

        for _entity, _idx in _entity2id.items():
            _labels[_idx] = _label(_entity)

        _descriptions = {}

        if dataset == "DB15K":
            with open(
                path / "entity_descriptions.json",
                encoding="utf-8",
            ) as _file:
                _descriptions = json.load(_file)

        _contexts = [[] for _ in range(_count)]

        if dataset == "MKG_W":
            _link_path = (
                path / "dbpedia_to_wikidata_id.json"
            )

            if _link_path.exists():
                with open(
                    _link_path,
                    encoding="utf-8",
                ) as _file:
                    _links = json.load(_file)

                for _entity, _idx in _links.items():
                    _labels[int(_idx)] = _label(_entity)

            with open(
                path / "relation2id.json",
                encoding="utf-8",
            ) as _file:
                _relation2id = json.load(_file)

            _relations = {
                _idx: _label(_relation)
                for _relation, _idx in _relation2id.items()
            }

            for _h, _r, _t in graph_data[dataset]["train"].tolist():
                _relation = _relations[_r]

                if len(_contexts[_h]) < 3:
                    _contexts[_h].append(
                        f"{_relation}: {_labels[_t]}"
                    )

                if len(_contexts[_t]) < 3:
                    _contexts[_t].append(
                        f"{_relation}: {_labels[_h]}"
                    )

        with open(
            path / "image_captions.json",
            encoding="utf-8",
        ) as _file:
            _captions = json.load(_file)

        _texts = [""] * _count
        _visual_texts = [""] * _count

        _visual_mask = torch.zeros(
            _count,
            dtype=torch.bool,
        )

        for _entity, _idx in _entity2id.items():
            _description = (
                _descriptions.get(str(_idx), "") or ""
            ).strip()

            _context = "; ".join(_contexts[_idx])

            _texts[_idx] = (
                _description
                or (
                    f"{_labels[_idx]}. {_context}"
                    if _context
                    else _labels[_idx]
                )
            )

            _caption = (
                _captions.get(str(_idx))
                or _captions.get(_entity)
            )

            if isinstance(_caption, str) and _caption.strip():
                _visual_texts[_idx] = _caption.strip()
                _visual_mask[_idx] = True
            else:
                _visual_texts[_idx] = _labels[_idx]

        return {
            "texts": _texts,
            "visual_texts": _visual_texts,
            "visual_mask": _visual_mask,
        }

    return (prepare_entity_texts,)


@app.cell
def _(ROOT, caption_repo, feature_sources, json, load_dataset, re, unquote):
    paper_flow_version = "mghisal-paper-flow-v6f"

    mllm_config = {
        "model": "HuggingFaceTB/SmolVLM-256M-Instruct",
        "max_entities_per_dataset": 16,
        "max_image_size": 500,
        "max_image_bytes": 8_000_000,
        "max_new_tokens": 128,
        "save_every": 5,
    }

    paper_flow_root = ROOT / "features" / paper_flow_version
    paper_flow_root.mkdir(parents=True, exist_ok=True)


    def build_image_manifest(sources, repo, limit):
        def _keys(value):
            _value = unquote(
                str(value or "").strip().strip("<>")
            ).rstrip("/")

            if not _value:
                return []

            _leaf = (
                _value.rsplit("/", 1)[-1]
                .rsplit("#", 1)[-1]
            )

            _result = ["uri:" + _value.casefold()]

            if re.fullmatch(r"Q\d+", _leaf, re.I):
                _result.append("qid:" + _leaf.upper())

            if "dbpedia.org/resource/" in _value.lower():
                _result.append(
                    "db:" + _leaf.casefold()
                )

            _name = re.sub(
                r"[\s_]+", " ", _leaf
            ).strip().casefold()

            if _name and not re.fullmatch(
                r"Q\d+", _leaf, re.I
            ):
                _result.append("name:" + _name)

            return list(dict.fromkeys(_result))

        _lookups = {}
        _manifest = {}

        for _dataset, _path in sources.items():
            with open(
                _path / "entity2id.json",
                encoding="utf-8",
            ) as _file:
                _entity2id = json.load(_file)

            _index = {}

            def _register(value, idx):
                for _key in _keys(value):
                    if _key not in _index:
                        _index[_key] = int(idx)
                    elif _index[_key] != int(idx):
                        _index[_key] = None

            for _entity, _idx in _entity2id.items():
                _register(_entity, _idx)

            if _dataset == "DB15K":
                with open(
                    _path / "sameas_links.json",
                    encoding="utf-8",
                ) as _file:
                    _links = json.load(_file)

                for _freebase, _dbpedia in _links.items():
                    _matches = {
                        _index[_key]
                        for _key in _keys(_dbpedia)
                        if _index.get(_key) is not None
                    }

                    if len(_matches) == 1:
                        _register(
                            _freebase,
                            next(iter(_matches)),
                        )

            else:
                with open(
                    _path / "dbpedia_to_wikidata_id.json",
                    encoding="utf-8",
                ) as _file:
                    _links = json.load(_file)

                for _dbpedia, _idx in _links.items():
                    _register(_dbpedia, _idx)

            _lookups[_dataset] = _index
            _manifest[_dataset] = {}

        def _match(index, values):
            _matches = set()

            for _value in values:
                for _key in _keys(_value):
                    if index.get(_key) is not None:
                        _matches.add(index[_key])

            return (
                next(iter(_matches))
                if len(_matches) == 1
                else None
            )

        # Step 1: Connect DBpedia entities to Wikidata QIDs
        _entities = load_dataset(
            repo,
            "entities",
            split="train",
            streaming=True,
        )

        _joined = {name: 0 for name in sources}

        for _row in _entities:
            _dataset = str(
                _row.get("dataset", "")
            ).replace("-", "_")

            if _dataset not in _lookups:
                continue

            _index = _lookups[_dataset]

            _idx = _match(
                _index,
                [
                    _row.get("dbpedia_url"),
                    _row.get("wikidata_url"),
                    _row.get("entity_name"),
                ],
            )

            if _idx is None:
                continue

            _qid = _row.get("entity_qid")

            if _qid:
                _key = _keys(_qid)[-1]

                if _key.startswith("qid:"):
                    if _key not in _index:
                        _index[_key] = _idx
                    elif _index[_key] != _idx:
                        _index[_key] = None

            _joined[_dataset] += 1


        # Step 2: Match image metadata
        _metadata = load_dataset(
            repo,
            "image_metadata",
            split="train",
            streaming=True,
        )

        _counts = {name: 0 for name in sources}
        _examples = {}

        for _row in _metadata:
            _dataset = str(
                _row.get("dataset", "")
            ).replace("-", "_")

            if _dataset not in _lookups:
                continue

            _counts[_dataset] += 1

            if _dataset not in _examples:
                _examples[_dataset] = {
                    _key: _row.get(_key)
                    for _key in (
                        "dataset",
                        "entity_name",
                        "entity_qid",
                        "dbpedia_url",
                        "wikidata_url",
                        "image_id",
                        "image_url",
                    )
                }

            if (
                limit is not None
                and len(_manifest[_dataset]) >= limit
            ):
                continue

            _image_id = str(
                _row.get("image_id") or ""
            )

            _qid_match = re.match(
                r"(Q\d+)_",
                _image_id,
                re.I,
            )

            _idx = _match(
                _lookups[_dataset],
                [
                    _row.get("entity_qid"),
                    _row.get("dbpedia_url"),
                    _row.get("wikidata_url"),
                    _row.get("entity_name"),
                    _qid_match.group(1)
                    if _qid_match else None,
                ],
            )

            if _idx is None:
                continue

            _url = _row.get("image_url")

            if not isinstance(_url, str):
                continue

            if not _url.startswith("https://"):
                continue

            _manifest[_dataset].setdefault(
                str(_idx),
                {
                    "image_url": _url,
                    "page_url": _row.get("page_url") or "",
                    "source": "wikipedia_enriched",
                },
            )

        for _dataset, _items in _manifest.items():
            _output = paper_flow_root / _dataset
            _output.mkdir(parents=True, exist_ok=True)

            with open(
                _output / "image_manifest.json",
                "w",
                encoding="utf-8",
            ) as _file:
                json.dump(
                    _items,
                    _file,
                    ensure_ascii=False,
                    indent=2,
                )


        return _manifest


    image_manifest = build_image_manifest(
        feature_sources,
        caption_repo,
        mllm_config["max_entities_per_dataset"],
    )
    return paper_flow_root, paper_flow_version


@app.cell
def _(feature_sources, json, paper_flow_root):
    mllm_records = {}

    for _dataset, _path in feature_sources.items():
        _output = paper_flow_root / _dataset
        _output.mkdir(parents=True, exist_ok=True)

        _cache = _output / "mllm_records.json"

        # Preserve any successful image-grounded records
        if _cache.exists():
            with open(_cache, encoding="utf-8") as _file:
                _records = json.load(_file)
        else:
            _records = {}

            with open(
                _cache,
                "w",
                encoding="utf-8",
            ) as _file:
                json.dump(_records, _file)

        mllm_records[_dataset] = _records

        _caption_path = _path / "image_captions.json"

        if _caption_path.exists():
            with open(
                _caption_path,
                encoding="utf-8",
            ) as _file:
                _captions = json.load(_file)
        else:
            _captions = {}


    return (mllm_records,)


@app.cell
def _(hashlib, json, paper_flow_version, torch):
    def encode_features(
        encoder,
        inputs,
        output_dir,
        model_name,
    ):
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        _cache_path = output_dir / "features.pt"

        _fingerprint = hashlib.sha256(
            json.dumps(
                [
                    model_name,
                    paper_flow_version,
                    inputs["texts"],
                    inputs["visual_texts"],
                    inputs["semantic_summaries"],
                    inputs["visual_mask"].tolist(),
                    inputs["mllm_mask"].tolist(),
                ],
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()

        if _cache_path.exists():
            _cached = torch.load(
                _cache_path,
                map_location="cpu",
                weights_only=True,
            )

            if _cached.get("fingerprint") == _fingerprint:

                return {
                    _key: _cached[_key]
                    for _key in (
                        "text",
                        "original_text",
                        "summary",
                        "visual_text",
                        "visual_mask",
                        "mllm_mask",
                    )
                }

        _all_texts = (
            inputs["texts"]
            + inputs["visual_texts"]
            + inputs["semantic_summaries"]
        )

        _unique = list(
            dict.fromkeys(_all_texts)
        )

        _positions = {
            _value: _idx
            for _idx, _value in enumerate(_unique)
        }

        _vectors = torch.from_numpy(
            encoder.encode(
                _unique,
                batch_size=32,
                show_progress_bar=True,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
        ).float()

        def _gather(texts):
            _indices = torch.tensor(
                [
                    _positions[_value]
                    for _value in texts
                ],
                dtype=torch.long,
            )

            return _vectors[
                _indices
            ].contiguous()

        _original = _gather(inputs["texts"])
        _visual = _gather(inputs["visual_texts"])
        _summary = _gather(
            inputs["semantic_summaries"]
        )

        _features = {
            "text": _summary,
            "original_text": _original,
            "summary": _summary,
            "visual_text": _visual,
            "visual_mask": inputs["visual_mask"],
            "mllm_mask": inputs["mllm_mask"],
        }

        torch.save(
            {
                **_features,
                "encoder": model_name,
                "fingerprint": _fingerprint,
            },
            _cache_path,
        )


        return _features

    return (encode_features,)


@app.cell
def _(
    DEVICE,
    SentenceTransformer,
    caption_repo,
    encode_features,
    encoder_name,
    feature_sources,
    json,
    mllm_records,
    paper_flow_root,
    paper_flow_version,
    prepare_entity_texts,
    torch,
):
    feature_data = {}

    _encoder = SentenceTransformer(
        encoder_name,
        device=DEVICE,
    )

    for _dataset, _path in feature_sources.items():

        _inputs = prepare_entity_texts(
            _dataset,
            _path,
        )

        _records = mllm_records.get(_dataset, {})
        _count = len(_inputs["texts"])

        _texts = list(_inputs["texts"])
        _visual_texts = list(_inputs["visual_texts"])
        _summaries = list(_texts)

        _visual_mask = _inputs["visual_mask"].clone()
        _mllm_mask = torch.zeros(
            _count,
            dtype=torch.bool,
        )

        for _idx in range(_count):
            _record = _records.get(str(_idx), {})

            _visual = _record.get(
                "visual_description", ""
            )

            _summary = _record.get(
                "semantic_summary", ""
            )

            if _visual and _summary:
                _visual_texts[_idx] = _visual
                _summaries[_idx] = _summary

                _visual_mask[_idx] = True
                _mllm_mask[_idx] = True

            elif _visual_mask[_idx].item():
                # Precomputed caption fallback
                _summaries[_idx] = (
                    _texts[_idx]
                    + "\nVisual context: "
                    + _visual_texts[_idx]
                )

        # Inputs required by encode_features()
        _feature_inputs = {
            "texts": _texts,
            "visual_texts": _visual_texts,
            "semantic_summaries": _summaries,
            "visual_mask": _visual_mask,
            "mllm_mask": _mllm_mask,
        }

        _output = paper_flow_root / _dataset
        _output.mkdir(
            parents=True,
            exist_ok=True,
        )

        _features = encode_features(
            encoder=_encoder,
            inputs=_feature_inputs,
            output_dir=_output,
            model_name=encoder_name,
        )

        feature_data[_dataset] = _features

        _metadata = {
            "dataset": _dataset,
            "version": paper_flow_version,
            "encoder": encoder_name,
            "caption_source": caption_repo,
            "feature_mode": "caption_summary_fallback",
            "num_entities": _count,
            "embedding_dim": _features["text"].shape[1],
            "mllm_entities": int(
                _mllm_mask.sum().item()
            ),
            "precomputed_caption_entities": int(
                (_visual_mask & ~_mllm_mask).sum().item()
            ),
            "missing_visual_entities": int(
                (~_visual_mask).sum().item()
            ),
            "mllm_coverage": float(
                _mllm_mask.float().mean().item()
            ),
            "visual_coverage": float(
                _visual_mask.float().mean().item()
            ),
        }

        with open(
            _output / "metadata.json",
            "w",
            encoding="utf-8",
        ) as _file:
            json.dump(
                _metadata,
                _file,
                indent=2,
            )





    del _encoder

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return (feature_data,)


@app.cell
def _(
    WANDB_PROJECT,
    caption_repo,
    encoder_name,
    feature_sources,
    json,
    paper_flow_root,
    paper_flow_version,
    wandb,
):
    with wandb.init(
        project=WANDB_PROJECT,
        job_type="feature-preparation",
        name=f"mghisal-{paper_flow_version}-features",
        group=paper_flow_version,
        tags=[
            "caption-fallback",
            "semantic-summary",
            paper_flow_version,
        ],
        config={
            "version": paper_flow_version,
            "encoder": encoder_name,
            "caption_source": caption_repo,
            "feature_mode": "caption_summary_fallback",
            "mllm_enabled": False,
        },
    ) as _run:

        for _dataset, _path in feature_sources.items():
            _name = _dataset.lower().replace("_", "-")
            _output = paper_flow_root / _dataset

            with open(
                _output / "metadata.json",
                encoding="utf-8",
            ) as _file:
                _metadata = json.load(_file)

            _artifact = wandb.Artifact(
                name=f"mghisal-{_name}-{paper_flow_version}-features",
                type="dataset",
                metadata=_metadata,
            )

            for _file_path in (
                _output / "features.pt",
                _output / "metadata.json",
                _path / "image_captions.json",
            ):
                if _file_path.exists():
                    _artifact.add_file(
                        str(_file_path)
                    )

            _run.log({
                f"{_name}/visual_coverage": (
                    _metadata["visual_coverage"]
                ),
                f"{_name}/mllm_coverage": (
                    _metadata["mllm_coverage"]
                ),
            })

            _run.log_artifact(
                _artifact,
                aliases=["latest", "v6"],
            ).wait()

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. MgHiSal Training Pipeline
    """)
    return


@app.cell
def _(DEVICE, ROOT, paper_flow_version):
    import torch.nn.functional as F

    experiment_version = paper_flow_version

    training_config = {
        _name: {
            "epochs": 50,
            "steps_per_epoch": 10,
            "batch_size": 512 if DEVICE == "cuda" else 128,
            "num_negatives": 32,
            "temperature": 1.0,

            "embedding_dim": 256,
            "attention_heads": 4,
            "dropout": 0.1,

            "contrastive_temperature": 0.1,
            "contrastive_samples": 256,
            "visual_contrastive_weight": 0.03,
            "hier_contrastive_weight": 0.01,
            "gate_reg_weight": 0.001,

            "learning_rate": 7e-4,
            "weight_decay": 1e-5,
            "grad_clip": 1.0,

            "valid_samples": 512 if DEVICE == "cuda" else 128,
            "validate_every": 2,
            "valid_batch_size": 16 if DEVICE == "cuda" else 4,
            "candidate_chunk_size": 4096 if DEVICE == "cuda" else 1024,
            "tie_tolerance": 1e-5,

            "patience": 6,
            "seed": _seed,
            "resume": True,
        }
        for _name, _seed in (
            ("DB15K", 42),
            ("MKG_W", 43),
        )
    }

    checkpoint_root = ROOT / "checkpoints" / experiment_version
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    return F, checkpoint_root, experiment_version, training_config


@app.cell
def _(torch):
    def prepare_modal_features(features, device):
        _text = features["text"].to(
            device=device,
            dtype=torch.float32,
        )

        _visual = features["visual_text"].to(
            device=device,
            dtype=torch.float32,
        )

        _mask = features["visual_mask"].to(
            device=device,
            dtype=torch.bool,
        )

        _visual = _visual.masked_fill(
            ~_mask.unsqueeze(1),
            0.0,
        )

        return _text, _visual


    def sample_negative_tails(
        positive_triples,
        num_entities,
        num_negatives,
        known_tails,
        generator,
    ):
        _rows = positive_triples.repeat_interleave(
            num_negatives,
            dim=0,
        ).tolist()

        _pending = list(range(len(_rows)))

        for _ in range(32):
            if not _pending:
                break

            _samples = torch.randint(
                0,
                num_entities,
                (len(_pending),),
                generator=generator,
            ).tolist()

            _retry = []

            for _idx, _candidate in zip(
                _pending,
                _samples,
            ):
                _h, _r, _t = _rows[_idx]

                if (
                    _candidate == _t
                    or _candidate in known_tails.get(
                        (_h, _r), ()
                    )
                ):
                    _retry.append(_idx)
                else:
                    _rows[_idx][2] = _candidate

            _pending = _retry

        for _idx in _pending:
            _h, _r, _t = _rows[_idx]

            _forbidden = set(
                known_tails.get((_h, _r), ())
            )
            _forbidden.add(_t)

            if len(_forbidden) >= num_entities:
                raise RuntimeError(
                    f"No valid negative for ({_h}, {_r}, {_t})"
                )

            _start = int(
                torch.randint(
                    num_entities,
                    (1,),
                    generator=generator,
                ).item()
            )

            for _offset in range(num_entities):
                _candidate = (
                    _start + _offset
                ) % num_entities

                if _candidate not in _forbidden:
                    _rows[_idx][2] = _candidate
                    break

        return torch.tensor(
            _rows,
            dtype=torch.long,
        )

    return prepare_modal_features, sample_negative_tails


@app.cell
def _(torch):
    def build_validation_filters(data):
        _head_filter = {}
        _tail_filter = {}

        for _split in ("train", "valid"):
            for _h, _r, _t in data[_split].tolist():
                _head_filter.setdefault(
                    (_r, _t), set()
                ).add(_h)

                _tail_filter.setdefault(
                    (_h, _r), set()
                ).add(_t)

        return _head_filter, _tail_filter


    @torch.inference_mode()
    def evaluate_filtered_validation(
        model,
        embeddings,
        triples,
        head_filter,
        tail_filter,
        config,
    ):
        _device = embeddings.device
        _num_entities = embeddings.shape[0]
        _num_relations = model.relations.num_embeddings // 2

        _batch_size = config["valid_batch_size"]
        _chunk_size = config["candidate_chunk_size"]
        _tolerance = config["tie_tolerance"]

        _all_ranks = {}

        for _direction, _lookup in (
            ("head", head_filter),
            ("tail", tail_filter),
        ):
            _rank_parts = []

            for _start in range(
                0, len(triples), _batch_size
            ):
                _batch_cpu = triples[
                    _start:_start + _batch_size
                ].long().cpu()

                _batch = _batch_cpu.to(_device)

                if _direction == "head":
                    _queries = torch.stack(
                        [
                            _batch[:, 2],
                            _batch[:, 1] + _num_relations,
                            _batch[:, 0],
                        ],
                        dim=1,
                    )
                else:
                    _queries = _batch

                _gold = model.score(
                    embeddings,
                    _queries,
                ).unsqueeze(1)

                _mask_rows = []
                _mask_entities = []

                for _row, (_h, _r, _t) in enumerate(
                    _batch_cpu.tolist()
                ):
                    if _direction == "head":
                        _key = (_r, _t)
                        _target = _h
                    else:
                        _key = (_h, _r)
                        _target = _t

                    for _entity in (
                        _lookup.get(_key, set()) | {_target}
                    ):
                        _mask_rows.append(_row)
                        _mask_entities.append(_entity)

                _mask_rows = torch.tensor(
                    _mask_rows,
                    dtype=torch.long,
                    device=_device,
                )

                _mask_entities = torch.tensor(
                    _mask_entities,
                    dtype=torch.long,
                    device=_device,
                )

                _greater = torch.zeros(
                    len(_batch),
                    dtype=torch.long,
                    device=_device,
                )

                _ties = torch.zeros_like(_greater)

                for _offset in range(
                    0,
                    _num_entities,
                    _chunk_size,
                ):
                    _end = min(
                        _offset + _chunk_size,
                        _num_entities,
                    )

                    _candidates = torch.arange(
                        _offset,
                        _end,
                        device=_device,
                    )

                    _scores = model.score_candidates(
                        embeddings,
                        _queries,
                        _candidates,
                        direction="tail",
                    )

                    _within = (
                        (_mask_entities >= _offset)
                        & (_mask_entities < _end)
                    )

                    _scores[
                        _mask_rows[_within],
                        _mask_entities[_within] - _offset,
                    ] = -torch.inf

                    _greater += (
                        _scores > _gold + _tolerance
                    ).sum(dim=1)

                    _ties += (
                        (
                            _scores - _gold
                        ).abs() <= _tolerance
                    ).sum(dim=1)

                _ranks = (
                    1.0
                    + _greater.float()
                    + 0.5 * _ties.float()
                )

                _rank_parts.append(
                    _ranks.cpu()
                )

            _all_ranks[_direction] = torch.cat(
                _rank_parts
            )

        def _metrics(ranks):
            return {
                "mr": ranks.mean().item(),
                "mrr": (
                    1.0 / ranks
                ).mean().item(),
                "hits1": (
                    ranks <= 1
                ).float().mean().item(),
                "hits3": (
                    ranks <= 3
                ).float().mean().item(),
                "hits10": (
                    ranks <= 10
                ).float().mean().item(),
            }

        _both = torch.cat([
            _all_ranks["head"],
            _all_ranks["tail"],
        ])

        return {
            "head": _metrics(_all_ranks["head"]),
            "tail": _metrics(_all_ranks["tail"]),
            "both": _metrics(_both),
        }

    return build_validation_filters, evaluate_filtered_validation


@app.cell
def _(
    DEVICE,
    F,
    MgHiSal,
    WANDB_PROJECT,
    build_validation_filters,
    checkpoint_root,
    evaluate_filtered_validation,
    experiment_version,
    feature_data,
    graph_data,
    json,
    prepare_modal_features,
    sample_negative_tails,
    torch,
    training_config,
    wandb,
):
    def contrastive_alignment_loss(
        aux,
        entity_ids,
        temperature=0.1,
        max_samples=256,
    ):
        _ids = torch.unique(
            entity_ids.reshape(-1)
        )

        if _ids.numel() > max_samples:
            _ids = _ids[
                torch.randperm(
                    _ids.numel(),
                    device=_ids.device,
                )[:max_samples]
            ]

        def _pair(_left, _right):
            if len(_left) < 2:
                return _left.sum() * 0.0

            _left = F.normalize(_left, dim=-1)
            _right = F.normalize(_right, dim=-1)

            _logits = (
                _left @ _right.T
            ) / temperature

            _targets = torch.arange(
                len(_left),
                device=_left.device,
            )

            return 0.5 * (
                F.cross_entropy(_logits, _targets)
                + F.cross_entropy(
                    _logits.T,
                    _targets,
                )
            )

        # Only entities with visual descriptions
        _visual_ids = _ids[
            aux["visual_mask"][_ids]
        ]

        _visual_loss = _pair(
            aux["text"][_visual_ids],
            aux["visual"][_visual_ids],
        )

        _hierarchy_loss = _pair(
            aux["local"][_ids],
            aux["fused"][_ids],
        )

        return _visual_loss, _hierarchy_loss


    def train_model(dataset_name):
        _data = graph_data[dataset_name]
        _features = feature_data[dataset_name]
        _config = training_config[dataset_name]

        _device = torch.device(DEVICE)
        _rng = torch.Generator().manual_seed(
            _config["seed"]
        )

        torch.manual_seed(_config["seed"])

        _output = checkpoint_root / dataset_name
        _output.mkdir(parents=True, exist_ok=True)

        _last_path = _output / "last.pt"
        _best_path = _output / "best.pt"

        _train = _data["train"].long().cpu()
        _valid_raw = _data["valid"].long().cpu()
        _test = _data["test"].long().cpu()
    
        _train_set = set(map(tuple, _train.tolist()))
        _test_set = set(map(tuple, _test.tolist()))
    
        _clean_rows = []
        _seen = set()
    
        _removed_train = 0
        _removed_test = 0
        _removed_duplicate = 0

        for _row in _valid_raw.tolist():
            _triple = tuple(_row)
    
            if dataset_name == "DB15K":
                if _triple in _train_set:
                    _removed_train += 1
                    continue
    
                if _triple in _test_set:
                    _removed_test += 1
                    continue
    
                if _triple in _seen:
                    _removed_duplicate += 1
                    continue
    
            _clean_rows.append(_row)
            _seen.add(_triple)
    
            _valid = torch.tensor(
                _clean_rows,
                dtype=torch.long,
            ).reshape(-1, 3)
        
            if len(_valid) == 0:
                raise RuntimeError(
                    f"{dataset_name}: no validation triples remain."
                )
        
            _validation_stats = {
                "policy": "exclude_train_test_overlap",
                "dataset": dataset_name,
                "original_valid": len(_valid_raw),
                "clean_valid": len(_valid),
                "removed_train": _removed_train,
                "removed_test": _removed_test,
                "removed_duplicate": _removed_duplicate,
            }
        
            _clean_valid_path = _output / "valid_clean2id.txt"
        
            with open(
                _clean_valid_path,
                "w",
                encoding="utf-8",
            ) as _file:
                for _h, _r, _t in _valid.tolist():
                    _file.write(f"{_h}\t{_r}\t{_t}\n")
        
            _validation_report_path = (
                _output / "validation_split_policy.json"
            )
        
            with open(
                _validation_report_path,
                "w",
                encoding="utf-8",
            ) as _file:
                json.dump(
                    _validation_stats,
                    _file,
                    indent=2,
                )
        

        _num_entities = _data["num_entities"]
        _num_relations = _data["num_relations"]

        # Reciprocal triples
        _inverse = torch.stack(
            [
                _train[:, 2],
                _train[:, 1] + _num_relations,
                _train[:, 0],
            ],
            dim=1,
        )

        _known_tails = {}

        for _h, _r, _t in torch.cat(
            [_train, _inverse],
            dim=0,
        ).tolist():
            _known_tails.setdefault(
                (_h, _r), set()
            ).add(_t)

        _text, _visual = prepare_modal_features(
            _features,
            _device,
        )

        _edge_index = torch.stack(
            [
                torch.cat([
                    _train[:, 0],
                    _train[:, 2],
                ]),
                torch.cat([
                    _train[:, 2],
                    _train[:, 0],
                ]),
            ],
            dim=0,
        ).to(_device)

        # Hierarchical MgHiSal v3
        _model = MgHiSal(
            num_entities=_num_entities,
            num_relations=2 * _num_relations,
            dim=_config["embedding_dim"],
            heads=_config["attention_heads"],
            dropout=_config["dropout"],
        ).to(_device)

        with torch.no_grad():
            _model.text_proj(_text[:1])
            _model.visual_text_proj(_visual[:1])

        _optimizer = torch.optim.AdamW(
            _model.parameters(),
            lr=_config["learning_rate"],
            weight_decay=_config["weight_decay"],
        )

        # Fixed filtered validation
        _valid_rng = torch.Generator().manual_seed(
            _config["seed"] + 1000
        )

        _valid_count = min(
            _config["valid_samples"],
            len(_valid),
        )

        _valid_batch = _valid[
            torch.randperm(
                len(_valid),
                generator=_valid_rng,
            )[:_valid_count]
        ]

        _head_filter, _tail_filter = build_validation_filters({
            "train": _train,
            "valid": _valid,
        })

        _history = []
        _best_mrr = -1.0
        _best_epoch = 0
        _stale = 0
        _start_epoch = 1

        # Resume same version only
        if _config["resume"] and _last_path.exists():
            _state = torch.load(
                _last_path,
                map_location="cpu",
                weights_only=True,
            )

            if (
                _state.get("experiment_version")
                != experiment_version
                or _state.get("architecture")
                != "hierarchical_gated_neighbor_v3"
                or _state.get("num_relations")
                != _num_relations
            ):
                raise RuntimeError(
                    "Incompatible v3 checkpoint."
                )

            _model.load_state_dict(
                _state["model_state_dict"]
            )

            _optimizer.load_state_dict(
                _state["optimizer_state_dict"]
            )

            for _group in _optimizer.param_groups:
                _group["lr"] = _config["learning_rate"]

            _history = _state["history"]
            _best_mrr = _state["best_mrr"]
            _best_epoch = _state["best_epoch"]
            _stale = _state["stale_epochs"]
            _start_epoch = _state["epoch"] + 1

            _rng.set_state(
                _state["generator_state"].cpu()
            )


        if _start_epoch > _config["epochs"]:
            return {
                "history": _history,
                "best_mrr": _best_mrr,
                "best_epoch": _best_epoch,
                "checkpoint": str(_best_path),
            }

        # Versioned W&B run
        with wandb.init(
            project=WANDB_PROJECT,
            job_type="training",
            name=(
                f"mghisal-{dataset_name.lower()}-"
                f"{experiment_version}"
            ),
            group=experiment_version,
            tags=[
                "hierarchical-gated",
                "neighbor-attention",
                "contrastive",
                "reciprocal",
                dataset_name,
            ],
            config={
                **_config,
                "dataset": dataset_name,
                "experiment_version": experiment_version,
                "architecture": "hierarchical_gated_neighbor_v3",
                "objective": "sampled_softmax_plus_contrastive",
                "model_relations": 2 * _num_relations,
                "validation_policy": "exclude_train_test_overlap",
                "validation_stats": _validation_stats,
            },
        ) as _run:

            for _epoch in range(
                _start_epoch,
                _config["epochs"] + 1,
            ):
                _model.train()

                _epoch_losses = []
                _link_losses = []
                _tv_losses = []
                _hier_losses = []

                for _step in range(
                    _config["steps_per_epoch"]
                ):
                    # Balanced original / inverse batches
                    _half = _config["batch_size"] // 2

                    _orig_ids = torch.randint(
                        len(_train),
                        (_half,),
                        generator=_rng,
                    )

                    _inv_ids = torch.randint(
                        len(_inverse),
                        (
                            _config["batch_size"] - _half,
                        ),
                        generator=_rng,
                    )

                    _positive_cpu = torch.cat(
                        [
                            _train[_orig_ids],
                            _inverse[_inv_ids],
                        ],
                        dim=0,
                    )

                    _positive_cpu = _positive_cpu[
                        torch.randperm(
                            len(_positive_cpu),
                            generator=_rng,
                        )
                    ]

                    _negative_cpu = sample_negative_tails(
                        positive_triples=_positive_cpu,
                        num_entities=_num_entities,
                        num_negatives=_config["num_negatives"],
                        known_tails=_known_tails,
                        generator=_rng,
                    )

                    _positive = _positive_cpu.to(_device)

                    _negative_ids = (
                        _negative_cpu[:, 2]
                        .reshape(
                            len(_positive_cpu),
                            _config["num_negatives"],
                        )
                        .to(_device)
                    )

                    _optimizer.zero_grad(
                        set_to_none=True
                    )

                    # Forward with intermediate representations
                    _embeddings, _aux = _model(
                        _text,
                        _visual,
                        _edge_index,
                        return_aux=True,
                    )

                    # Reciprocal ComplEx scores
                    _positive_scores = _model.score(
                        _embeddings,
                        _positive,
                    ).unsqueeze(1)

                    _negative_scores = (
                        _model.score_candidates(
                            _embeddings,
                            _positive,
                            _negative_ids,
                            direction="tail",
                        )
                    )

                    _logits = torch.cat(
                        [
                            _positive_scores,
                            _negative_scores,
                        ],
                        dim=1,
                    )

                    _logits = (
                        _logits / _config["temperature"]
                    )

                    _targets = torch.zeros(
                        len(_positive),
                        dtype=torch.long,
                        device=_device,
                    )

                    # Link prediction
                    _link_loss = F.cross_entropy(
                        _logits,
                        _targets,
                    )

                    # Hierarchical contrastive alignment
                    _tv_loss, _hier_loss = (
                        contrastive_alignment_loss(
                            _aux,
                            _positive[:, [0, 2]],
                            temperature=_config[
                                "contrastive_temperature"
                            ],
                            max_samples=_config[
                                "contrastive_samples"
                            ],
                        )
                    )

                    # Combined objective
                    _loss = (
                        _link_loss
                        + _config["visual_contrastive_weight"]
                        * _tv_loss
                        + _config["hier_contrastive_weight"]
                        * _hier_loss
                        + _config["gate_reg_weight"]
                        * _aux["gate_reg"]
                    )

                    _loss.backward()

                    torch.nn.utils.clip_grad_norm_(
                        _model.parameters(),
                        _config["grad_clip"],
                    )

                    _optimizer.step()

                    _epoch_losses.append(
                        _loss.item()
                    )
                    _link_losses.append(
                        _link_loss.item()
                    )
                    _tv_losses.append(
                        _tv_loss.item()
                    )
                    _hier_losses.append(
                        _hier_loss.item()
                    )

                    del (
                        _embeddings,
                        _aux,
                        _positive_scores,
                        _negative_scores,
                        _logits,
                        _loss,
                    )

                _loss_avg = (
                    sum(_epoch_losses)
                    / len(_epoch_losses)
                )

                # Full filtered Head + Tail validation
                _validate = (
                    _epoch == 1
                    or _epoch % _config["validate_every"] == 0
                    or _epoch == _config["epochs"]
                )

                _metrics = None
                _is_best = False

                if _validate:
                    _model.eval()

                    with torch.inference_mode():
                        _val_embeddings = _model(
                            _text,
                            _visual,
                            _edge_index,
                        )

                        _metrics = evaluate_filtered_validation(
                            model=_model,
                            embeddings=_val_embeddings,
                            triples=_valid_batch,
                            head_filter=_head_filter,
                            tail_filter=_tail_filter,
                            config=_config,
                        )

                    del _val_embeddings

                    _current_mrr = (
                        _metrics["both"]["mrr"]
                    )

                    if _current_mrr > _best_mrr:
                        _best_mrr = _current_mrr
                        _best_epoch = _epoch
                        _stale = 0
                        _is_best = True
                    else:
                        _stale += 1

                _record = {
                    "epoch": _epoch,
                    "train_loss": _loss_avg,
                    "link_loss": (
                        sum(_link_losses) / len(_link_losses)
                    ),
                    "contrastive_tv": (
                        sum(_tv_losses) / len(_tv_losses)
                    ),
                    "contrastive_hier": (
                        sum(_hier_losses) / len(_hier_losses)
                    ),
                }

                if _metrics is not None:
                    _record.update({
                        "filtered_mrr": _metrics["both"]["mrr"],
                        "head_mrr": _metrics["head"]["mrr"],
                        "tail_mrr": _metrics["tail"]["mrr"],
                        "filtered_hits10": (
                            _metrics["both"]["hits10"]
                        ),
                    })

                _history.append(_record)

                # Save v3 checkpoint
                _state = {
                    "epoch": _epoch,
                    "dataset": dataset_name,
                    "experiment_version": experiment_version,
                    "architecture": "hierarchical_gated_neighbor_v3",
                    "training_mode": "reciprocal",
                    "objective": "sampled_softmax_plus_contrastive",
                    "num_entities": _num_entities,
                    "num_relations": _num_relations,
                    "model_num_relations": 2 * _num_relations,
                    "model_state_dict": _model.state_dict(),
                    "optimizer_state_dict": (
                        _optimizer.state_dict()
                    ),
                    "generator_state": _rng.get_state(),
                    "history": _history,
                    "best_mrr": _best_mrr,
                    "best_epoch": _best_epoch,
                    "stale_epochs": _stale,
                    "config": _config,
                    "validation_policy": "exclude_train_test_overlap",
                    "validation_stats": _validation_stats,
                }

                _temp = _output / "last.tmp"

                torch.save(
                    _state,
                    _temp,
                )

                _temp.replace(
                    _last_path
                )

                if _is_best:
                    torch.save(
                        _state,
                        _best_path,
                    )

                # W&B logging
                _log = {
                    "train/loss": _record["train_loss"],
                    "train/link_loss": _record["link_loss"],
                    "train/contrastive_tv": (
                        _record["contrastive_tv"]
                    ),
                    "train/contrastive_hier": (
                        _record["contrastive_hier"]
                    ),
                    "best/filtered_mrr": _best_mrr,
                }

                if _metrics is not None:
                    for _direction in (
                        "head",
                        "tail",
                        "both",
                    ):
                        for _metric, _value in (
                            _metrics[_direction].items()
                        ):
                            _log[
                                f"valid/{_direction}/{_metric}"
                            ] = _value

                _run.log(
                    _log,
                    step=_epoch,
                )


                if _stale >= _config["patience"]:
                    break

            # Versioned model artifact
            _artifact = wandb.Artifact(
                name=(
                    f"mghisal-{dataset_name.lower()}-"
                    f"{experiment_version}-model"
                ),
                type="model",
                metadata={
                    "dataset": dataset_name,
                    "experiment_version": experiment_version,
                    "architecture": "hierarchical_gated_neighbor_v3",
                    "best_epoch": _best_epoch,
                    "best_filtered_mrr": _best_mrr,
                },
            )

            _artifact.add_file(
                str(_best_path)
            )

            _artifact.add_file(str(_clean_valid_path))
            _artifact.add_file(str(_validation_report_path))
            _run.log_artifact(
                _artifact,
                aliases=["latest", "best"],
            ).wait()

        return {
            "history": _history,
            "best_mrr": _best_mrr,
            "best_epoch": _best_epoch,
            "checkpoint": str(_best_path),
            "experiment_version": experiment_version,
        }

    return (train_model,)


@app.cell
def _(mo):
    train_db15k_button = mo.ui.run_button(
        label="Train DB15K"
    )

    train_mkg_w_button = mo.ui.run_button(
        label="Train MKG-W"
    )

    mo.vstack([
        train_db15k_button,
        train_mkg_w_button,
    ])
    return train_db15k_button, train_mkg_w_button


@app.cell
def _(mo, train_db15k_button, train_model):
    mo.stop(
        not train_db15k_button.value,
        mo.md("Click **Train DB15K** to start."),
    )

    train_result_db15k = train_model("DB15K")



    return


@app.cell
def _(mo, train_mkg_w_button, train_model):
    mo.stop(
        not train_mkg_w_button.value,
        mo.md("Click **Train MKG-W** to start."),
    )

    train_result_mkg_w = train_model("MKG_W")



    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Full-Ranking Evaluation
    """)
    return


@app.cell
def _(DEVICE, experiment_version):
    evaluation_version = f"{experiment_version}-fulltest"

    evaluation_config = {
        _name: {
            "max_test_triples": None,
            "batch_size": 16 if DEVICE == "cuda" else 4,
            "candidate_chunk_size": 4096 if DEVICE == "cuda" else 1024,
            "tie_tolerance": 1e-5,
            "seed": _seed,
            "log_wandb": True,
        }
        for _name, _seed in (
            ("DB15K", 42),
            ("MKG_W", 43),
        )
    }

    return evaluation_config, evaluation_version


@app.function
def build_filter_maps(data):
    _head_filter = {}
    _tail_filter = {}

    for _split in ("train", "valid", "test"):
        _triples = data[_split].tolist()

        for _h, _r, _t in _triples:
            _head_filter.setdefault(
                (_r, _t), set()
            ).add(_h)

            _tail_filter.setdefault(
                (_h, _r), set()
            ).add(_t)

    return _head_filter, _tail_filter


@app.cell
def _(evaluate_filtered_validation, torch):
    @torch.inference_mode()
    def evaluate_filtered_ranking(
        model,
        embeddings,
        triples,
        head_filter,
        tail_filter,
        config,
    ):
        _config = {
            **config,
            "valid_batch_size": config["batch_size"],
        }

        return evaluate_filtered_validation(
            model=model,
            embeddings=embeddings,
            triples=triples,
            head_filter=head_filter,
            tail_filter=tail_filter,
            config=_config,
        )

    return (evaluate_filtered_ranking,)


@app.cell
def _(
    DEVICE,
    MgHiSal,
    WANDB_PROJECT,
    checkpoint_root,
    evaluate_filtered_ranking,
    evaluation_config,
    evaluation_version,
    experiment_version,
    feature_data,
    graph_data,
    json,
    prepare_modal_features,
    torch,
    wandb,
):
    def run_evaluation(dataset_name):
        _data = graph_data[dataset_name]
        _features = feature_data[dataset_name]
        _config = evaluation_config[dataset_name]
        _device = torch.device(DEVICE)

        _output = checkpoint_root / dataset_name
        _checkpoint_path = _output / "best.pt"

        # 1. Load reciprocal checkpoint
        _checkpoint = torch.load(
            _checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )

        if _checkpoint.get("training_mode") != "reciprocal":
            raise RuntimeError(
                "Expected a reciprocal checkpoint."
            )

        _model_config = _checkpoint["config"]
        _num_relations = _data["num_relations"]

        # 2. Prepare multimodal features
        _text, _visual = prepare_modal_features(
            _features,
            _device,
        )

        # 3. Initialize Reciprocal ComplEx
        _model = MgHiSal(
            num_entities=_data["num_entities"],
            num_relations=2 * _num_relations,
            dim=_model_config["embedding_dim"],
            heads=_model_config["attention_heads"],
        ).to(_device)

        # Initialize LazyLinear layers
        with torch.no_grad():
            _model.text_proj(_text[:1])
            _model.visual_text_proj(_visual[:1])

        _model.load_state_dict(
            _checkpoint["model_state_dict"]
        )

        _model.eval()

        # 4. Build graph using train triples only
        _train = _data["train"].long().cpu()

        _edge_index = torch.stack(
            [
                torch.cat([
                    _train[:, 0],
                    _train[:, 2],
                ]),
                torch.cat([
                    _train[:, 2],
                    _train[:, 0],
                ]),
            ],
            dim=0,
        ).to(_device)

        # 5. Select test triples
        _test_all = _data["test"].long().cpu()
        _test = _test_all

        _limit = _config["max_test_triples"]

        if (
            _limit is not None
            and _limit < len(_test_all)
        ):
            _rng = torch.Generator().manual_seed(
                _config["seed"]
            )

            _indices = torch.randperm(
                len(_test_all),
                generator=_rng,
            )[:_limit]

            _test = _test_all[_indices]

        # 6. Build filtered maps
        # Filtering uses train + valid + test
        _head_filter, _tail_filter = build_filter_maps(
            _data
        )


        # 7. Generate entity embeddings and evaluate
        with torch.inference_mode():
            _embeddings = _model(
                _text,
                _visual,
                _edge_index,
            )

            _metrics = evaluate_filtered_ranking(
                model=_model,
                embeddings=_embeddings,
                triples=_test,
                head_filter=_head_filter,
                tail_filter=_tail_filter,
                config=_config,
            )

        del _embeddings

        # 8. Prepare evaluation report
        _result = {
            "dataset": dataset_name,
            "training_mode": "reciprocal",
            "scoring": "ComplEx",
            "checkpoint_epoch": _checkpoint["epoch"],
            "best_epoch": _checkpoint["best_epoch"],
            "num_entities": _data["num_entities"],
            "original_relations": _num_relations,
            "model_relations": 2 * _num_relations,
            "test_triples_total": len(_test_all),
            "test_triples_evaluated": len(_test),
            "test_subset": len(_test) < len(_test_all),
            "candidate_policy": "all_entities",
            "filter_policy": "train_valid_test",
            "head_prediction": "inverse_relation_tail",
            "tail_prediction": "original_relation_tail",
            "tie_policy": "average",
            "metrics": _metrics,
        }
        _result["experiment_version"] = experiment_version
        _result["objective"] = "sampled_softmax"
        _result["evaluation_version"] = evaluation_version

        _report_path = (
            _output / f"filtered_evaluation_{evaluation_version}.json"
        )

        with open(
            _report_path,
            "w",
            encoding="utf-8",
        ) as _file:
            json.dump(
                _result,
                _file,
                indent=2,
            )

        # 9. Log evaluation to W&B
        if _config["log_wandb"]:
            with wandb.init(
                project=WANDB_PROJECT,
                job_type="evaluation",
                name=f"mghisal-{dataset_name.lower()}-{evaluation_version}-eval",
                group=experiment_version,
                tags=[
                    "reciprocal",
                    "sampled-softmax",
                    experiment_version,
                    dataset_name,
                ],
                config={
                    "dataset": dataset_name,
                    "experiment_version": experiment_version,
                    "training_mode": "reciprocal",
                    "objective": "sampled_softmax",
                    **_config,
                },
            ) as _run:

                _run.log({
                    f"filtered/{_direction}/{_metric}": _value
                    for _direction, _values in _metrics.items()
                    for _metric, _value in _values.items()
                })

                _run.summary["checkpoint_epoch"] = (
                    _checkpoint["epoch"]
                )

                _run.summary["best_epoch"] = (
                    _checkpoint["best_epoch"]
                )

                _run.summary["test_triples_evaluated"] = (
                    len(_test)
                )

                _artifact = wandb.Artifact(
                    name=f"mghisal-{dataset_name.lower()}-{evaluation_version}-evaluation",
                    type="evaluation",
                )

                _artifact.add_file(
                    str(_report_path)
                )

                _run.log_artifact(
                    _artifact,
                    aliases=["latest"],
                ).wait()

        # 10. Print metrics
        for _direction in ("head", "tail", "both"):
            _scores = _metrics[_direction]



        return _result

    return (run_evaluation,)


@app.cell
def _(mo):
    evaluate_db15k_button = mo.ui.run_button(
        label="Evaluate DB15K"
    )

    evaluate_mkg_w_button = mo.ui.run_button(
        label="Evaluate MKG-W"
    )

    mo.vstack([
        evaluate_db15k_button,
        evaluate_mkg_w_button,
    ])
    return evaluate_db15k_button, evaluate_mkg_w_button


@app.cell
def _(evaluate_db15k_button, mo, run_evaluation):
    mo.stop(
        not evaluate_db15k_button.value,
        mo.md("Click **Evaluate DB15K** to start."),
    )

    evaluation_result_db15k = run_evaluation("DB15K")

    return


@app.cell
def _(evaluate_mkg_w_button, mo, run_evaluation):
    mo.stop(
        not evaluate_mkg_w_button.value,
        mo.md("Click **Evaluate MKG-W** to start."),
    )

    evaluation_result_mkg_w = run_evaluation("MKG_W")

    return


@app.cell
def _(checkpoint_root, evaluation_version, json, mo):
    _report_rows = [
        "| Dataset | Triples | MRR | MR | Hits@1 | Hits@3 | Hits@10 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    _missing = []
    _found = 0

    for _dataset in ("DB15K", "MKG_W"):
        _folder = checkpoint_root / _dataset

        _path = (
            _folder
            / f"filtered_evaluation_{evaluation_version}.json"
        )

        if not _path.exists():
            _available = [
                _file.name
                for _file in _folder.glob(
                    "filtered_evaluation*.json"
                )
            ]

            _missing.append(
                f"**{_dataset}**: Missing `{_path.name}`. "
                f"Available: {_available}"
            )
            continue

        with open(
            _path,
            encoding="utf-8",
        ) as _file:
            _report = json.load(_file)

        _scores = _report["metrics"]["both"]

        _report_rows.append(
            f"| {_dataset} "
            f"| {_report['test_triples_evaluated']}"
            f"/{_report['test_triples_total']} "
            f"| {_scores['mrr']:.4f} "
            f"| {_scores['mr']:.2f} "
            f"| {_scores['hits1']:.4f} "
            f"| {_scores['hits3']:.4f} "
            f"| {_scores['hits10']:.4f} |"
        )

        _found += 1

    _report_text = (
        f"### Full Test Evaluation — {evaluation_version}\n\n"
    )

    if _found:
        _report_text += "\n".join(_report_rows)
    else:
        _report_text += "No matching evaluation reports found."

    if _missing:
        _report_text += (
            "\n\n### Missing Reports\n\n"
            + "\n\n".join(_missing)
        )

    mo.output.replace(mo.md(_report_text))
    return



if __name__ == "__main__":
    app.run()
