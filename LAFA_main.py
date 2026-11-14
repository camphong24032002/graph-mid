# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "marimo>=0.24,<1",
#     "torch>=2.6,<3",
#     "wandb==0.31.0",
#     "transformers==4.57.6",
#     "pillow>=11,<13",
#     "huggingface-hub==0.36.2",
#     "gdrive-fsspec==0.4.1",
#     "python-dotenv==1.2.4",
#     "google-api-python-client==2.202.0",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    import os
    import json
    import math
    import random
    import hashlib
    import zipfile
    import shutil

    from pathlib import Path
    from contextlib import nullcontext

    import numpy as np
    import pandas as pd
    import requests
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import marimo as mo
    import wandb

    ROOT = Path.cwd() / "lafa"

    RAW_DIR = ROOT / "raw"
    PROC_DIR = ROOT / "processed"
    FEATURE_DIR = ROOT / "visual_features"
    CHECKPOINT_DIR = ROOT / "checkpoints"

    for _path in (
        RAW_DIR,
        PROC_DIR,
        FEATURE_DIR,
        CHECKPOINT_DIR,
    ):
        _path.mkdir(parents=True, exist_ok=True)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    EXPERIMENT = "lafa-paper-flow-v1"

    DRIVE_ROOT = "Master/Graph/Mid/LAFA/datasets"

    return (
        CHECKPOINT_DIR,
        DEVICE,
        DRIVE_ROOT,
        EXPERIMENT,
        F,
        FEATURE_DIR,
        PROC_DIR,
        Path,
        ROOT,
        hashlib,
        json,
        math,
        mo,
        nn,
        np,
        nullcontext,
        os,
        random,
        requests,
        shutil,
        torch,
        wandb,
        zipfile,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Setup and Configuration
    """)
    return


@app.cell
def _(DEVICE, json, np, random, torch):
    CONFIG = {
        "seed": 42,
        "dim": 100,
        "image_dim": 768,
        "num_images": 10,

        "fusion_heads": 2,
        "aggregation_heads": 2,
        "num_neighbors": 2,

        "max_train_edges": 20000,
        "node_chunk": 128,

        "decoder_h": 10,
        "decoder_w": 10,
        "conv_channels": 16,

        "dropout": 0.2,
        "label_smoothing": 0.1,

        "epochs": 6,
        "steps_per_epoch": 4,
        "batch_size": 64 if DEVICE == "cuda" else 16,

        "learning_rate": 0.001,
        "weight_decay": 1e-5,
        "patience": 5,

        "valid_count": 256,
        "eval_query_batch": 32,
        "max_test_triples": 512,

        "allow_missing_images": False,
        "train_reciprocal": True,
    }

    random.seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CONFIG["seed"])

    return (CONFIG,)


@app.cell
def _(EXPERIMENT, os, wandb):
    # W&B authentication
    wandb.login(
        key=os.environ["WANDB_API_KEY"]
    )

    # Project configuration
    WANDB_PROJECT = "lafa-reproduction"
    WANDB_ENTITY = wandb.Api().default_entity

    # Experiment metadata
    WANDB_GROUP = EXPERIMENT

    return WANDB_ENTITY, WANDB_PROJECT


@app.cell
def _(CONFIG, EXPERIMENT, json):
    WANDB_CONFIG = {
        **CONFIG,
        "paper": "LAFA",
        "conference": "AAAI 2024",
        "experiment_version": EXPERIMENT,
        "datasets": [
            "FB15k-237-IMG",
            "WN18-IMG",
        ],
        "framework": "PyTorch",
        "environment": "Molab",
    }

    return (WANDB_CONFIG,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Dataset Acquisition
    """)
    return


@app.cell
def _(DRIVE_ROOT, ROOT):
    from gdrive_fsspec import GoogleDriveFileSystem

    drive_fs = GoogleDriveFileSystem(
        use_listings_cache=False,
        skip_instance_cache=True,
        auth_kwargs={"use_local_webserver": False},
    )

    dataset_specs = {
        "FB15k-237-IMG": {
            "graph_source": "FB15k-237",
            "entities": 14541,
            "relations": 237,
            "train": 272115,
            "valid": 17535,
            "test": 20466,
            "images_per_entity": 10,
        },
        "WN18-IMG": {
            "graph_source": "wn18",
            "entities": 40943,
            "relations": 18,
            "train": 141442,
            "valid": 5000,
            "test": 5000,
            "images_per_entity": 10,
        },
    }

    dataset_root = ROOT / "datasets"
    dataset_root.mkdir(parents=True, exist_ok=True)

    image_archive_drive = (
        f"{DRIVE_ROOT}/fb15k_wn18_images.zip"
    )

    return dataset_root, dataset_specs, drive_fs


@app.cell
def _(DRIVE_ROOT, drive_fs):
    _shortcut_path = (
        f"{DRIVE_ROOT}/fb15k_wn18_images.zip"
    )

    _shortcut_info = drive_fs.info(_shortcut_path)

    _shortcut_meta = drive_fs.files.get(
        fileId=_shortcut_info["id"],
        fields=(
            "id,name,mimeType,"
            "shortcutDetails(targetId,targetMimeType,targetResourceKey)"
        ),
        supportsAllDrives=True,
    ).execute()

    _target_details = _shortcut_meta["shortcutDetails"]

    zip_target_id = _target_details["targetId"]

    zip_resource_key = _target_details.get(
        "targetResourceKey"
    )

    _target_request = drive_fs.files.get(
        fileId=zip_target_id,
        fields="id,name,mimeType,size",
        supportsAllDrives=True,
    )

    if zip_resource_key:
        _target_request.headers[
            "X-Goog-Drive-Resource-Keys"
        ] = f"{zip_target_id}/{zip_resource_key}"

    zip_target_meta = _target_request.execute()

    return zip_resource_key, zip_target_id, zip_target_meta


@app.cell
def _(
    dataset_root,
    drive_fs,
    shutil,
    zip_resource_key,
    zip_target_id,
    zip_target_meta,
    zipfile,
):
    from googleapiclient.http import MediaIoBaseDownload

    zip_local_path = (
        dataset_root / "fb15k_wn18_images.zip"
    )

    _expected_size = int(
        zip_target_meta.get("size", 0)
    )

    _free_space = shutil.disk_usage(
        dataset_root
    ).free

    if (
        not zip_local_path.exists()
        or zip_local_path.stat().st_size != _expected_size
    ):
        if _expected_size > _free_space:
            raise RuntimeError(
                "Not enough Molab disk space for ZIP."
            )

        _request = drive_fs.files.get_media(
            fileId=zip_target_id,
            supportsAllDrives=True,
        )

        if zip_resource_key:
            _request.headers[
                "X-Goog-Drive-Resource-Keys"
            ] = f"{zip_target_id}/{zip_resource_key}"

        _temp_path = zip_local_path.with_suffix(".tmp")

        with open(_temp_path, "wb") as _file:
            _downloader = MediaIoBaseDownload(
                _file,
                _request,
                chunksize=8 * 1024 * 1024,
            )

            _done = False
            _last_percent = -1

            while not _done:
                _status, _done = _downloader.next_chunk()

                if _status:
                    _percent = int(
                        _status.progress() * 100
                    )

                    if _percent // 10 != _last_percent // 10:
                        _last_percent = _percent

        _temp_path.replace(zip_local_path)


    with zipfile.ZipFile(zip_local_path) as _zip:
        _members = _zip.namelist()

        _images = [
            _name
            for _name in _members
            if _name.lower().endswith(
                (".jpg", ".jpeg", ".png", ".webp")
            )
        ]

    return (zip_local_path,)


@app.cell
def _(dataset_root, dataset_specs, requests):
    def download_graph_splits(specs, output_root):
        _base = (
            "https://raw.githubusercontent.com/"
            "DeepGraphLearning/KnowledgeGraphEmbedding/"
            "master/data"
        )

        _paths = {}

        for _dataset, _spec in specs.items():
            _folder = output_root / _dataset / "graph"
            _folder.mkdir(parents=True, exist_ok=True)

            _paths[_dataset] = {}

            for _split in ("train", "valid", "test"):
                _path = _folder / f"{_split}.txt"

                if not _path.exists():
                    _url = (
                        f"{_base}/"
                        f"{_spec['graph_source']}/"
                        f"{_split}.txt"
                    )

                    with requests.get(
                        _url,
                        timeout=120,
                        stream=True,
                    ) as _response:
                        _response.raise_for_status()

                        with open(_path, "wb") as _file:
                            for _chunk in _response.iter_content(
                                chunk_size=1024 * 1024
                            ):
                                if _chunk:
                                    _file.write(_chunk)

                with open(
                    _path,
                    encoding="utf-8",
                ) as _file:
                    _count = sum(
                        bool(_line.strip())
                        for _line in _file
                    )

                if _count != _spec[_split]:
                    raise ValueError(
                        f"{_dataset} {_split}: "
                        f"expected {_spec[_split]}, got {_count}"
                    )

                _paths[_dataset][_split] = _path


        return _paths


    raw_paths = download_graph_splits(
        dataset_specs,
        dataset_root,
    )
    return (raw_paths,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Knowledge Graph Preprocessing
    """)
    return


@app.function
def read_kg_triples(path):
    _triples = []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as _file:
        for _line in _file:
            _parts = _line.strip().split()

            if not _parts:
                continue

            if len(_parts) != 3:
                raise ValueError(
                    f"Invalid triple in {path}: {_line[:120]}"
                )

            _triples.append(tuple(_parts))

    return _triples


@app.cell
def _(dataset_specs, raw_paths, torch):
    graph_data = {}

    for _dataset, _paths in raw_paths.items():

        _raw = {
            _split: read_kg_triples(_path)
            for _split, _path in _paths.items()
        }

        _entities = sorted({
            _entity
            for _triples in _raw.values()
            for _h, _r, _t in _triples
            for _entity in (_h, _t)
        })

        _relations = sorted({
            _r
            for _triples in _raw.values()
            for _h, _r, _t in _triples
        })

        _entity2id = {
            _entity: _idx
            for _idx, _entity in enumerate(_entities)
        }

        _relation2id = {
            _relation: _idx
            for _idx, _relation in enumerate(_relations)
        }

        _splits = {}

        for _split, _triples in _raw.items():
            _splits[_split] = torch.tensor(
                [
                    [
                        _entity2id[_h],
                        _relation2id[_r],
                        _entity2id[_t],
                    ]
                    for _h, _r, _t in _triples
                ],
                dtype=torch.long,
            ).reshape(-1, 3)

        graph_data[_dataset] = {
            "num_entities": len(_entities),
            "num_relations": len(_relations),
            "entity2id": _entity2id,
            "relation2id": _relation2id,
            "id2entity": _entities,
            "id2relation": _relations,
            **_splits,
        }

        _spec = dataset_specs[_dataset]

        for _key, _actual in (
            ("entities", len(_entities)),
            ("relations", len(_relations)),
            ("train", len(_splits["train"])),
            ("valid", len(_splits["valid"])),
            ("test", len(_splits["test"])),
        ):
            if _actual != _spec[_key]:
                raise ValueError(
                    f"{_dataset} {_key}: "
                    f"expected {_spec[_key]}, got {_actual}"
                )


        for _split, _tensor in _splits.items():
            pass
    return (graph_data,)


@app.cell
def _(EXPERIMENT, PROC_DIR, graph_data, json, np, torch):
    processed_paths = {}

    for _dataset, _data in graph_data.items():
        _output = PROC_DIR / _dataset
        _output.mkdir(parents=True, exist_ok=True)

        _train_set = set(
            map(tuple, _data["train"].tolist())
        )

        _valid_set = set(
            map(tuple, _data["valid"].tolist())
        )

        _test_set = set(
            map(tuple, _data["test"].tolist())
        )

        _overlap = {
            "train_valid": len(_train_set & _valid_set),
            "train_test": len(_train_set & _test_set),
            "valid_test": len(_valid_set & _test_set),
        }

        for _split in ("train", "valid", "test"):
            np.savetxt(
                _output / f"{_split}2id.txt",
                _data[_split].numpy(),
                fmt="%d",
                delimiter="\t",
            )

        for _filename, _content in {
            "entity2id.json": _data["entity2id"],
            "relation2id.json": _data["relation2id"],
            "id2entity.json": _data["id2entity"],
            "id2relation.json": _data["id2relation"],
        }.items():
            with open(
                _output / _filename,
                "w",
                encoding="utf-8",
            ) as _file:
                json.dump(
                    _content,
                    _file,
                    ensure_ascii=False,
                    indent=2,
                )

        torch.save(
            {
                "num_entities": _data["num_entities"],
                "num_relations": _data["num_relations"],
                "train": _data["train"],
                "valid": _data["valid"],
                "test": _data["test"],
            },
            _output / "graph.pt",
        )

        _metadata = {
            "dataset": _dataset,
            "experiment": EXPERIMENT,
            "source": "DeepGraphLearning/KnowledgeGraphEmbedding",
            "num_entities": _data["num_entities"],
            "num_relations": _data["num_relations"],
            "splits": {
                _split: len(_data[_split])
                for _split in ("train", "valid", "test")
            },
            "split_overlap": _overlap,
            "split_policy": "preserve_original",
            "entity_mapping": "lexicographic_all_splits",
            "image_alignment_status": "pending",
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

        processed_paths[_dataset] = _output

    return (processed_paths,)


@app.cell
def _(
    EXPERIMENT,
    WANDB_CONFIG,
    WANDB_ENTITY,
    WANDB_PROJECT,
    json,
    processed_paths,
    wandb,
):
    with wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        job_type="preprocessing",
        name=f"lafa-preprocessing-{EXPERIMENT}",
        group=EXPERIMENT,
        tags=[
            "lafa",
            "preprocessing",
            EXPERIMENT,
        ],
        config=WANDB_CONFIG,
    ) as _run:

        for _dataset, _path in processed_paths.items():
            _name = _dataset.lower()

            with open(
                _path / "metadata.json",
                encoding="utf-8",
            ) as _file:
                _metadata = json.load(_file)

            _artifact = wandb.Artifact(
                name=f"lafa-{_name}-{EXPERIMENT}-processed",
                type="dataset",
                metadata=_metadata,
            )

            _artifact.add_dir(str(_path))

            _run.log_artifact(
                _artifact,
                aliases=["latest", "v1"],
            ).wait()

            _run.log({
                f"{_name}/entities": _metadata["num_entities"],
                f"{_name}/relations": _metadata["num_relations"],
                f"{_name}/train_triples": _metadata["splits"]["train"],
            })

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Image-to-Entity Alignment
    """)
    return


@app.cell
def _(CONFIG, PROC_DIR, graph_data, json, zip_local_path, zipfile):
    import re


    def normalize_entity_key(value):
        _value = str(value).strip().strip("<>")
        _leaf = _value.rstrip("/").split("/")[-1]

        if _value.startswith("/m/"):
            return "m." + _leaf.lower()

        if _value.startswith("/en/"):
            return "en." + _leaf.lower()

        if re.fullmatch(r"\d{8}", _leaf):
            return "n" + _leaf

        return _leaf.lower()


    def build_zip_image_manifest(
        archive_path,
        graphs,
        output_root,
        max_images=10,
    ):
        _lookups = {}
        _all_images = {}
        _archive_counts = {}

        for _dataset, _data in graphs.items():
            _lookups[_dataset] = {
                normalize_entity_key(_entity): _idx
                for _idx, _entity in enumerate(
                    _data["id2entity"]
                )
            }

            _all_images[_dataset] = {}
            _archive_counts[_dataset] = 0

        _extensions = (
            ".jpg", ".jpeg", ".png", ".webp", ".bmp"
        )

        with zipfile.ZipFile(archive_path) as _archive:
            for _member in _archive.infolist():
                if _member.is_dir():
                    continue

                _path = _member.filename.replace("\\", "/")

                if not _path.lower().endswith(_extensions):
                    continue

                _parts = _path.split("/")
                _dataset = None
                _folder_index = None

                for _idx, _part in enumerate(_parts[:-1]):
                    _name = _part.lower()

                    if (
                        _name.startswith("fb15k")
                        and "image" in _name
                    ):
                        _dataset = "FB15k-237-IMG"
                        _folder_index = _idx
                        break

                    if (
                        _name.startswith("wn18")
                        and "image" in _name
                    ):
                        _dataset = "WN18-IMG"
                        _folder_index = _idx
                        break

                if _dataset is None:
                    continue

                _archive_counts[_dataset] += 1

                if _folder_index + 1 >= len(_parts) - 1:
                    continue

                _folder_name = _parts[_folder_index + 1]
                _key = normalize_entity_key(_folder_name)

                _entity_id = _lookups[_dataset].get(_key)

                if _entity_id is None:
                    continue

                _entity_id = str(_entity_id)

                _all_images[_dataset].setdefault(
                    _entity_id, []
                ).append(_path)

        _manifests = {}
        _statistics = {}

        def _natural_key(path):
            return [
                int(_part) if _part.isdigit()
                else _part.casefold()
                for _part in re.split(r"(\d+)", path)
            ]

        for _dataset, _images in _all_images.items():
            _manifest = {
                _idx: sorted(
                    _paths,
                    key=_natural_key,
                )[:max_images]
                for _idx, _paths in _images.items()
            }

            _output = output_root / _dataset
            _output.mkdir(parents=True, exist_ok=True)

            with open(
                _output / "image_manifest.json",
                "w",
                encoding="utf-8",
            ) as _file:
                json.dump(
                    _manifest,
                    _file,
                    ensure_ascii=False,
                    indent=2,
                )

            _num_entities = graphs[_dataset]["num_entities"]

            _stats = {
                "dataset": _dataset,
                "archive": str(archive_path),
                "num_entities": _num_entities,
                "archive_image_entries": _archive_counts[_dataset],
                "entities_with_images": len(_manifest),
                "selected_images": sum(
                    len(_paths)
                    for _paths in _manifest.values()
                ),
                "entities_with_10_images": sum(
                    len(_paths) == max_images
                    for _paths in _manifest.values()
                ),
                "max_images_per_entity": max_images,
                "coverage": len(_manifest) / _num_entities,
            }

            with open(
                _output / "image_alignment.json",
                "w",
                encoding="utf-8",
            ) as _file:
                json.dump(
                    _stats,
                    _file,
                    indent=2,
                )

            _manifests[_dataset] = _manifest
            _statistics[_dataset] = _stats


        return _manifests, _statistics


    image_manifest, image_alignment_stats = (
        build_zip_image_manifest(
            archive_path=zip_local_path,
            graphs=graph_data,
            output_root=PROC_DIR,
            max_images=CONFIG["num_images"],
        )
    )
    return image_alignment_stats, image_manifest


@app.cell
def _(graph_data, image_manifest, zip_local_path, zipfile):
    from PIL import Image

    with zipfile.ZipFile(zip_local_path) as _archive:
        for _dataset, _manifest in image_manifest.items():

            for _entity_id, _paths in list(
                _manifest.items()
            )[:3]:
                _entity_name = graph_data[_dataset][
                    "id2entity"
                ][int(_entity_id)]


                with _archive.open(_paths[0]) as _file:
                    with Image.open(_file) as _image:
                        pass

    return (Image,)


@app.cell
def _(
    CONFIG,
    EXPERIMENT,
    PROC_DIR,
    WANDB_ENTITY,
    WANDB_PROJECT,
    image_alignment_stats,
    wandb,
):
    with wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        job_type="image-alignment",
        name=f"lafa-zip-image-alignment-{EXPERIMENT}",
        group=EXPERIMENT,
        tags=[
            "image-alignment",
            "zip-dataset",
            EXPERIMENT,
        ],
        config={
            "experiment": EXPERIMENT,
            "source": "fb15k_wn18_images.zip",
            "max_images_per_entity": CONFIG["num_images"],
        },
    ) as _run:

        for _dataset, _stats in image_alignment_stats.items():
            _output = PROC_DIR / _dataset
            _name = _dataset.lower()

            _run.log({
                f"{_name}/image_coverage": _stats["coverage"],
                f"{_name}/entities_with_images": (
                    _stats["entities_with_images"]
                ),
                f"{_name}/selected_images": (
                    _stats["selected_images"]
                ),
            })

            _artifact = wandb.Artifact(
                name=f"lafa-{_name}-{EXPERIMENT}-images",
                type="dataset",
                metadata=_stats,
            )

            _artifact.add_file(
                str(_output / "image_manifest.json")
            )

            _artifact.add_file(
                str(_output / "image_alignment.json")
            )

            _run.log_artifact(
                _artifact,
                aliases=["latest", "v1"],
            ).wait()

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. ViT Visual Feature Extraction
    """)
    return


@app.cell
def _(CONFIG, DEVICE, nn):
    from torchvision.models import vit_b_16, ViT_B_16_Weights

    vit_weights = ViT_B_16_Weights.IMAGENET1K_V1

    vision_encoder = vit_b_16(
        weights=vit_weights,
    )

    vision_encoder.heads = nn.Identity()
    vision_encoder = vision_encoder.to(DEVICE).eval()
    vision_encoder.requires_grad_(False)

    vision_transform = vit_weights.transforms()

    vision_model_id = "torchvision-vit-b16-imagenet1k-v1"

    return vision_encoder, vision_model_id, vision_transform


@app.cell
def _(CONFIG, DEVICE, EXPERIMENT, FEATURE_DIR, json, vision_model_id):
    vision_config = {
        "model": vision_model_id,
        "feature_dim": CONFIG["image_dim"],
        "max_images": CONFIG["num_images"],
        "batch_size": 16 if DEVICE == "cuda" else 2,
        "shard_size": 128,
        "max_shards_per_run": 32,
        "storage_dtype": "float16",
    }

    vision_root = (
        FEATURE_DIR
        / EXPERIMENT
        / vision_model_id
    )

    vision_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    return vision_config, vision_root


@app.cell
def _(
    DEVICE,
    Image,
    graph_data,
    hashlib,
    image_manifest,
    json,
    math,
    nullcontext,
    torch,
    vision_config,
    vision_encoder,
    vision_model_id,
    vision_root,
    vision_transform,
    zip_local_path,
    zipfile,
):
    def extract_vit_shards(
        dataset_name,
        max_shards=None,
    ):
        _data = graph_data[dataset_name]
        _manifest = image_manifest[dataset_name]

        _num_entities = _data["num_entities"]
        _num_images = vision_config["max_images"]
        _dim = vision_config["feature_dim"]
        _shard_size = vision_config["shard_size"]
        _batch_size = vision_config["batch_size"]

        _output = vision_root / dataset_name
        _output.mkdir(parents=True, exist_ok=True)

        _fingerprint = hashlib.sha256(
            json.dumps(
                _manifest,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()

        _metadata = {
            "dataset": dataset_name,
            "model": vision_model_id,
            "num_entities": _num_entities,
            "num_images": _num_images,
            "feature_dim": _dim,
            "shard_size": _shard_size,
            "manifest_hash": _fingerprint,
            "dtype": "float16",
        }

        _metadata_path = _output / "metadata.json"

        if _metadata_path.exists():
            with open(
                _metadata_path,
                encoding="utf-8",
            ) as _file:
                _previous = json.load(_file)

            if _previous != _metadata:
                raise RuntimeError(
                    "Existing feature cache has different "
                    "configuration or image manifest."
                )
        else:
            with open(
                _metadata_path,
                "w",
                encoding="utf-8",
            ) as _file:
                json.dump(
                    _metadata,
                    _file,
                    indent=2,
                )

        _new_shards = []
        _successful_images = 0
        _failed_images = 0

        _starts = range(
            0,
            _num_entities,
            _shard_size,
        )

        with zipfile.ZipFile(zip_local_path) as _archive:
            for _start in _starts:
                _shard_id = _start // _shard_size

                _shard_path = (
                    _output
                    / f"shard_{_shard_id:05d}.pt"
                )

                if _shard_path.exists():
                    continue

                if (
                    max_shards is not None
                    and len(_new_shards) >= max_shards
                ):
                    break

                _end = min(
                    _start + _shard_size,
                    _num_entities,
                )

                _size = _end - _start

                _features = torch.zeros(
                    (_size, _num_images, _dim),
                    dtype=torch.float16,
                )

                _mask = torch.zeros(
                    (_size, _num_images),
                    dtype=torch.bool,
                )

                _jobs = []

                for _entity_id in range(_start, _end):
                    _paths = _manifest.get(
                        str(_entity_id), []
                    )

                    for _slot, _path in enumerate(
                        _paths[:_num_images]
                    ):
                        _jobs.append(
                            (
                                _entity_id - _start,
                                _slot,
                                _path,
                            )
                        )

                _errors = []

                with torch.inference_mode():
                    for _offset in range(
                        0,
                        len(_jobs),
                        _batch_size,
                    ):
                        _batch_jobs = _jobs[
                            _offset:_offset + _batch_size
                        ]

                        _images = []
                        _positions = []

                        for _row, _slot, _path in _batch_jobs:
                            try:
                                with _archive.open(
                                    _path
                                ) as _stream:
                                    with Image.open(
                                        _stream
                                    ) as _image:
                                        _tensor = (
                                            vision_transform(
                                                _image.convert("RGB")
                                            )
                                        )

                                _images.append(_tensor)

                                _positions.append(
                                    (_row, _slot)
                                )

                            except Exception as _error:
                                _errors.append({
                                    "path": _path,
                                    "error": str(_error)[:200],
                                })

                        if not _images:
                            continue

                        _batch = torch.stack(
                            _images
                        ).to(DEVICE)

                        with (
                            torch.autocast(
                                device_type="cuda",
                                dtype=torch.float16,
                            )
                            if DEVICE == "cuda"
                            else nullcontext()
                        ):
                            _vectors = vision_encoder(
                                _batch
                            )

                        _vectors = (
                            _vectors.float()
                            .cpu()
                            .to(torch.float16)
                        )

                        _rows = [
                            _position[0]
                            for _position in _positions
                        ]

                        _slots = [
                            _position[1]
                            for _position in _positions
                        ]

                        _features[_rows, _slots] = _vectors
                        _mask[_rows, _slots] = True

                        _successful_images += len(_images)

                        del _batch, _vectors

                _failed_images += len(_errors)

                _state = {
                    "entity_ids": torch.arange(
                        _start, _end,
                        dtype=torch.long,
                    ),
                    "features": _features,
                    "mask": _mask,
                    "manifest_hash": _fingerprint,
                    "model": vision_model_id,
                }

                _temp_path = (
                    _output / f"shard_{_shard_id:05d}.tmp"
                )

                torch.save(
                    _state,
                    _temp_path,
                )

                _temp_path.replace(
                    _shard_path
                )

                if _errors:
                    with open(
                        _output
                        / f"errors_{_shard_id:05d}.json",
                        "w",
                        encoding="utf-8",
                    ) as _file:
                        json.dump(
                            _errors,
                            _file,
                            indent=2,
                        )

                _new_shards.append(
                    str(_shard_path)
                )


        _total_shards = math.ceil(
            _num_entities / _shard_size
        )

        _completed = sum(
            (
                _output / f"shard_{_idx:05d}.pt"
            ).exists()
            for _idx in range(_total_shards)
        )


        return {
            "dataset": dataset_name,
            "new_shards": _new_shards,
            "completed_shards": _completed,
            "total_shards": _total_shards,
            "successful_images": _successful_images,
            "failed_images": _failed_images,
        }

    return (extract_vit_shards,)


@app.cell
def _(graph_data, mo):
    vision_dataset_selector = mo.ui.dropdown(
        options=list(graph_data),
        value="FB15k-237-IMG",
        label="Dataset",
    )

    vit_extract_button = mo.ui.run_button(
        label="Extract ViT Features"
    )

    mo.vstack([
        vision_dataset_selector,
        vit_extract_button,
    ])
    return vision_dataset_selector, vit_extract_button


@app.cell
def _(
    EXPERIMENT,
    Path,
    WANDB_ENTITY,
    WANDB_PROJECT,
    extract_vit_shards,
    mo,
    vision_config,
    vision_dataset_selector,
    vision_model_id,
    vit_extract_button,
    wandb,
):
    mo.stop(
        not vit_extract_button.value,
        mo.md("Click **Extract ViT Features** to start."),
    )

    vit_extraction_result = extract_vit_shards(
        dataset_name=vision_dataset_selector.value,
        max_shards=vision_config["max_shards_per_run"],
    )

    if vit_extraction_result["new_shards"]:
        with wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            job_type="feature-extraction",
            name=(
                f"lafa-vit-"
                f"{vision_dataset_selector.value.lower()}-"
                f"{EXPERIMENT}"
            ),
            group=EXPERIMENT,
            tags=[
                "vit",
                "imagenet1k",
                "visual-features",
                EXPERIMENT,
            ],
            config={
                **vision_config,
                "dataset": vision_dataset_selector.value,
                "experiment": EXPERIMENT,
            },
        ) as _run:

            for _shard_path in vit_extraction_result["new_shards"]:
                _part = Path(_shard_path).stem.split("_")[-1]

                _artifact = wandb.Artifact(
                    name=(
                        f"lafa-"
                        f"{vision_dataset_selector.value.lower()}-"
                        f"{EXPERIMENT}-vit-shard-{_part}"
                    ),
                    type="dataset",
                    metadata={
                        "dataset": vision_dataset_selector.value,
                        "model": vision_model_id,
                        "shard": _part,
                    },
                )

                _artifact.add_file(
                    _shard_path
                )

                _run.log_artifact(
                    _artifact,
                    aliases=["latest"],
                ).wait()

            _run.log({
                "extraction/completed_shards": (
                    vit_extraction_result["completed_shards"]
                ),
                "extraction/total_shards": (
                    vit_extraction_result["total_shards"]
                ),
                "extraction/successful_images": (
                    vit_extraction_result["successful_images"]
                ),
                "extraction/failed_images": (
                    vit_extraction_result["failed_images"]
                ),
            })

    return


@app.cell
def _(graph_data, math, torch, vision_config, vision_root):
    def load_visual_features(dataset_name):
        _data = graph_data[dataset_name]

        _num_entities = _data["num_entities"]
        _num_images = vision_config["max_images"]
        _dim = vision_config["feature_dim"]
        _shard_size = vision_config["shard_size"]

        _folder = vision_root / dataset_name

        _num_shards = math.ceil(
            _num_entities / _shard_size
        )

        _missing = [
            _idx
            for _idx in range(_num_shards)
            if not (
                _folder / f"shard_{_idx:05d}.pt"
            ).exists()
        ]

        if _missing:
            raise RuntimeError(
                f"{dataset_name}: "
                f"{len(_missing)} feature shards missing."
            )

        _features = torch.zeros(
            (_num_entities, _num_images, _dim),
            dtype=torch.float16,
        )

        _mask = torch.zeros(
            (_num_entities, _num_images),
            dtype=torch.bool,
        )

        for _idx in range(_num_shards):
            _state = torch.load(
                _folder / f"shard_{_idx:05d}.pt",
                map_location="cpu",
                weights_only=True,
            )

            _ids = _state["entity_ids"]

            _features[_ids] = _state["features"]
            _mask[_ids] = _state["mask"]

        return {
            "images": _features,
            "mask": _mask,
            "num_entities": _num_entities,
            "feature_dim": _dim,
        }

    return (load_visual_features,)


@app.cell
def _(
    EXPERIMENT,
    Path,
    WANDB_ENTITY,
    WANDB_PROJECT,
    graph_data,
    hashlib,
    json,
    math,
    vision_config,
    vision_model_id,
    vision_root,
    wandb,
    zipfile,
):
    LAFA_VISION_ARTIFACT_VERSION = (
        f"{EXPERIMENT}-{vision_model_id}"
        f"-img{vision_config['max_images']}"
        f"-sh{vision_config['shard_size']}"
        "-fp16-r1"
    )


    def lafa_sync_visual_shards(dataset_name):
        import tempfile
        import os

        _dataset = graph_data[dataset_name]
        _folder = vision_root / dataset_name
        _folder.mkdir(parents=True, exist_ok=True)

        _n_entities = int(_dataset["num_entities"])
        _shard_size = int(vision_config["shard_size"])
        _n_shards = math.ceil(
            _n_entities / _shard_size
        )

        # Keep uploads small enough to resume across sessions
        _shards_per_artifact = 16

        _entity_fingerprint = hashlib.sha256(
            json.dumps(
                _dataset["id2entity"],
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

        _api = wandb.Api()
        _entity = WANDB_ENTITY or _api.default_entity

        _slug = dataset_name.lower()

        _run = None
        _uploaded = 0
        _restored = 0
        _reused = 0

        def _digest(_paths):
            _hash = hashlib.sha256()

            for _path in _paths:
                _hash.update(
                    _path.name.encode("utf-8")
                )

                with open(_path, "rb") as _stream:
                    while True:
                        _block = _stream.read(4 * 1024 * 1024)

                        if not _block:
                            break

                        _hash.update(_block)

            return _hash.hexdigest()

        def _available(_paths):
            return all(
                _path.is_file()
                and _path.stat().st_size > 0
                for _path in _paths
            )

        try:
            for _part, _start in enumerate(
                range(
                    0,
                    _n_shards,
                    _shards_per_artifact,
                )
            ):
                _stop = min(
                    _start + _shards_per_artifact,
                    _n_shards,
                )

                _paths = [
                    _folder / f"shard_{_idx:05d}.pt"
                    for _idx in range(_start, _stop)
                ]

                _name = (
                    f"lafa-vit-{_slug}-"
                    f"{LAFA_VISION_ARTIFACT_VERSION}"
                    f"-p{_part:03d}"
                )

                _ref = (
                    f"{_entity}/{WANDB_PROJECT}/"
                    f"{_name}:complete"
                )

                _marker = (
                    _folder
                    / (
                        f".wandb_part_{_part:03d}_"
                        f"{LAFA_VISION_ARTIFACT_VERSION}.json"
                    )
                )

                # Completed local backup from a previous session
                if _marker.exists() and _available(_paths):
                    with open(
                        _marker,
                        encoding="utf-8",
                    ) as _file:
                        _saved_marker = json.load(_file)

                    if (
                        _saved_marker.get("artifact_ref")
                        == _ref
                        and _saved_marker.get("digest")
                        == _digest(_paths)
                    ):
                        _reused += 1
                        continue

                # Check if this bundle is already on W&B
                try:
                    _remote = _api.artifact(_ref)

                except Exception as _error:
                    _message = str(_error).lower()

                    if (
                        "404" in _message
                        or "not found" in _message
                        or "does not exist" in _message
                    ):
                        _remote = None
                    else:
                        raise

                # Restore from remote if local shards are missing
                if _remote is not None:
                    _meta = _remote.metadata or {}

                    if (
                        _meta.get("dataset") != dataset_name
                        or _meta.get("entity_fingerprint")
                        != _entity_fingerprint
                        or _meta.get("shard_start") != _start
                        or _meta.get("shard_stop") != _stop
                    ):
                        raise RuntimeError(
                            f"Incompatible ViT artifact: {_ref}"
                        )

                    if not _available(_paths):
                        with tempfile.TemporaryDirectory(
                            prefix="lafa_vit_restore_"
                        ) as _tmp:
                            _downloaded = _remote.download(
                                root=_tmp
                            )

                            _zip_path = (
                                Path(_downloaded) / "shards.zip"
                            )

                            with zipfile.ZipFile(
                                _zip_path
                            ) as _zip:
                                _index = json.loads(
                                    _zip.read(
                                        "index.json"
                                    ).decode("utf-8")
                                )

                                for _path in _paths:
                                    _expected_hash = (
                                        _index["files"][
                                            _path.name
                                        ]
                                    )

                                    _temp_path = _path.with_suffix(
                                        ".pt.restore"
                                    )

                                    _hasher = hashlib.sha256()

                                    try:
                                        with (
                                            _zip.open(
                                                _path.name
                                            ) as _source,
                                            open(
                                                _temp_path, "wb"
                                            ) as _dest,
                                        ):
                                            while True:
                                                _block = _source.read(
                                                    4 * 1024 * 1024
                                                )

                                                if not _block:
                                                    break

                                                _hasher.update(
                                                    _block
                                                )

                                                _dest.write(
                                                    _block
                                                )

                                        if (
                                            _hasher.hexdigest()
                                            != _expected_hash
                                        ):
                                            raise RuntimeError(
                                                "ViT shard checksum "
                                                f"mismatch: {_path.name}"
                                            )

                                        os.replace(
                                            _temp_path,
                                            _path,
                                        )

                                    finally:
                                        _temp_path.unlink(
                                            missing_ok=True
                                        )

                        _restored += 1


                    _local_digest = _digest(_paths)

                    if (
                        _local_digest
                        != _meta.get("bundle_digest")
                    ):
                        raise RuntimeError(
                            f"Local and W&B features differ: {_ref}. "
                            "Use a new visual artifact version "
                            "if the extraction has changed."
                        )

                else:
                    # No remote backup: local bundle must be complete
                    if not _available(_paths):
                        _missing = [
                            _path.name
                            for _path in _paths
                            if (
                                not _path.exists()
                                or _path.stat().st_size == 0
                            )
                        ]

                        raise FileNotFoundError(
                            f"{dataset_name}: Missing ViT shards "
                            f"{_missing[:8]}. "
                            "No W&B backup exists for this bundle. "
                            "Complete Section 5 extraction first."
                        )

                    _local_digest = _digest(_paths)

                    # Upload this completed bundle only once
                    if _run is None:
                        _run = wandb.init(
                            project=WANDB_PROJECT,
                            entity=WANDB_ENTITY,
                            job_type="vision-feature-backup",
                            name=(
                                f"lafa-vit-backup-{_slug}-"
                                f"{LAFA_VISION_ARTIFACT_VERSION}"
                            ),
                            group=LAFA_VISION_ARTIFACT_VERSION,
                            tags=[
                                "LAFA",
                                "ViT",
                                "visual-features",
                                "reusable",
                                dataset_name,
                            ],
                            config={
                                "dataset": dataset_name,
                                "vision_model": vision_model_id,
                                "artifact_version":
                                    LAFA_VISION_ARTIFACT_VERSION,
                                "num_entities": _n_entities,
                                "num_shards": _n_shards,
                                "shard_size": _shard_size,
                                "max_images":
                                    vision_config["max_images"],
                            },
                        )

                    with tempfile.TemporaryDirectory(
                        prefix="lafa_vit_upload_"
                    ) as _tmp:
                        _archive_path = (
                            Path(_tmp) / "shards.zip"
                        )

                        _file_hashes = {}

                        with zipfile.ZipFile(
                            _archive_path,
                            "w",
                            compression=zipfile.ZIP_DEFLATED,
                            compresslevel=3,
                        ) as _zip:
                            for _path in _paths:
                                _hasher = hashlib.sha256()

                                with open(
                                    _path, "rb"
                                ) as _stream:
                                    while True:
                                        _block = _stream.read(
                                            4 * 1024 * 1024
                                        )

                                        if not _block:
                                            break

                                        _hasher.update(
                                            _block
                                        )

                                _file_hashes[
                                    _path.name
                                ] = _hasher.hexdigest()

                                _zip.write(
                                    _path,
                                    arcname=_path.name,
                                )

                            _zip.writestr(
                                "index.json",
                                json.dumps({
                                    "dataset": dataset_name,
                                    "version":
                                        LAFA_VISION_ARTIFACT_VERSION,
                                    "files": _file_hashes,
                                }),
                            )

                        _artifact = wandb.Artifact(
                            name=_name,
                            type="dataset",
                            description=(
                                "Precomputed ViT image features "
                                "for LAFA knowledge graph entities"
                            ),
                            metadata={
                                "dataset": dataset_name,
                                "vision_model": vision_model_id,
                                "vision_version":
                                    LAFA_VISION_ARTIFACT_VERSION,
                                "entity_fingerprint":
                                    _entity_fingerprint,
                                "num_entities": _n_entities,
                                "shard_start": _start,
                                "shard_stop": _stop,
                                "bundle_digest": _local_digest,
                            },
                        )

                        _artifact.add_file(
                            str(_archive_path),
                            name="shards.zip",
                        )

                        _run.log_artifact(
                            _artifact,
                            aliases=["complete", "latest"],
                        ).wait()

                    _uploaded += 1


                # Mark this bundle as fully backed up
                _marker_data = {
                    "artifact_ref": _ref,
                    "digest": _local_digest,
                    "dataset": dataset_name,
                    "part": _part,
                }

                with open(
                    _marker,
                    "w",
                    encoding="utf-8",
                ) as _file:
                    json.dump(
                        _marker_data,
                        _file,
                        indent=2,
                    )

        finally:
            if _run is not None:
                _run.finish()


        return True

    return (lafa_sync_visual_shards,)


@app.cell
def _(lafa_sync_visual_shards, load_visual_features):
    visual_data = {}

    for _dataset in (
        "FB15k-237-IMG",
        "WN18-IMG",
    ):
        try:
            lafa_sync_visual_shards(_dataset)

        except FileNotFoundError as _error:
            continue

        visual_data[_dataset] = load_visual_features(
            _dataset
        )

        _features = visual_data[_dataset]

    return (visual_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. LAFA Model Architecture
    """)
    return


@app.cell
def _(math, nn, torch):
    class LAFAImageFusion(nn.Module):
        def __init__(self, dim, image_dim, heads):
            super().__init__()

            self.dim = dim
            self.heads = heads

            self.link_projection = nn.Linear(
                3 * dim,
                heads * dim,
                bias=False,
            )

            self.visual_projection = nn.Linear(
                image_dim,
                heads * dim,
                bias=False,
            )

            self.structural_projection = nn.Linear(
                dim,
                heads * dim,
                bias=False,
            )

            self.merge = nn.Linear(
                heads * dim,
                dim,
                bias=False,
            )

        def _fuse(
            self,
            structural,
            images,
            mask,
            link_query,
        ):
            _batch_size, _num_images = images.shape[:2]

            _visual = self.visual_projection(
                images
            ).reshape(
                _batch_size,
                _num_images,
                self.heads,
                self.dim,
            )

            _logits = torch.einsum(
                "bpd,bkpd->bkp",
                link_query,
                _visual,
            ) / math.sqrt(self.dim)

            _logits = _logits.masked_fill(
                ~mask.unsqueeze(-1),
                -1e4,
            )

            _attention = torch.softmax(
                _logits,
                dim=1,
            )

            _attention = (
                _attention
                * mask.unsqueeze(-1).to(
                    _attention.dtype
                )
            )

            _attention = _attention / (
                _attention.sum(
                    dim=1,
                    keepdim=True,
                ).clamp_min(1e-12)
            )

            _pooled = torch.einsum(
                "bkp,bkpd->bpd",
                _attention,
                _visual,
            )

            _structural = (
                self.structural_projection(
                    structural
                ).reshape(
                    _batch_size,
                    self.heads,
                    self.dim,
                )
            )

            _fused = torch.sigmoid(
                _structural + _pooled
            )

            return self.merge(
                _fused.flatten(1)
            )

        def forward(
            self,
            head,
            relation,
            tail,
            head_images,
            tail_images,
            head_mask,
            tail_mask,
        ):
            # Eq. 2: triple-conditioned attention
            _context = torch.cat(
                [head, relation, tail],
                dim=-1,
            )

            _link_query = self.link_projection(
                _context
            ).reshape(
                len(head),
                self.heads,
                self.dim,
            )

            # Eq. 3–4: head image attention
            _fused_head = self._fuse(
                head,
                head_images,
                head_mask,
                _link_query,
            )

            # Eq. 5–6: tail image attention
            _fused_tail = self._fuse(
                tail,
                tail_images,
                tail_mask,
                _link_query,
            )

            # Eq. 7–11: visual + structural fusion
            return _fused_head, _fused_tail

    return (LAFAImageFusion,)


@app.cell
def _(nn):
    class LAFANeighborAggregation(nn.Module):
        def __init__(self, dim, heads):
            super().__init__()

            self.dim = dim
            self.heads = heads

            # Eq. 12: structural query and key
            self.query_projection = nn.Linear(
                dim,
                heads * dim,
                bias=False,
            )

            self.key_projection = nn.Linear(
                dim,
                heads * dim,
                bias=False,
            )

            # Eq. 15: fused head-tail message
            self.message_projection = nn.Linear(
                2 * dim,
                heads * dim,
                bias=False,
            )

            # Eq. 16: multi-head aggregation
            self.merge = nn.Linear(
                heads * dim,
                dim,
                bias=False,
            )

            # Molab adaptation:
            # fallback for entities without outgoing edges
            self.isolated_projection = nn.Linear(
                dim,
                dim,
                bias=False,
            )

    return (LAFANeighborAggregation,)


@app.cell
def _(F, nn, torch):
    class LAFAConvE(nn.Module):
        def __init__(
            self,
            dim,
            h,
            w,
            conv_channels,
            dropout,
            num_entities,
        ):
            super().__init__()

            if h * w != dim:
                raise ValueError(
                    "decoder_h * decoder_w must equal dim"
                )

            self.h = h
            self.w = w

            self.input_bn = nn.BatchNorm2d(1)

            self.convolution = nn.Conv2d(
                in_channels=1,
                out_channels=conv_channels,
                kernel_size=3,
                padding=1,
            )

            self.conv_bn = nn.BatchNorm2d(
                conv_channels
            )

            self.conv_dropout = nn.Dropout2d(
                dropout
            )

            self.linear = nn.Linear(
                conv_channels * (2 * h) * w,
                dim,
            )

            self.output_bn = nn.BatchNorm1d(dim)

            self.hidden_dropout = nn.Dropout(
                dropout
            )

            self.entity_bias = nn.Parameter(
                torch.zeros(num_entities)
            )

        def query(self, head, relation):
            _batch_size = len(head)

            _head = head.reshape(
                _batch_size,
                1,
                self.h,
                self.w,
            )

            _relation = relation.reshape(
                _batch_size,
                1,
                self.h,
                self.w,
            )

            # Stack head and relation embeddings
            _x = torch.cat(
                [_head, _relation],
                dim=2,
            )

            _x = self.input_bn(_x)

            _x = self.convolution(_x)

            _x = F.relu(
                self.conv_bn(_x)
            )

            _x = self.conv_dropout(_x)

            _x = self.linear(
                _x.flatten(1)
            )

            _x = F.relu(
                self.output_bn(_x)
            )

            _x = self.hidden_dropout(_x)

            return _x

    return (LAFAConvE,)


@app.cell
def _(LAFAConvE, LAFAImageFusion, LAFANeighborAggregation, math, nn, torch):
    class LAFA(nn.Module):
        def __init__(
            self,
            num_entities,
            num_relations,
            dim,
            image_dim,
            fusion_heads,
            aggregation_heads,
            decoder_h,
            decoder_w,
            conv_channels,
            dropout,
        ):
            super().__init__()

            # 1. Structural embeddings
            self.entity_embeddings = nn.Embedding(
                num_entities,
                dim,
            )

            self.relation_embeddings = nn.Embedding(
                num_relations,
                dim,
            )

            # 2. Modality interaction + Link-aware fusion
            self.link_fusion = LAFAImageFusion(
                dim=dim,
                image_dim=image_dim,
                heads=fusion_heads,
            )

            # 3. Link-aware aggregation
            self.link_aggregation = (
                LAFANeighborAggregation(
                    dim=dim,
                    heads=aggregation_heads,
                )
            )

            # 4. ConvE decoder
            self.decoder = LAFAConvE(
                dim=dim,
                h=decoder_h,
                w=decoder_w,
                conv_channels=conv_channels,
                dropout=dropout,
                num_entities=num_entities,
            )

            nn.init.xavier_uniform_(
                self.entity_embeddings.weight
            )

            nn.init.xavier_uniform_(
                self.relation_embeddings.weight
            )

        def _edge_message(
            self,
            head_ids,
            relation_ids,
            tail_ids,
            images,
            mask,
        ):
            _device = (
                self.entity_embeddings.weight.device
            )

            # Structural embeddings of each triple
            _head = self.entity_embeddings(
                head_ids
            )

            _relation = self.relation_embeddings(
                relation_ids
            )

            _tail = self.entity_embeddings(
                tail_ids
            )

            # Visual features from pretrained ViT
            _head_images = images.index_select(
                0,
                head_ids.to(images.device),
            ).to(
                device=_device,
                dtype=_head.dtype,
            )

            _tail_images = images.index_select(
                0,
                tail_ids.to(images.device),
            ).to(
                device=_device,
                dtype=_head.dtype,
            )

            _head_mask = mask.index_select(
                0,
                head_ids.to(mask.device),
            ).to(_device)

            _tail_mask = mask.index_select(
                0,
                tail_ids.to(mask.device),
            ).to(_device)

            # MIA + LAF, separately for each link
            _fused_head, _fused_tail = (
                self.link_fusion(
                    head=_head,
                    relation=_relation,
                    tail=_tail,
                    head_images=_head_images,
                    tail_images=_tail_images,
                    head_mask=_head_mask,
                    tail_mask=_tail_mask,
                )
            )

            # Eq. 15:
            # b_j = [fused_head || fused_tail] W_b
            _messages = (
                self.link_aggregation.message_projection(
                    torch.cat(
                        [_fused_head, _fused_tail],
                        dim=-1,
                    )
                )
            )

            return _messages.reshape(
                len(head_ids),
                self.link_aggregation.heads,
                self.link_aggregation.dim,
            )

        def encode_all(
            self,
            train_triples,
            visual_features,
            edge_chunk_size=1024,
            use_checkpoint=True,
        ):
            from torch.utils.checkpoint import (
                checkpoint as _checkpoint
            )

            _device = (
                self.entity_embeddings.weight.device
            )

            _edges = train_triples.to(
                device=_device,
                dtype=torch.long,
            )

            _structural = (
                self.entity_embeddings.weight
            )

            _num_entities, _dim = (
                _structural.shape
            )

            _num_heads = (
                self.link_aggregation.heads
            )

            if len(_edges) == 0:
                return (
                    self.link_aggregation
                    .isolated_projection(_structural)
                )

            _src, _rel, _dst = (
                _edges.unbind(dim=1)
            )

            # -------------------------------------
            # Eq. 12: structural query/key
            # -------------------------------------
            _query = (
                self.link_aggregation.query_projection(
                    _structural
                ).reshape(
                    _num_entities,
                    _num_heads,
                    _dim,
                )
            )

            _key = (
                self.link_aggregation.key_projection(
                    _structural
                ).reshape(
                    _num_entities,
                    _num_heads,
                    _dim,
                )
            )

            # -------------------------------------
            # Eq. 13: neighbor attention logits
            # -------------------------------------
            _logits = (
                _query[_src] * _key[_dst]
            ).sum(
                dim=-1
            ) / math.sqrt(_dim)

            _src_index = (
                _src.unsqueeze(1).expand(
                    -1,
                    _num_heads,
                )
            )

            # -------------------------------------
            # Eq. 14: grouped softmax
            # -------------------------------------
            _group_max = torch.full(
                (_num_entities, _num_heads),
                -torch.inf,
                dtype=_logits.dtype,
                device=_device,
            )

            _group_max.scatter_reduce_(
                0,
                _src_index,
                _logits.detach(),
                reduce="amax",
                include_self=True,
            )

            _weights = torch.exp(
                _logits - _group_max[_src]
            )

            _group_sum = torch.zeros(
                (_num_entities, _num_heads),
                dtype=_weights.dtype,
                device=_device,
            )

            _group_sum.index_add_(
                0,
                _src,
                _weights,
            )

            _weights = (
                _weights
                / _group_sum[_src].clamp_min(1e-12)
            )

            # -------------------------------------
            # Eq. 15: aggregate all neighbor edges
            # -------------------------------------
            _aggregate = torch.zeros(
                (
                    _num_entities,
                    _num_heads,
                    _dim,
                ),
                dtype=_structural.dtype,
                device=_device,
            )

            _images = visual_features["images"]
            _mask = visual_features["mask"]

            for _start in range(
                0,
                len(_src),
                edge_chunk_size,
            ):
                _end = min(
                    _start + edge_chunk_size,
                    len(_src),
                )

                _h = _src[_start:_end]
                _r = _rel[_start:_end]
                _t = _dst[_start:_end]

                if self.training and use_checkpoint:
                    _messages = _checkpoint(
                        self._edge_message,
                        _h,
                        _r,
                        _t,
                        _images,
                        _mask,
                        use_reentrant=False,
                    )
                else:
                    _messages = self._edge_message(
                        _h,
                        _r,
                        _t,
                        _images,
                        _mask,
                    )

                _weighted_messages = (
                    _messages
                    * _weights[
                        _start:_end
                    ].unsqueeze(-1)
                )

                _aggregate.index_add_(
                    0,
                    _h,
                    _weighted_messages,
                )

            # -------------------------------------
            # Eq. 16: multi-head aggregation
            # -------------------------------------
            _encoded = (
                self.link_aggregation.merge(
                    _aggregate.flatten(1)
                )
            )

            # -------------------------------------
            # Handle entities without outgoing edges
            # Implementation adaptation
            # -------------------------------------
            _degree = torch.bincount(
                _src,
                minlength=_num_entities,
            )

            _fallback = (
                self.link_aggregation
                .isolated_projection(
                    _structural
                )
            )

            return torch.where(
                (_degree > 0).unsqueeze(-1),
                _encoded,
                _fallback,
            )

        def score_all(
            self,
            heads,
            relations,
            embeddings,
        ):
            # ConvE query: (h, r, ?)
            _query = self.decoder.query(
                embeddings[heads],
                self.relation_embeddings(
                    relations
                ),
            )

            # 1-vs-All candidate scores
            _logits = (
                _query @ embeddings.T
                + self.decoder.entity_bias
            )

            return _logits

    return (LAFA,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Graph Neighborhood Preparation
    """)
    return


@app.cell
def _(PROC_DIR, graph_data, torch):
    neighbor_data = {}

    for _dataset, _data in graph_data.items():
        _train = _data["train"].long().cpu()
        _num_entities = _data["num_entities"]

        # Sort triples by head entity
        _order = torch.argsort(
            _train[:, 0],
            stable=True,
        )

        _sorted = _train[_order]

        _degrees = torch.bincount(
            _sorted[:, 0],
            minlength=_num_entities,
        )

        _indptr = torch.zeros(
            _num_entities + 1,
            dtype=torch.long,
        )

        _indptr[1:] = torch.cumsum(
            _degrees,
            dim=0,
        )

        neighbor_data[_dataset] = {
            "indptr": _indptr,
            "neighbor_ids": _sorted[:, 2].contiguous(),
            "relation_ids": _sorted[:, 1].contiguous(),
            "degrees": _degrees,
        }

        _output = PROC_DIR / _dataset

        torch.save(
            neighbor_data[_dataset],
            _output / "adjacency.pt",
        )

    return (neighbor_data,)


@app.cell
def _(CONFIG, neighbor_data, torch):
    def sample_neighbors(
        dataset_name,
        entity_ids,
        num_neighbors=None,
        generator=None,
    ):
        _graph = neighbor_data[dataset_name]

        _ids = torch.as_tensor(
            entity_ids,
            dtype=torch.long,
        ).reshape(-1).cpu()

        _count = (
            CONFIG["num_neighbors"]
            if num_neighbors is None
            else int(num_neighbors)
        )

        _batch_size = len(_ids)

        _neighbors = torch.zeros(
            (_batch_size, _count),
            dtype=torch.long,
        )

        _relations = torch.zeros_like(_neighbors)

        _mask = torch.zeros(
            (_batch_size, _count),
            dtype=torch.bool,
        )

        _indptr = _graph["indptr"]

        for _row, _entity in enumerate(_ids.tolist()):
            _start = int(_indptr[_entity])
            _end = int(_indptr[_entity + 1])

            _degree = _end - _start

            if _degree == 0:
                continue

            if _degree <= _count:
                _positions = torch.arange(
                    _start,
                    _end,
                )

            elif generator is None:
                # Deterministic sampling for evaluation
                _positions = (
                    _start
                    + torch.arange(_count) * _degree // _count
                )

            else:
                # Random sampling for training
                _positions = (
                    _start
                    + torch.randperm(
                        _degree,
                        generator=generator,
                    )[:_count]
                )

            _n = len(_positions)

            _neighbors[_row, :_n] = (
                _graph["neighbor_ids"][_positions]
            )

            _relations[_row, :_n] = (
                _graph["relation_ids"][_positions]
            )

            _mask[_row, :_n] = True

        return {
            "neighbor_ids": _neighbors,
            "neighbor_relations": _relations,
            "neighbor_mask": _mask,
        }

    return (sample_neighbors,)


@app.cell
def _(PROC_DIR, graph_data, torch):
    query_data = {}

    for _dataset, _data in graph_data.items():
        _target_sets = {}

        for _h, _r, _t in _data["train"].tolist():
            _target_sets.setdefault(
                (_h, _r),
                set(),
            ).add(_t)

        _query_keys = sorted(_target_sets)

        _target_values = []
        _target_offsets = [0]

        for _key in _query_keys:
            _tails = sorted(_target_sets[_key])

            _target_values.extend(_tails)
            _target_offsets.append(
                len(_target_values)
            )

        _queries = torch.tensor(
            _query_keys,
            dtype=torch.long,
        ).reshape(-1, 2)

        _targets = torch.tensor(
            _target_values,
            dtype=torch.long,
        )

        _offsets = torch.tensor(
            _target_offsets,
            dtype=torch.long,
        )

        query_data[_dataset] = {
            "queries": _queries,
            "target_ids": _targets,
            "target_ptr": _offsets,
        }

        _output = PROC_DIR / _dataset

        torch.save(
            query_data[_dataset],
            _output / "train_queries.pt",
        )

    return (query_data,)


@app.cell
def _(graph_data, query_data, sample_neighbors, torch):
    def prepare_lafa_batch(
        dataset_name,
        query_indices,
        generator=None,
        num_neighbors=None,
    ):
        _data = query_data[dataset_name]

        _indices = torch.as_tensor(
            query_indices,
            dtype=torch.long,
        ).reshape(-1).cpu()

        _queries = _data["queries"].index_select(
            0,
            _indices,
        )

        _num_entities = graph_data[
            dataset_name
        ]["num_entities"]

        _labels = torch.zeros(
            (len(_indices), _num_entities),
            dtype=torch.float32,
        )

        _offsets = _data["target_ptr"]
        _target_ids = _data["target_ids"]

        for _row, _idx in enumerate(_indices.tolist()):
            _start = int(_offsets[_idx])
            _end = int(_offsets[_idx + 1])

            _positives = _target_ids[
                _start:_end
            ]

            _labels[_row, _positives] = 1.0

        _neighbors = sample_neighbors(
            dataset_name=dataset_name,
            entity_ids=_queries[:, 0],
            num_neighbors=num_neighbors,
            generator=generator,
        )

        return {
            "head_ids": _queries[:, 0],
            "relation_ids": _queries[:, 1],
            "labels": _labels,
            **_neighbors,
        }

    return (prepare_lafa_batch,)


@app.cell
def _(
    CONFIG,
    EXPERIMENT,
    PROC_DIR,
    WANDB_ENTITY,
    WANDB_PROJECT,
    graph_data,
    neighbor_data,
    query_data,
    wandb,
):
    with wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        job_type="graph-preparation",
        name=f"lafa-graph-neighborhoods-{EXPERIMENT}",
        group=EXPERIMENT,
        tags=[
            "lafa",
            "graph-neighborhoods",
            "training-queries",
            EXPERIMENT,
        ],
        config={
            "experiment": EXPERIMENT,
            "neighbor_sampling": "train_outgoing",
            "num_neighbors": CONFIG["num_neighbors"],
            "query_policy": "multi_positive",
        },
    ) as _run:

        for _dataset in graph_data:
            _output = PROC_DIR / _dataset
            _name = _dataset.lower()

            _artifact = wandb.Artifact(
                name=(
                    f"lafa-{_name}-"
                    f"{EXPERIMENT}-graph-preparation"
                ),
                type="dataset",
                metadata={
                    "dataset": _dataset,
                    "num_entities": graph_data[
                        _dataset
                    ]["num_entities"],
                    "num_queries": len(
                        query_data[_dataset]["queries"]
                    ),
                    "num_neighbors": CONFIG["num_neighbors"],
                },
            )

            _artifact.add_file(
                str(_output / "adjacency.pt")
            )

            _artifact.add_file(
                str(_output / "train_queries.pt")
            )

            _run.log_artifact(
                _artifact,
                aliases=["latest", "v1"],
            ).wait()

            _run.log({
                f"{_name}/num_queries": len(
                    query_data[_dataset]["queries"]
                ),
                f"{_name}/num_edges": len(
                    neighbor_data[_dataset]["neighbor_ids"]
                ),
            })

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. LAFA Training Pipeline
    """)
    return


@app.cell
def _(CHECKPOINT_DIR, CONFIG, DEVICE, torch):
    training_version = (
        "lafa-v4-paper-reproduction-v4-longrun"
    )

    training_config = {
        "epochs": 40,
        "steps_per_epoch": 30,

        "batch_size": 16,
        "accumulation_steps": 32,

        "candidate_mode": "full",

        "edge_chunk_size": (
            1024 if torch.device(DEVICE).type == "cuda"
            else 128
        ),
        "use_gradient_checkpointing": True,

        "label_smoothing": 0.1,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "grad_clip": 1.0,

        "valid_samples": 256,
        "validate_every": 2,
        "valid_batch_size": (
            16 if torch.device(DEVICE).type == "cuda"
            else 4
        ),
        "head_query_batch_size": (
            256 if torch.device(DEVICE).type == "cuda"
            else 32
        ),
        "tie_tolerance": 1e-5,

        "patience": 12,
        "seed": CONFIG["seed"],

        "resume": True,
        "reciprocal": False,

        # Save after each optimizer update
        "checkpoint_every_steps": 1,
    }

    training_root = CHECKPOINT_DIR / training_version
    training_root.mkdir(parents=True, exist_ok=True)


    return training_config, training_root, training_version


@app.cell
def _(sample_neighbors, torch, training_config):
    def encode_lafa_entities(
        model,
        dataset_name,
        entity_ids,
        visual_features,
        generator=None,
        chunk_size=32,
    ):
        _ids = torch.as_tensor(
            entity_ids,
            dtype=torch.long,
        ).reshape(-1).cpu()

        _outputs = []

        for _chunk in _ids.split(chunk_size):
            _neighbors = sample_neighbors(
                dataset_name=dataset_name,
                entity_ids=_chunk,
                num_neighbors=training_config["num_neighbors"],
                generator=generator,
            )

            _encoded = model.encode_entities(
                entity_ids=_chunk,
                neighbor_ids=_neighbors["neighbor_ids"],
                neighbor_relations=_neighbors["neighbor_relations"],
                neighbor_mask=_neighbors["neighbor_mask"],
                image_features=visual_features["images"],
                image_mask=visual_features["mask"],
            )

            _outputs.append(_encoded)

        return torch.cat(_outputs, dim=0)


    def select_lafa_candidates(
        batch_labels,
        num_entities,
        pool_size,
        generator,
    ):
        # Include every positive entity in the batch.
        _positives = torch.nonzero(
            batch_labels.any(dim=0),
            as_tuple=False,
        ).flatten()

        _count = max(
            pool_size,
            len(_positives),
        )

        _count = min(
            _count,
            num_entities,
        )

        _permutation = torch.randperm(
            num_entities,
            generator=generator,
        )

        _negative_pool = _permutation[
            ~torch.isin(
                _permutation,
                _positives,
            )
        ]

        _negatives = _negative_pool[
            :_count - len(_positives)
        ]

        return torch.cat(
            [_positives, _negatives]
        ).sort().values

    return


@app.cell
def _(graph_data, torch):
    def build_lafa_validation_filters(dataset_name):
        _heads = {}
        _tails = {}

        for _split in ("train", "valid"):
            for _h, _r, _t in graph_data[
                dataset_name
            ][_split].tolist():
                _heads.setdefault(
                    (_r, _t), set()
                ).add(_h)

                _tails.setdefault(
                    (_h, _r), set()
                ).add(_t)

        return _heads, _tails


    @torch.inference_mode()
    def rank_lafa_paper(
        model,
        embeddings,
        triples,
        head_filter,
        tail_filter,
        batch_size=16,
        head_query_batch_size=256,
        tolerance=1e-5,
    ):
        model.eval()

        _device = embeddings.device
        _triples = triples.long().cpu()

        if not len(_triples):
            raise ValueError("Empty evaluation triples.")

        def _filtered_rank(
            scores,
            batch,
            targets,
            lookup,
            direction,
        ):
            _gold = scores.gather(
                1,
                targets.unsqueeze(1),
            ).clone()

            for _row, (_h, _r, _t) in enumerate(
                batch.tolist()
            ):
                _key = (
                    (_r, _t)
                    if direction == "head"
                    else (_h, _r)
                )

                _other = [
                    _entity
                    for _entity in lookup.get(_key, ())
                    if _entity != int(targets[_row])
                ]

                if _other:
                    scores[
                        _row,
                        torch.tensor(
                            _other,
                            device=_device,
                            dtype=torch.long,
                        ),
                    ] = -torch.inf

            scores.scatter_(
                1,
                targets.unsqueeze(1),
                -torch.inf,
            )

            return (
                1.0
                + (scores > _gold + tolerance)
                .sum(dim=1).float()
                + 0.5 * (
                    (scores - _gold).abs() <= tolerance
                ).sum(dim=1).float()
            ).cpu()

        _tail_ranks = []

        for _batch_cpu in _triples.split(batch_size):
            _batch = _batch_cpu.to(_device)
            _h, _r, _t = _batch.unbind(1)

            _query = model.decoder.query(
                embeddings[_h],
                model.relation_embeddings(_r),
            )

            _scores = (
                _query @ embeddings.T
                + model.decoder.entity_bias
            )

            _tail_ranks.append(
                _filtered_rank(
                    _scores,
                    _batch_cpu,
                    _t,
                    tail_filter,
                    "tail",
                )
            )

        _head_ranks = []

        for _relation in torch.unique(
            _triples[:, 1]
        ).tolist():
            _position = torch.nonzero(
                _triples[:, 1] == _relation
            ).flatten()

            _relation_vector = model.relation_embeddings(
                torch.tensor([_relation], device=_device)
            )

            _candidate_queries = []

            for _entity_chunk in embeddings.split(
                head_query_batch_size
            ):
                _candidate_queries.append(
                    model.decoder.query(
                        _entity_chunk,
                        _relation_vector.expand(
                            len(_entity_chunk), -1
                        ),
                    )
                )

            _all_head_queries = torch.cat(
                _candidate_queries,
                dim=0,
            )

            for _positions in _position.split(batch_size):
                _batch_cpu = _triples[_positions]
                _batch = _batch_cpu.to(_device)

                _h, _r, _t = _batch.unbind(1)

                _scores = (
                    embeddings[_t]
                    @ _all_head_queries.T
                )

                _head_ranks.append(
                    _filtered_rank(
                        _scores,
                        _batch_cpu,
                        _h,
                        head_filter,
                        "head",
                    )
                )

        _head_ranks = torch.cat(_head_ranks)
        _tail_ranks = torch.cat(_tail_ranks)

        def _metrics(ranks):
            return {
                "mr": ranks.mean().item(),
                "mrr": (1.0 / ranks).mean().item(),
                "hits1": (ranks <= 1).float().mean().item(),
                "hits3": (ranks <= 3).float().mean().item(),
                "hits10": (ranks <= 10).float().mean().item(),
            }

        return {
            "head": _metrics(_head_ranks),
            "tail": _metrics(_tail_ranks),
            "both": _metrics(
                torch.cat([_head_ranks, _tail_ranks])
            ),
        }

    return build_lafa_validation_filters, rank_lafa_paper


@app.cell
def _(
    CONFIG,
    DEVICE,
    F,
    LAFA,
    WANDB_ENTITY,
    WANDB_PROJECT,
    build_lafa_validation_filters,
    graph_data,
    rank_lafa_paper,
    torch,
    training_config,
    training_root,
    training_version,
    vision_model_id,
    visual_data,
    wandb,
):
    def train_lafa(dataset_name):
        _data = graph_data[dataset_name]
        _visual = visual_data[dataset_name]
        _config = training_config
        _device = torch.device(DEVICE)

        _train = _data["train"].long().cpu()
        _valid = _data["valid"].long().cpu()

        _num_entities = _data["num_entities"]
        _num_relations = _data["num_relations"]

        _micro_size = _config["batch_size"]
        _accum_steps = _config["accumulation_steps"]
        _effective_size = _micro_size * _accum_steps

        _rng = torch.Generator().manual_seed(
            _config["seed"]
        )

        torch.manual_seed(_config["seed"])

        if _device.type == "cuda":
            torch.cuda.manual_seed_all(_config["seed"])

        _output = training_root / dataset_name
        _output.mkdir(parents=True, exist_ok=True)

        _last_path = _output / "last.pt"
        _best_path = _output / "best.pt"

        _artifact_prefix = (
            f"lafa-{dataset_name.lower()}-"
            f"{training_version}"
        )

        # 1. Restore last.pt from W&B after restart
        if _config["resume"] and not _last_path.exists():
            _api = wandb.Api()
            _entity = WANDB_ENTITY or _api.default_entity

            _resume_ref = (
                f"{_entity}/{WANDB_PROJECT}/"
                f"{_artifact_prefix}-resume:latest"
            )

            try:

                _remote = _api.artifact(_resume_ref)
                _remote.download(root=str(_output))


            except Exception as _error:
                pass

        # Avoid silently starting over when only best.pt exists
        if (
            _config["resume"]
            and not _last_path.exists()
            and _best_path.exists()
        ):
            raise RuntimeError(
                "best.pt exists but last.pt is missing. "
                "Use Section 9 for evaluation; "
                "optimizer state cannot be resumed from best.pt."
            )

        # 2. Model
        _model = LAFA(
            num_entities=_num_entities,
            num_relations=_num_relations,
            dim=CONFIG["dim"],
            image_dim=CONFIG["image_dim"],
            fusion_heads=CONFIG["fusion_heads"],
            aggregation_heads=CONFIG["aggregation_heads"],
            decoder_h=CONFIG["decoder_h"],
            decoder_w=CONFIG["decoder_w"],
            conv_channels=CONFIG["conv_channels"],
            dropout=CONFIG["dropout"],
        ).to(_device)

        _optimizer = torch.optim.Adam(
            _model.parameters(),
            lr=_config["learning_rate"],
            weight_decay=_config["weight_decay"],
        )

        # 3. Training labels
        _known_tails = {}

        for _h, _r, _t in _train.tolist():
            _known_tails.setdefault(
                (_h, _r), set()
            ).add(_t)

        _known_tails = {
            _key: sorted(_targets)
            for _key, _targets in _known_tails.items()
        }

        # 4. Validation
        _head_filter, _tail_filter = (
            build_lafa_validation_filters(dataset_name)
        )

        _valid_limit = _config["valid_samples"]

        if (
            _valid_limit is not None
            and _valid_limit < len(_valid)
        ):
            _valid_rng = torch.Generator().manual_seed(
                _config["seed"] + 1000
            )

            _valid_indices = torch.randperm(
                len(_valid),
                generator=_valid_rng,
            )[:_valid_limit]

            _valid_batch = _valid[_valid_indices]
        else:
            _valid_batch = _valid

        # 5. Training state
        _history = []
        _best_mrr = -1.0
        _best_epoch = 0
        _stale = 0
        _global_step = 0
        _start_epoch = 1

        _resume_indices = None
        _resume_step = 0
        _resume_loss_sum = 0.0
        _resume_updates = 0
        _finished = False

        # 6. Load local or restored checkpoint
        if _config["resume"] and _last_path.exists():
            _state = torch.load(
                _last_path,
                map_location="cpu",
                weights_only=True,
            )

            if (
                _state.get("training_version")
                != training_version
                or _state.get("training_mode")
                != "paper_aligned_accumulation"
            ):
                raise RuntimeError(
                    "Checkpoint version or mode mismatch."
                )

            _previous_config = _state["config"]

            for _key in (
                "batch_size",
                "accumulation_steps",
                "steps_per_epoch",
                "edge_chunk_size",
                "label_smoothing",
                "learning_rate",
            ):
                if (
                    _previous_config.get(_key)
                    != _config.get(_key)
                ):
                    raise RuntimeError(
                        f"Resume config mismatch: {_key}"
                    )

            _model.load_state_dict(
                _state["model_state_dict"]
            )

            _optimizer.load_state_dict(
                _state["optimizer_state_dict"]
            )

            _history = _state["history"]
            _best_mrr = _state["best_mrr"]
            _best_epoch = _state["best_epoch"]
            _stale = _state["stale"]
            _global_step = _state["global_step"]

            _rng.set_state(
                _state["generator_state"].cpu()
            )

            if "torch_rng_state" in _state:
                torch.set_rng_state(
                    _state["torch_rng_state"].cpu()
                )

            if (
                _device.type == "cuda"
                and _state.get("cuda_rng_state_all") is not None
            ):
                torch.cuda.set_rng_state_all(
                    _state["cuda_rng_state_all"]
                )

            if _state.get("epoch_complete", True):
                _start_epoch = _state["epoch"] + 1
            else:
                _start_epoch = _state["epoch"]
                _resume_indices = _state["epoch_indices"]
                _resume_step = _state["next_step"]
                _resume_loss_sum = _state["epoch_loss_sum"]
                _resume_updates = _state["epoch_updates"]

            _finished = (
                _state.get("finished", False)
                or (
                    _state.get("epoch_complete", True)
                    and (
                        _state["epoch"] >= _config["epochs"]
                        or _stale >= _config["patience"]
                    )
                )
            )


        # 7. Do not train again if experiment finished
        if _finished or _start_epoch > _config["epochs"]:

            return {
                "history": _history,
                "best_mrr": _best_mrr,
                "best_epoch": _best_epoch,
                "global_step": _global_step,
                "checkpoint": str(_best_path),
                "training_version": training_version,
                "finished": True,
            }

        # 8. Save checkpoint locally
        def _save_checkpoint(
            epoch,
            next_step,
            epoch_complete,
            epoch_indices,
            epoch_loss_sum,
            epoch_updates,
            is_best=False,
            finished=False,
        ):
            _state_to_save = {
                "epoch": epoch,
                "next_step": next_step,
                "epoch_complete": epoch_complete,
                "epoch_indices": (
                    None if epoch_complete
                    else epoch_indices
                ),
                "epoch_loss_sum": epoch_loss_sum,
                "epoch_updates": epoch_updates,
                "global_step": _global_step,
                "finished": finished,

                "dataset": dataset_name,
                "training_version": training_version,
                "training_mode": "paper_aligned_accumulation",
                "candidate_mode": "full",
                "reciprocal": False,
                "num_relations": _num_relations,
                "effective_batch_size": _effective_size,

                "model_state_dict": _model.state_dict(),
                "optimizer_state_dict": (
                    _optimizer.state_dict()
                ),

                "generator_state": _rng.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state_all": (
                    torch.cuda.get_rng_state_all()
                    if _device.type == "cuda"
                    else None
                ),

                "history": _history,
                "best_epoch": _best_epoch,
                "best_mrr": _best_mrr,
                "stale": _stale,
                "config": dict(_config),
            }

            _temp_path = _output / "last.tmp"

            torch.save(_state_to_save, _temp_path)
            _temp_path.replace(_last_path)

            if is_best:
                _best_temp = _output / "best.tmp"

                torch.save(
                    _state_to_save,
                    _best_temp,
                )

                _best_temp.replace(_best_path)

        _visual_coverage = float(
            _visual["mask"].any(dim=1)
            .float().mean().item()
        )

        # 9. Initialize W&B
        with wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            job_type="training",
            name=f"lafa-{dataset_name.lower()}-{training_version}",
            group=training_version,
            tags=[
                "LAFA",
                "paper-reproduction",
                "gradient-accumulation",
                "full-bce",
                "wandb-checkpoint",
                dataset_name,
            ],
            config={
                **_config,
                "dataset": dataset_name,
                "training_version": training_version,
                "training_mode": "paper_aligned_accumulation",
                "effective_batch_size": _effective_size,
                "visual_coverage": _visual_coverage,
                "visual_encoder": vision_model_id,
            },
        ) as _run:

            # 10. Upload checkpoint to W&B
            def _sync_checkpoint(_path, _kind):
                if not _path.exists():
                    raise FileNotFoundError(str(_path))

                _artifact = wandb.Artifact(
                    name=f"{_artifact_prefix}-{_kind}",
                    type="model",
                    metadata={
                        "dataset": dataset_name,
                        "training_version": training_version,
                        "training_mode": (
                            "paper_aligned_accumulation"
                        ),
                        "global_step": _global_step,
                        "best_epoch": _best_epoch,
                        "best_mrr": _best_mrr,
                    },
                )

                _artifact.add_file(
                    str(_path),
                    name=_path.name,
                )

                _aliases = (
                    ["best", "latest"]
                    if _kind == "model"
                    else ["latest"]
                )

                _run.log_artifact(
                    _artifact,
                    aliases=_aliases,
                ).wait()


            # 11. Epoch loop
            for _epoch in range(
                _start_epoch,
                _config["epochs"] + 1,
            ):
                _model.train()

                if (
                    _epoch == _start_epoch
                    and _resume_indices is not None
                ):
                    _epoch_indices = _resume_indices
                    _start_step = _resume_step
                    _epoch_loss_sum = _resume_loss_sum
                    _epoch_updates = _resume_updates

                else:
                    _epoch_indices = torch.randperm(
                        len(_train),
                        generator=_rng,
                    )

                    _step_limit = _config["steps_per_epoch"]

                    if _step_limit is not None:
                        _epoch_indices = _epoch_indices[
                            :_step_limit * _effective_size
                        ]

                    _start_step = 0
                    _epoch_loss_sum = 0.0
                    _epoch_updates = 0

                _groups = _epoch_indices.split(
                    _effective_size
                )

                # 12. Optimizer steps
                for _step_index in range(
                    _start_step,
                    len(_groups),
                ):
                    _group_rows = _train[
                        _groups[_step_index]
                    ]

                    _group_size = len(_group_rows)

                    _optimizer.zero_grad(
                        set_to_none=True
                    )

                    # Full graph encoding
                    _graph_embeddings = _model.encode_all(
                        train_triples=_train,
                        visual_features=_visual,
                        edge_chunk_size=_config[
                            "edge_chunk_size"
                        ],
                        use_checkpoint=_config[
                            "use_gradient_checkpointing"
                        ],
                    )

                    _embedding_leaf = (
                        _graph_embeddings.detach()
                        .requires_grad_(True)
                    )

                    _group_loss = 0.0

                    # 13. Gradient accumulation
                    for _micro_rows in _group_rows.split(
                        _micro_size
                    ):
                        _actual_size = len(_micro_rows)

                        if _actual_size == 1:
                            _micro_rows = torch.cat(
                                [_micro_rows, _micro_rows],
                                dim=0,
                            )

                        _heads = _micro_rows[:, 0].to(
                            _device
                        )

                        _relations = _micro_rows[:, 1].to(
                            _device
                        )

                        _labels = torch.zeros(
                            (
                                len(_micro_rows),
                                _num_entities,
                            ),
                            dtype=torch.float32,
                            device=_device,
                        )

                        for _row, (_h, _r, _t) in enumerate(
                            _micro_rows.tolist()
                        ):
                            _labels[
                                _row,
                                _known_tails[(_h, _r)],
                            ] = 1.0

                        _epsilon = _config[
                            "label_smoothing"
                        ]

                        _labels = (
                            (1.0 - _epsilon) * _labels
                            + _epsilon / _num_entities
                        )

                        _logits = _model.score_all(
                            heads=_heads,
                            relations=_relations,
                            embeddings=_embedding_leaf,
                        )

                        _loss = (
                            F.binary_cross_entropy_with_logits(
                                _logits,
                                _labels,
                                reduction="mean",
                            )
                        )

                        _weighted_loss = (
                            _loss * (
                                _actual_size / _group_size
                            )
                        )

                        _weighted_loss.backward()

                        _group_loss += float(
                            _weighted_loss.detach().item()
                        )

                        del (
                            _heads,
                            _relations,
                            _labels,
                            _logits,
                            _loss,
                            _weighted_loss,
                        )

                    # 14. Full graph backward
                    if _embedding_leaf.grad is None:
                        raise RuntimeError(
                            "Missing accumulated gradient."
                        )

                    _graph_embeddings.backward(
                        _embedding_leaf.grad
                    )

                    _grad_norm = (
                        torch.nn.utils.clip_grad_norm_(
                            _model.parameters(),
                            _config["grad_clip"],
                        )
                    )

                    if not torch.isfinite(_grad_norm):
                        raise RuntimeError(
                            "Non-finite LAFA gradient."
                        )

                    _optimizer.step()

                    _global_step += 1
                    _epoch_updates += 1
                    _epoch_loss_sum += _group_loss

                    del (
                        _graph_embeddings,
                        _embedding_leaf,
                    )

                    # Local checkpoint per optimizer step
                    if (
                        _global_step
                        % _config.get(
                            "checkpoint_every_steps", 1
                        ) == 0
                    ):
                        _save_checkpoint(
                            epoch=_epoch,
                            next_step=_step_index + 1,
                            epoch_complete=False,
                            epoch_indices=_epoch_indices,
                            epoch_loss_sum=_epoch_loss_sum,
                            epoch_updates=_epoch_updates,
                        )

                _train_loss = (
                    _epoch_loss_sum
                    / max(_epoch_updates, 1)
                )

                # 15. Validation
                _validate = (
                    _epoch == 1
                    or _epoch % _config[
                        "validate_every"
                    ] == 0
                    or _epoch == _config["epochs"]
                )

                _metrics = None
                _is_best = False

                if _validate:
                    _model.eval()

                    with torch.inference_mode():
                        _validation_embeddings = (
                            _model.encode_all(
                                train_triples=_train,
                                visual_features=_visual,
                                edge_chunk_size=_config[
                                    "edge_chunk_size"
                                ],
                                use_checkpoint=False,
                            )
                        )

                        _metrics = rank_lafa_paper(
                            model=_model,
                            embeddings=_validation_embeddings,
                            triples=_valid_batch,
                            head_filter=_head_filter,
                            tail_filter=_tail_filter,
                            batch_size=_config[
                                "valid_batch_size"
                            ],
                            head_query_batch_size=_config[
                                "head_query_batch_size"
                            ],
                            tolerance=_config[
                                "tie_tolerance"
                            ],
                        )

                    del _validation_embeddings

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

                # 16. Record history
                _record = {
                    "epoch": _epoch,
                    "train_loss": _train_loss,
                    "global_step": _global_step,
                    "optimizer_updates": _epoch_updates,
                    "train_presentations": len(
                        _epoch_indices
                    ),
                }

                if _metrics is not None:
                    _record.update({
                        "filtered_mrr": (
                            _metrics["both"]["mrr"]
                        ),
                        "head_mrr": (
                            _metrics["head"]["mrr"]
                        ),
                        "tail_mrr": (
                            _metrics["tail"]["mrr"]
                        ),
                        "hits10": (
                            _metrics["both"]["hits10"]
                        ),
                    })

                _history.append(_record)

                _finished = (
                    _epoch >= _config["epochs"]
                    or _stale >= _config["patience"]
                )

                # 17. Save final epoch state
                _save_checkpoint(
                    epoch=_epoch,
                    next_step=len(_groups),
                    epoch_complete=True,
                    epoch_indices=None,
                    epoch_loss_sum=_epoch_loss_sum,
                    epoch_updates=_epoch_updates,
                    is_best=_is_best,
                    finished=_finished,
                )

                # 18. Upload resume checkpoint every epoch
                _sync_checkpoint(
                    _last_path,
                    "resume",
                )

                # Upload best checkpoint when improved
                if _is_best:
                    _sync_checkpoint(
                        _best_path,
                        "model",
                    )

                # 19. W&B logs
                _log = {
                    "train/loss": _train_loss,
                    "train/global_step": _global_step,
                    "train/optimizer_updates": _epoch_updates,
                    "train/presentations": len(
                        _epoch_indices
                    ),
                    "best/filtered_mrr": _best_mrr,
                }

                if _metrics is not None:
                    for _direction, _values in _metrics.items():
                        for _metric, _value in _values.items():
                            _log[
                                f"valid/{_direction}/{_metric}"
                            ] = _value

                _run.log(_log, step=_epoch)


                _resume_indices = None
                _resume_step = 0

                if _finished:
                    break

            # Ensure best checkpoint is available remotely
            if _best_path.exists():
                _run.summary["best_epoch"] = _best_epoch
                _run.summary["best_mrr"] = _best_mrr

        if _device.type == "cuda":
            torch.cuda.empty_cache()

        return {
            "history": _history,
            "best_mrr": _best_mrr,
            "best_epoch": _best_epoch,
            "global_step": _global_step,
            "checkpoint": str(_best_path),
            "training_version": training_version,
            "finished": _finished,
        }

    return (train_lafa,)


@app.cell
def _(graph_data, mo):
    train_dataset_selector = mo.ui.dropdown(
        options=list(graph_data.keys()),
        value="FB15k-237-IMG",
        label="Dataset",
    )

    train_lafa_button = mo.ui.run_button(
        label="Train LAFA"
    )

    mo.vstack([
        train_dataset_selector,
        train_lafa_button,
    ])
    return train_dataset_selector, train_lafa_button


@app.cell
def _(mo, train_dataset_selector, train_lafa, train_lafa_button):
    mo.stop(
        not train_lafa_button.value,
        mo.md("Click **Train LAFA** to start."),
    )

    lafa_training_result = train_lafa(
        train_dataset_selector.value
    )




    return


@app.cell
def _(mo, torch, train_dataset_selector, training_root):
    _history_path = (
        training_root
        / train_dataset_selector.value
        / "last.pt"
    )

    if _history_path.exists():
        _state = torch.load(
            _history_path,
            map_location="cpu",
            weights_only=True,
        )

        _history = _state["history"]

        _rows = [
            "| Epoch | Train Loss | Sampled Val MRR | Hits@10 |",
            "|---:|---:|---:|---:|",
        ]

        for _record in _history:
            _rows.append(
                f"| {_record['epoch']} "
                f"| {_record['train_loss']:.4f} "
                f"| {_record.get('sampled_val_mrr', float('nan')):.4f} "
                f"| {_record.get('sampled_val_hits10', float('nan')):.4f} |"
            )

        mo.output.replace(
            mo.md(
                f"### LAFA Training — {train_dataset_selector.value}\n\n"
                + "\n".join(_rows)
            )
        )
    else:
        mo.output.replace(
            mo.md("No training checkpoint found.")
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. LAFA Full Filtered Evaluation
    """)
    return


@app.cell
def _(CHECKPOINT_DIR, CONFIG, DEVICE, json, training_version):
    evaluation_version = f"{training_version}-fulltest-v1"

    evaluation_config = {
        "max_test_triples": None,
        "entity_batch_size": 64 if DEVICE == "cuda" else 16,
        "query_batch_size": 32 if DEVICE == "cuda" else 8,
        "head_query_batch_size": 256 if DEVICE == "cuda" else 32,
        "tie_tolerance": 1e-5,
        "seed": CONFIG["seed"],
        "log_wandb": True,
    }

    evaluation_root = CHECKPOINT_DIR / evaluation_version
    evaluation_root.mkdir(parents=True, exist_ok=True)

    return evaluation_config, evaluation_root, evaluation_version


@app.cell
def _(graph_data):
    def build_lafa_filters(dataset_name):
        _data = graph_data[dataset_name]

        _head_filter = {}
        _tail_filter = {}

        for _split in ("train", "valid", "test"):
            for _h, _r, _t in _data[_split].tolist():
                _head_filter.setdefault(
                    (_r, _t), set()
                ).add(_h)

                _tail_filter.setdefault(
                    (_h, _r), set()
                ).add(_t)

        return _head_filter, _tail_filter

    return (build_lafa_filters,)


@app.cell
def _(graph_data, sample_neighbors, torch, training_config):
    @torch.inference_mode()
    def encode_all_lafa_entities(
        model,
        dataset_name,
        visual_features,
        batch_size,
    ):
        _num_entities = graph_data[
            dataset_name
        ]["num_entities"]

        _parts = []

        model.eval()

        for _start in range(
            0, _num_entities, batch_size
        ):
            _end = min(
                _start + batch_size,
                _num_entities,
            )

            _ids = torch.arange(
                _start,
                _end,
                dtype=torch.long,
            )

            _neighbors = sample_neighbors(
                dataset_name=dataset_name,
                entity_ids=_ids,
                num_neighbors=training_config["num_neighbors"],
                generator=None,
            )

            _encoded = model.encode_entities(
                entity_ids=_ids,
                neighbor_ids=_neighbors["neighbor_ids"],
                neighbor_relations=_neighbors["neighbor_relations"],
                neighbor_mask=_neighbors["neighbor_mask"],
                image_features=visual_features["images"],
                image_mask=visual_features["mask"],
            )

            _parts.append(
                _encoded.float().cpu()
            )

        return torch.cat(
            _parts,
            dim=0,
        )

    return


@app.cell
def _(rank_lafa_paper, torch):
    @torch.inference_mode()
    def evaluate_lafa_filtered(
        model,
        embeddings,
        triples,
        head_filter,
        tail_filter,
        config,
    ):
        return rank_lafa_paper(
            model=model,
            embeddings=embeddings,
            triples=triples,
            head_filter=head_filter,
            tail_filter=tail_filter,
            batch_size=config["query_batch_size"],
            head_query_batch_size=config[
                "head_query_batch_size"
            ],
            tolerance=config["tie_tolerance"],
        )

    return (evaluate_lafa_filtered,)


@app.cell
def _(
    CONFIG,
    DEVICE,
    LAFA,
    WANDB_ENTITY,
    WANDB_PROJECT,
    build_lafa_filters,
    evaluate_lafa_filtered,
    evaluation_config,
    evaluation_root,
    evaluation_version,
    graph_data,
    json,
    torch,
    training_root,
    training_version,
    vision_model_id,
    visual_data,
    wandb,
):
    def run_lafa_evaluation(dataset_name):
        _data = graph_data[dataset_name]
        _visual = visual_data[dataset_name]
        _config = evaluation_config
        _device = torch.device(DEVICE)

        _output = evaluation_root / dataset_name
        _output.mkdir(parents=True, exist_ok=True)

        _checkpoint_path = (
            training_root / dataset_name / "best.pt"
        )

        # 1. Automatically restore best.pt from W&B
        if not _checkpoint_path.exists():
            _api = wandb.Api()
            _entity = WANDB_ENTITY or _api.default_entity

            _artifact_name = (
                f"lafa-{dataset_name.lower()}-"
                f"{training_version}-model"
            )

            _artifact_ref = (
                f"{_entity}/{WANDB_PROJECT}/"
                f"{_artifact_name}:best"
            )


            _checkpoint_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            _remote = _api.artifact(_artifact_ref)

            _remote.download(
                root=str(_checkpoint_path.parent)
            )

        if not _checkpoint_path.exists():
            raise FileNotFoundError(
                f"Best checkpoint unavailable: "
                f"{_checkpoint_path}"
            )

        # 2. Load checkpoint
        _checkpoint = torch.load(
            _checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )

        if (
            _checkpoint.get("training_version")
            != training_version
        ):
            raise RuntimeError(
                "Checkpoint training version mismatch."
            )

        if _checkpoint.get("training_mode") not in (
            "paper_aligned",
            "paper_aligned_accumulation",
        ):
            raise RuntimeError(
                "Unsupported LAFA checkpoint mode: "
                f"{_checkpoint.get('training_mode')}"
            )

        _model_config = _checkpoint["config"]

        # 3. Initialize model
        _model = LAFA(
            num_entities=_data["num_entities"],
            num_relations=_data["num_relations"],
            dim=CONFIG["dim"],
            image_dim=CONFIG["image_dim"],
            fusion_heads=CONFIG["fusion_heads"],
            aggregation_heads=CONFIG["aggregation_heads"],
            decoder_h=CONFIG["decoder_h"],
            decoder_w=CONFIG["decoder_w"],
            conv_channels=CONFIG["conv_channels"],
            dropout=CONFIG["dropout"],
        ).to(_device)

        _model.load_state_dict(
            _checkpoint["model_state_dict"]
        )

        _model.eval()


        # 4. Full graph encoding
        _train = _data["train"].long().cpu()


        with torch.inference_mode():
            _embeddings = _model.encode_all(
                train_triples=_train,
                visual_features=_visual,
                edge_chunk_size=_model_config[
                    "edge_chunk_size"
                ],
                use_checkpoint=False,
            )


        # 5. Evaluation triples
        _test_all = _data["test"].long().cpu()
        _limit = _config["max_test_triples"]

        if _limit is None:
            _test = _test_all

        else:
            _limit = int(_limit)

            if _limit <= 0:
                raise ValueError(
                    "max_test_triples must be positive or None."
                )

            if _limit >= len(_test_all):
                _test = _test_all
            else:
                _rng = torch.Generator().manual_seed(
                    _config["seed"]
                )

                _indices = torch.randperm(
                    len(_test_all),
                    generator=_rng,
                )[:_limit]

                _test = _test_all[_indices]

        if len(_test) == 0:
            raise RuntimeError("Empty test set.")


        # 6. Filtered evaluation
        _head_filter, _tail_filter = (
            build_lafa_filters(dataset_name)
        )

        with torch.inference_mode():
            _metrics = evaluate_lafa_filtered(
                model=_model,
                embeddings=_embeddings,
                triples=_test,
                head_filter=_head_filter,
                tail_filter=_tail_filter,
                config=_config,
            )

        # 7. Result
        _result = {
            "dataset": dataset_name,
            "training_version": training_version,
            "evaluation_version": evaluation_version,
            "training_mode": _checkpoint["training_mode"],
            "architecture": "LAFA-MIA-LAF-LAA-ConvE",
            "decoder": "ConvE",
            "objective": "one_vs_all_bce",
            "reciprocal": False,

            "checkpoint_epoch": _checkpoint["epoch"],
            "best_epoch": _checkpoint["best_epoch"],
            "best_validation_mrr": _checkpoint["best_mrr"],
            "global_step": _checkpoint.get("global_step"),

            "num_entities": _data["num_entities"],
            "num_relations": _data["num_relations"],
            "num_train_edges": len(_train),

            "test_triples_total": len(_test_all),
            "test_triples_evaluated": len(_test),
            "test_subset": len(_test) < len(_test_all),

            "candidate_policy": "all_entities",
            "filter_policy": "train_valid_test",
            "neighbor_policy": "all_train_outgoing",
            "head_prediction": "original_relation_conve",
            "tail_prediction": "original_relation_conve",

            "visual_encoder": vision_model_id,
            "visual_coverage": float(
                _visual["mask"]
                .any(dim=1)
                .float()
                .mean()
                .item()
            ),

            "metrics": _metrics,
        }

        # 8. Save report
        _report_path = (
            _output / "filtered_evaluation.json"
        )

        with open(
            _report_path,
            "w",
            encoding="utf-8",
        ) as _file:
            json.dump(
                _result,
                _file,
                ensure_ascii=False,
                indent=2,
            )

        # 9. W&B evaluation artifact
        if _config["log_wandb"]:
            with wandb.init(
                project=WANDB_PROJECT,
                entity=WANDB_ENTITY,
                job_type="evaluation",
                name=(
                    f"lafa-{dataset_name.lower()}-"
                    f"{evaluation_version}"
                ),
                group=evaluation_version,
                tags=[
                    "LAFA",
                    "paper-reproduction",
                    "full-filtered",
                    "wandb-checkpoint-restore",
                    dataset_name,
                ],
                config={
                    **_config,
                    "dataset": dataset_name,
                    "training_version": training_version,
                    "evaluation_version": evaluation_version,
                    "training_mode": (
                        _checkpoint["training_mode"]
                    ),
                    "best_epoch": _checkpoint["best_epoch"],
                    "global_step": (
                        _checkpoint.get("global_step")
                    ),
                    "test_triples_evaluated": len(_test),
                },
            ) as _run:

                _run.log({
                    f"filtered/{_direction}/{_metric}": _value
                    for _direction, _values in _metrics.items()
                    for _metric, _value in _values.items()
                })

                _run.summary["best_epoch"] = (
                    _checkpoint["best_epoch"]
                )

                _run.summary["best_validation_mrr"] = (
                    _checkpoint["best_mrr"]
                )

                _run.summary["visual_coverage"] = (
                    _result["visual_coverage"]
                )

                _artifact = wandb.Artifact(
                    name=(
                        f"lafa-{dataset_name.lower()}-"
                        f"{evaluation_version}-report"
                    ),
                    type="evaluation",
                    metadata={
                        "dataset": dataset_name,
                        "training_version": training_version,
                        "evaluation_version": evaluation_version,
                        "best_epoch": (
                            _checkpoint["best_epoch"]
                        ),
                        "test_triples": len(_test),
                    },
                )

                _artifact.add_file(str(_report_path))

                _run.log_artifact(
                    _artifact,
                    aliases=["latest"],
                ).wait()

        # 10. Print results

        for _direction in ("head", "tail", "both"):
            _scores = _metrics[_direction]



        del _embeddings
        del _model

        if _device.type == "cuda":
            torch.cuda.empty_cache()

        return _result

    return (run_lafa_evaluation,)


@app.cell
def _(graph_data, mo):
    evaluate_dataset_selector = mo.ui.dropdown(
        options=list(graph_data.keys()),
        value="FB15k-237-IMG",
        label="Dataset",
    )

    evaluate_lafa_button = mo.ui.run_button(
        label="Evaluate LAFA"
    )

    mo.vstack([
        evaluate_dataset_selector,
        evaluate_lafa_button,
    ])
    return evaluate_dataset_selector, evaluate_lafa_button


@app.cell
def _(
    evaluate_dataset_selector,
    evaluate_lafa_button,
    mo,
    run_lafa_evaluation,
):
    mo.stop(
        not evaluate_lafa_button.value,
        mo.md("Click **Evaluate LAFA** to start."),
    )

    lafa_evaluation_result = run_lafa_evaluation(
        evaluate_dataset_selector.value
    )



    return


@app.cell
def _(evaluation_root, evaluation_version, graph_data, json, mo):
    _report_rows = [
        "| Dataset | Triples | MRR | MR | Hits@1 | Hits@3 | Hits@10 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    _found = 0

    for _dataset in graph_data:
        _path = (
            evaluation_root
            / _dataset
            / "filtered_evaluation.json"
        )

        if not _path.exists():
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

    if _found:
        _report_text = "\n".join(_report_rows)
    else:
        _report_text = "No evaluation reports found."

    mo.output.replace(
        mo.md(
            f"### LAFA — {evaluation_version}\n\n"
            + _report_text
        )
    )
    return



if __name__ == "__main__":
    app.run()
