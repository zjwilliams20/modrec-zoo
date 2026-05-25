#!/usr/bin/env python
import argparse
import json
import re
import shutil
import struct
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import math

import numpy as np
import polars as pl
from tqdm import tqdm

from modreczoo.data import METADATA_FILE, SIGNALS_FILE, save_dataset


LABEL_MAP = {
    "bpsk": "2PSK",
    "qpsk": "4PSK",
    "8psk": "8PSK",
    "dqpsk": "pi/4-DQPSK",
    "msk": "MSK",
    "16qam": "16QAM",
    "64qam": "64QAM",
    "256qam": "256QAM",
}

SIGNAL_RE = re.compile(r"signal_(\d+)\.tim$")
CSPB_2018R2_POST_URL = "https://cyclostationary.blog/2023/09/25/cspb-ml-2018r2-correcting-an-rng-flaw-in-cspb-ml-2018/"
CSPB_METADATA_URL = "https://cyclostationary.blog/wp-content/uploads/2023/09/signal_record_C_2023.txt"
CSPB_BATCH_URL_TEMPLATE = "https://cyclostationary.blog/wp-content/uploads/2023/09/CSPB.ML_.2018R2_{batch}.zip"
CSPB_MISSING_SIGNAL_URL = "https://cyclostationary.blog/wp-content/uploads/2023/10/signal_31986.tim_.zip"


@dataclass(frozen=True)
class TimSource:
    cspb_signal_index: int
    path: str | None = None
    zip_path: str | None = None
    zip_member: str | None = None


def read_tim_bytes(raw: bytes, source_name: str = "<bytes>") -> np.ndarray:
    if len(raw) < 8:
        raise ValueError(f"{source_name}: expected at least 8 bytes.")

    flag, n_samples = struct.unpack("<ii", raw[:8])
    payload = raw[8:]
    if flag == 1:
        expected = n_samples * 4
        if len(payload) != expected:
            raise ValueError(f"{source_name}: expected {expected} real payload bytes, got {len(payload)}.")
        return np.frombuffer(payload, dtype="<f4").astype(np.complex64)
    if flag == 2:
        expected = 2 * n_samples * 4
        if len(payload) != expected:
            raise ValueError(f"{source_name}: expected {expected} complex payload bytes, got {len(payload)}.")
        interleaved = np.frombuffer(payload, dtype="<f4")
        return (interleaved[0::2] + 1j * interleaved[1::2]).astype(np.complex64)
    raise ValueError(f"{source_name}: unsupported CMS real/complex flag {flag}.")


def read_tim_source(source: TimSource) -> tuple[int, np.ndarray]:
    if source.path is not None:
        path = Path(source.path)
        return source.cspb_signal_index, read_tim_bytes(path.read_bytes(), str(path))

    if source.zip_path is None or source.zip_member is None:
        raise ValueError(f"Signal {source.cspb_signal_index} has no file or zip source.")
    with zipfile.ZipFile(source.zip_path) as zf:
        return source.cspb_signal_index, read_tim_bytes(zf.read(source.zip_member), source.zip_member)


def download_file(url: str, path: str | Path, force: bool = False) -> bool:
    path = Path(path)
    if path.exists() and not force:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    print(f"Downloading {url} -> {path}")
    with urllib.request.urlopen(url) as response, tmp_path.open("wb") as out:
        shutil.copyfileobj(response, out)
    tmp_path.replace(path)
    return True


def _zip_has_extracted_members(input_dir: Path, zip_path: Path) -> bool:
    with zipfile.ZipFile(zip_path) as zf:
        tim_members = [name for name in zf.namelist() if name.endswith(".tim")]
        if not tim_members:
            return True
        return all((input_dir / member).exists() for member in tim_members)


def extract_zip_if_needed(input_dir: str | Path, zip_path: str | Path, force: bool = False) -> bool:
    input_dir = Path(input_dir)
    zip_path = Path(zip_path)
    if not force and _zip_has_extracted_members(input_dir, zip_path):
        return False
    print(f"Extracting {zip_path} -> {input_dir}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(input_dir)
    return True


def parse_batches(value: str) -> list[int]:
    batches: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            batches.update(range(int(lo), int(hi) + 1))
        else:
            batches.add(int(part))
    result = sorted(batches)
    invalid = [batch for batch in result if batch < 1 or batch > 28]
    if invalid:
        raise ValueError(f"CSPB 2018R2 batch numbers must be in 1..28, got {invalid}.")
    return result


def prepare_cspb_inputs(
    input_dir: str | Path,
    metadata_path: str | Path,
    download_if_missing: bool = False,
    batches: Iterable[int] = range(1, 29),
    force_extract: bool = False,
) -> dict:
    input_dir = Path(input_dir)
    metadata_path = Path(metadata_path)
    input_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    extracted = 0
    if download_if_missing:
        downloaded += int(download_file(CSPB_METADATA_URL, metadata_path))
        for batch in batches:
            zip_path = input_dir / f"CSPB.ML_.2018R2_{batch}.zip"
            batch_dir = input_dir / f"Batch_Dir_{batch}"
            if not zip_path.exists() and not batch_dir.exists():
                downloaded += int(download_file(CSPB_BATCH_URL_TEMPLATE.format(batch=batch), zip_path))
        downloaded += int(download_file(CSPB_MISSING_SIGNAL_URL, input_dir / "signal_31986.tim_.zip"))

    for zip_path in sorted(input_dir.glob("*.zip")):
        extracted += int(extract_zip_if_needed(input_dir, zip_path, force=force_extract))

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing metadata file {metadata_path}. Re-run with --download-if-missing or provide --metadata."
        )
    if not discover_tim_sources(input_dir, batches=batches):
        raise FileNotFoundError(
            f"No .tim sources found in {input_dir}. Re-run with --download-if-missing or provide extracted/zip batches."
        )
    return {"downloaded": downloaded, "extracted": extracted}


def _batch_from_source(source: TimSource) -> int | None:
    if source.path:
        for part in Path(source.path).parts:
            if part.startswith("Batch_Dir_"):
                try:
                    return int(part.split("_")[-1])
                except ValueError:
                    pass
    if source.zip_path:
        stem = Path(source.zip_path).stem  # e.g. CSPB.ML_.2018R2_3
        try:
            return int(stem.split("_")[-1])
        except ValueError:
            pass
    return None


def discover_tim_sources(
    input_dir: str | Path,
    batches: Iterable[int] | None = None,
) -> dict[int, TimSource]:
    input_dir = Path(input_dir)
    batch_set = set(batches) if batches is not None else None
    sources: dict[int, TimSource] = {}
    for path in sorted([*input_dir.glob("Batch_Dir_*/*.tim"), *input_dir.glob("*.tim")]):
        match = SIGNAL_RE.match(path.name)
        if match:
            idx = int(match.group(1))
            source = TimSource(cspb_signal_index=idx, path=str(path))
            if batch_set is None or _batch_from_source(source) in batch_set:
                sources[idx] = source

    for zip_path in sorted(input_dir.glob("*.zip")):
        with zipfile.ZipFile(zip_path) as zf:
            for member in zf.namelist():
                match = SIGNAL_RE.search(Path(member).name)
                if match:
                    idx = int(match.group(1))
                    source = TimSource(cspb_signal_index=idx, zip_path=str(zip_path), zip_member=member)
                    if batch_set is None or _batch_from_source(source) in (batch_set | {None}):
                        sources.setdefault(idx, source)
    return sources


def parse_metadata_file(path: str | Path) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) != 9:
                raise ValueError(f"{path}:{line_no}: expected 9 whitespace-separated fields, got {len(parts)}.")

            cspb_signal_index = int(parts[0])
            cspb_modulation = parts[1].lower()
            if cspb_modulation not in LABEL_MAP:
                raise ValueError(f"{path}:{line_no}: unsupported CSPB modulation {cspb_modulation!r}.")

            base_symbol_period = float(parts[2])
            upsample_factor = float(parts[5])
            downsample_factor = float(parts[6])
            noise_spectral_density_db = float(parts[8])
            # downsample_factor == 0 means no resampling; the signal is stored at
            # base_symbol_period * upsample_factor samples/symbol. Standard
            # metadata stores the effective downsample factor so osr remains
            # upsample_factor / downsample_factor.
            effective_downsample = 1.0 if downsample_factor == 0 else downsample_factor
            osr = upsample_factor / effective_downsample

            rows[cspb_signal_index] = {
                "cspb_signal_index": cspb_signal_index,
                "cspb_modulation": cspb_modulation,
                "modulation": LABEL_MAP[cspb_modulation],
                "snr_db": float(parts[7]),
                "cfo": float(parts[3]),
                "ebw": float(parts[4]),
                "symbol_period": int(base_symbol_period),
                "upsample_factor": int(upsample_factor),
                "downsample_factor": int(effective_downsample),
                "noise_spectral_density_db": noise_spectral_density_db,
                "osr": float(osr),
                "symbol_rate": float(1.0 / (base_symbol_period * osr)),
            }
    return rows


def _source_metadata(source: TimSource) -> dict:
    return {
        "source_path": source.path or "",
        "source_zip": source.zip_path or "",
        "source_member": source.zip_member or "",
    }


def read_tim_shape(source: TimSource) -> int:
    signal_index, signal = read_tim_source(source)
    if signal_index != source.cspb_signal_index:
        raise ValueError(f"Read signal {signal_index}, expected {source.cspb_signal_index}.")
    return len(signal)


def load_tim_sources(
    sources: list[TimSource],
    output_dir: Path,
    num_workers: int,
) -> np.ndarray:
    n_samples = read_tim_shape(sources[0])
    tmp_path = output_dir / ".signals.tmp.npy"
    signals = np.lib.format.open_memmap(
        tmp_path,
        mode="w+",
        dtype=np.complex64,
        shape=(len(sources), n_samples),
    )

    def load_one(item: tuple[int, TimSource]) -> int:
        row, source = item
        _, signal = read_tim_source(source)
        if len(signal) != n_samples:
            raise ValueError(
                f"Signal {source.cspb_signal_index}: expected {n_samples} samples, got {len(signal)}."
            )
        signals[row] = signal
        return row

    items = list(enumerate(sources))
    if num_workers > 1:
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            for _ in tqdm(
                ex.map(load_one, items),
                total=len(items),
                desc="Reading CSPB .tim",
                unit="sig",
            ):
                pass
    else:
        for item in tqdm(items, desc="Reading CSPB .tim", unit="sig"):
            load_one(item)

    signals.flush()
    return signals


def write_manifest(output_dir: Path, metadata: pl.DataFrame, result: dict) -> None:
    manifest = {
        "signals": SIGNALS_FILE,
        "metadata": METADATA_FILE,
        "source": "CSPB.ML.2018R2",
        "source_post_url": CSPB_2018R2_POST_URL,
        "metadata_url": CSPB_METADATA_URL,
        "batch_url_template": CSPB_BATCH_URL_TEMPLATE,
        "missing_signal_url": CSPB_MISSING_SIGNAL_URL,
        "modulations": sorted(metadata["modulation"].unique().to_list()),
        "n_signals": result["n_signals"],
        "n_samples": result["n_samples"],
        "counters": {
            key: result[key]
            for key in (
                "batch_fraction",
                "seed",
                "metadata_rows",
                "tim_sources",
                "skipped_missing_metadata",
                "skipped_missing_tim",
                "downloaded",
                "extracted",
            )
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _select_per_batch(
    ids: list[int],
    sources: dict[int, TimSource],
    batch_fraction: float,
    seed: int | None = None,
) -> list[int]:
    """Return a sorted subset of ids taking batch_fraction from each batch."""
    if batch_fraction >= 1.0:
        return ids
    rng = np.random.default_rng(seed)
    by_batch: dict[int | None, list[int]] = {}
    for idx in ids:
        batch = _batch_from_source(sources[idx])
        by_batch.setdefault(batch, []).append(idx)
    selected: list[int] = []
    for batch_ids in by_batch.values():
        n = max(1, math.ceil(len(batch_ids) * batch_fraction))
        chosen = rng.choice(batch_ids, size=n, replace=False)
        selected.extend(chosen.tolist())
    return sorted(selected)


def convert_cspb_dataset(
    input_dir: str | Path,
    metadata_path: str | Path,
    output_dir: str | Path,
    num_workers: int = 1,
    max_signals: int | None = None,
    batch_fraction: float = 1.0,
    seed: int | None = None,
    force: bool = False,
    download_if_missing: bool = False,
    batches: Iterable[int] = range(1, 29),
    force_extract: bool = False,
) -> dict:
    num_workers = max(1, int(num_workers))
    if max_signals is not None and max_signals <= 0:
        raise ValueError("--max-signals must be positive when provided.")
    if not (0.0 < batch_fraction <= 1.0):
        raise ValueError("--batch-fraction must be in (0, 1].")
    output_dir = Path(output_dir)
    if output_dir.exists() and any((output_dir / name).exists() for name in (SIGNALS_FILE, METADATA_FILE)):
        if not force:
            raise FileExistsError(f"{output_dir} already contains a dataset; use --force to overwrite it.")
        for name in (SIGNALS_FILE, METADATA_FILE, "manifest.json"):
            p = output_dir / name
            if p.exists():
                p.unlink()

    prepared = prepare_cspb_inputs(
        input_dir,
        metadata_path,
        download_if_missing=download_if_missing,
        batches=batches,
        force_extract=force_extract,
    )
    sources = discover_tim_sources(input_dir, batches=batches)
    metadata_by_id = parse_metadata_file(metadata_path)
    selected_ids = sorted(set(sources) & set(metadata_by_id))
    selected_ids = _select_per_batch(selected_ids, sources, batch_fraction, seed=seed)
    if max_signals is not None:
        selected_ids = selected_ids[:max_signals]
    if not selected_ids:
        raise ValueError("No CSPB signals had both metadata and .tim data.")

    selected_sources = [sources[idx] for idx in selected_ids]
    output_dir.mkdir(parents=True, exist_ok=True)
    signals_tmp_path = output_dir / ".signals.tmp.npy"
    if signals_tmp_path.exists():
        signals_tmp_path.unlink()
    signals = load_tim_sources(selected_sources, output_dir, num_workers)
    rows = []
    for signal_id, cspb_idx in enumerate(selected_ids):
        source = sources[cspb_idx]
        row = {
            "signal_id": signal_id,
            **metadata_by_id[cspb_idx],
            "n_samples": int(signals.shape[1]),
            "channel": "unknown",
            "cpo": 0.0,
            "sto": 0.0,
            "channel_n_taps": 0,
            "channel_max_delay_samples": 0,
            "channel_rms_delay_samples": 0.0,
            "channel_tap_delays": json.dumps([]),
            "channel_tap_real": json.dumps([]),
            "channel_tap_imag": json.dumps([]),
            **_source_metadata(source),
        }
        rows.append(row)

    metadata = pl.DataFrame(rows)
    result = {
        "output_dir": str(output_dir),
        "n_signals": int(signals.shape[0]),
        "n_samples": int(signals.shape[1]),
        "batch_fraction": batch_fraction,
        "seed": seed,
        "metadata_rows": len(metadata_by_id),
        "tim_sources": len(sources),
        "skipped_missing_metadata": len(set(sources) - set(metadata_by_id)),
        "skipped_missing_tim": len(set(metadata_by_id) - set(sources)),
        **prepared,
    }
    try:
        save_dataset(str(output_dir), signals, metadata)
        write_manifest(output_dir, metadata, result)
    finally:
        if signals_tmp_path.exists():
            signals_tmp_path.unlink()
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert CSPB.ML.2018R2 .tim batches to ModRec dataset format.")
    parser.add_argument("--input-dir", default="data/cspb_2018r2/delivered")
    parser.add_argument("--metadata", default="data/cspb_2018r2/signal_record_C_2023.txt")
    parser.add_argument("--output-dir", default="data/cspb_2018r2/converted")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-signals", type=int, default=None)
    parser.add_argument(
        "--batch-fraction",
        type=float,
        default=1.0,
        help="Fraction of each batch to include (0, 1]. Default 1.0 (all). "
             "Use e.g. 0.1 to take 10%% of every batch for broad coverage.",
    )
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for per-batch random selection.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--download-if-missing", action="store_true")
    parser.add_argument("--batches", default="1-28", help="Batch numbers to download, such as '1-8' or '1,3,8'.")
    parser.add_argument("--force-extract", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = convert_cspb_dataset(
        input_dir=args.input_dir,
        metadata_path=args.metadata,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        max_signals=args.max_signals,
        batch_fraction=args.batch_fraction,
        seed=args.seed,
        force=args.force,
        download_if_missing=args.download_if_missing,
        batches=parse_batches(args.batches),
        force_extract=args.force_extract,
    )
    print(
        f"Wrote {result['n_signals']} signals x {result['n_samples']} samples to {result['output_dir']} "
        f"({result['skipped_missing_tim']} metadata rows missing .tim files)."
    )


if __name__ == "__main__":
    main()
