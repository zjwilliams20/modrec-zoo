import importlib.util
import json
import struct
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import polars as pl

from modreczoo.data import load_dataset


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "convert_cspb_2018r2.py"
SPEC = importlib.util.spec_from_file_location("convert_cspb_2018r2", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
converter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(converter)


def _tim_bytes(x: np.ndarray) -> bytes:
    interleaved = np.empty(2 * len(x), dtype="<f4")
    interleaved[0::2] = np.real(x).astype("<f4")
    interleaved[1::2] = np.imag(x).astype("<f4")
    return struct.pack("<ii", 2, len(x)) + interleaved.tobytes()


def test_read_tim_bytes_complex() -> None:
    x = np.asarray([1 + 2j, -3 + 4j], dtype=np.complex64)
    parsed = converter.read_tim_bytes(_tim_bytes(x), "test.tim")
    assert parsed.dtype == np.complex64
    assert np.allclose(parsed, x)


def test_parse_metadata_maps_labels_and_effective_osr(tmp_path: Path) -> None:
    path = tmp_path / "signal_record.txt"
    # row 1: downsample_factor=0 (no resampling) → osr = base_symbol_period * upsample_factor = 10*5 = 50
    # row 2: downsample_factor=2               → osr = base_symbol_period * upsample_factor / downsample_factor = 8*4/2 = 16
    path.write_text("1 dqpsk 10 -1e-4 0.35 5 0 7.0 0.0\n2 64qam 8 2e-4 0.5 4 2 3.0 0.0\n")

    rows = converter.parse_metadata_file(path)

    assert rows[1]["modulation"] == "pi/4-DQPSK"
    assert rows[1]["snr_db"] == 7.0
    assert rows[1]["base_symbol_period"] == 10.0
    assert rows[1]["symbol_period"] == 10
    assert rows[1]["upsample_factor"] == 5
    assert rows[1]["downsample_factor"] == 0
    assert rows[1]["osr"] == 50.0
    assert rows[2]["modulation"] == "64QAM"
    assert rows[2]["base_symbol_period"] == 8.0
    assert rows[2]["symbol_period"] == 8
    assert rows[2]["upsample_factor"] == 4
    assert rows[2]["downsample_factor"] == 2
    assert rows[2]["osr"] == 16.0


def test_discover_tim_sources_prefers_extracted_over_zip(tmp_path: Path) -> None:
    extracted = tmp_path / "Batch_Dir_1"
    extracted.mkdir()
    (extracted / "signal_1.tim").write_bytes(_tim_bytes(np.asarray([1 + 0j], dtype=np.complex64)))
    (tmp_path / "signal_31986.tim").write_bytes(_tim_bytes(np.asarray([4 + 0j], dtype=np.complex64)))
    with zipfile.ZipFile(tmp_path / "batch.zip", "w") as zf:
        zf.writestr("Batch_Dir_1/signal_1.tim", _tim_bytes(np.asarray([2 + 0j], dtype=np.complex64)))
        zf.writestr("Batch_Dir_1/signal_2.tim", _tim_bytes(np.asarray([3 + 0j], dtype=np.complex64)))

    sources = converter.discover_tim_sources(tmp_path)

    assert sources[1].path is not None
    assert sources[1].zip_path is None
    assert sources[2].zip_path is not None
    assert sources[31986].path is not None


def test_convert_cspb_dataset_end_to_end_with_zip_and_missing_rows(tmp_path: Path) -> None:
    metadata = tmp_path / "signal_record.txt"
    metadata.write_text(
        "\n".join(
            [
                "1 bpsk 2 -1e-4 0.35 7 6 1.0 0.0",
                "2 qpsk 3 -2e-4 0.45 8 6 2.0 0.0",
                "3 8psk 4 -3e-4 0.55 9 6 3.0 0.0",
            ]
        )
        + "\n"
    )
    delivered = tmp_path / "delivered"
    delivered.mkdir()
    batch = delivered / "Batch_Dir_1"
    batch.mkdir()
    x1 = np.asarray([1 + 2j, 3 + 4j], dtype=np.complex64)
    x2 = np.asarray([5 + 6j, 7 + 8j], dtype=np.complex64)
    (batch / "signal_1.tim").write_bytes(_tim_bytes(x1))
    with zipfile.ZipFile(delivered / "batch.zip", "w") as zf:
        zf.writestr("Batch_Dir_1/signal_2.tim", _tim_bytes(x2))
        zf.writestr("Batch_Dir_1/signal_4.tim", _tim_bytes(np.asarray([9 + 0j, 10 + 0j], dtype=np.complex64)))

    out = tmp_path / "converted"
    result = converter.convert_cspb_dataset(delivered, metadata, out, num_workers=1)
    signals, converted_metadata = load_dataset(out)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))

    assert result["n_signals"] == 2
    assert result["skipped_missing_tim"] == 1
    assert result["skipped_missing_metadata"] == 1
    assert signals.shape == (2, 2)
    assert np.allclose(signals[0], x1)
    assert np.allclose(signals[1], x2)
    assert converted_metadata["signal_id"].to_list() == [0, 1]
    assert converted_metadata["cspb_signal_index"].to_list() == [1, 2]
    assert converted_metadata["modulation"].to_list() == ["2PSK", "4PSK"]
    assert manifest["source"] == "CSPB.ML.2018R2"
    assert manifest["signals"] == converter.SIGNALS_FILE
    assert manifest["metadata"] == "metadata.parquet"
    assert manifest["n_signals"] == 2
    assert manifest["modulations"] == ["2PSK", "4PSK"]


def test_prepare_cspb_inputs_downloads_and_extracts_missing_files(tmp_path: Path) -> None:
    delivered = tmp_path / "delivered"
    metadata = tmp_path / "signal_record.txt"
    calls = []

    def fake_download(url: str, path: Path, force: bool = False) -> bool:
        calls.append((url, Path(path).name, force))
        if Path(path).suffix == ".txt":
            Path(path).write_text("1 bpsk 2 -1e-4 0.35 7 6 1.0 0.0\n")
        else:
            with zipfile.ZipFile(path, "w") as zf:
                zf.writestr("Batch_Dir_1/signal_1.tim", _tim_bytes(np.asarray([1 + 0j], dtype=np.complex64)))
        return True

    original = converter.download_file
    converter.download_file = fake_download
    try:
        result = converter.prepare_cspb_inputs(delivered, metadata, download_if_missing=True, batches=[1])
    finally:
        converter.download_file = original

    assert result["downloaded"] == 3
    assert result["extracted"] == 1
    assert metadata.exists()
    assert (delivered / "Batch_Dir_1" / "signal_1.tim").exists()
    assert [name for _, name, _ in calls] == [
        "signal_record.txt",
        "CSPB.ML_.2018R2_1.zip",
        "signal_31986.tim_.zip",
    ]


if __name__ == "__main__":
    test_read_tim_bytes_complex()
    with TemporaryDirectory() as tmp:
        test_parse_metadata_maps_labels_and_effective_osr(Path(tmp))
    with TemporaryDirectory() as tmp:
        test_discover_tim_sources_prefers_extracted_over_zip(Path(tmp))
    with TemporaryDirectory() as tmp:
        test_convert_cspb_dataset_end_to_end_with_zip_and_missing_rows(Path(tmp))
    with TemporaryDirectory() as tmp:
        test_prepare_cspb_inputs_downloads_and_extracts_missing_files(Path(tmp))
