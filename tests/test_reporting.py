from pathlib import Path

import numpy as np
import polars as pl

from modreczoo.reporting import build_prediction_table, error_slice_table, write_performance_explorer


def _metrics() -> dict:
    return {
        "accuracy": 0.5,
        "confusion": np.array([[1, 1], [1, 1]]),
        "y_true": np.array([0, 0, 1, 1]),
        "y_pred": np.array([0, 1, 0, 1]),
        "confidence": np.array([0.9, 0.8, 0.7, 0.6], dtype=np.float32),
        "nll_bits": np.array([0.1, 1.0, 2.0, 0.4], dtype=np.float32),
        "true_probability": np.array([0.9, 0.2, 0.3, 0.6], dtype=np.float32),
        "top2_pred": np.array([1, 0, 1, 0]),
        "top2_confidence": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
    }


def test_build_prediction_table_preserves_metadata_index_order() -> None:
    metadata = pl.DataFrame(
        {
            "signal_id": [10, 11, 12, 13, 14],
            "modulation": ["B", "A", "B", "A", "B"],
            "snr_db": [0.0, 4.0, 8.0, 12.0, 16.0],
        }
    )
    table = build_prediction_table(metadata, np.array([3, 1, 4, 0]), _metrics(), {0: "A", 1: "B"})

    assert table["signal_id"].to_list() == [13, 11, 14, 10]
    assert table["true_label"].to_list() == ["A", "A", "B", "B"]
    assert table["pred_label"].to_list() == ["A", "B", "A", "B"]
    assert table["correct"].to_list() == [True, False, False, True]
    assert table["error_pair"].to_list() == ["correct", "A->B", "B->A", "correct"]
    assert np.allclose(table["pred_margin"].to_numpy(), [0.8, 0.6, 0.4, 0.2])


def test_error_slice_table_ranks_metadata_correlated_errors() -> None:
    predictions = pl.DataFrame(
        {
            "signal_id": list(range(10)),
            "modulation": ["A"] * 10,
            "snr_db": [1.0] * 5 + [17.0] * 5,
            "channel": ["bad"] * 5 + ["good"] * 5,
            "correct": [False, False, False, False, True, True, True, True, True, False],
        }
    )
    slices = error_slice_table(predictions, min_count=2)

    row = slices.filter((pl.col("dimension") == "channel") & (pl.col("slice") == "bad")).row(0, named=True)
    assert row["errors"] == 4
    assert row["error_rate"] == 0.8
    assert row["error_lift"] > 1.0


def test_write_performance_explorer_smoke(tmp_path: Path) -> None:
    predictions = build_prediction_table(
        pl.DataFrame(
            {
                "signal_id": [0, 1, 2, 3],
                "modulation": ["A", "A", "B", "B"],
                "snr_db": [0.0, 4.0, 8.0, 12.0],
                "osr": [2.0, 2.0, 4.0, 4.0],
                "ebw": [0.2, 0.4, 0.6, 0.8],
            }
        ),
        np.arange(4),
        _metrics(),
        {0: "A", 1: "B"},
    )
    slices = error_slice_table(predictions, min_count=1)
    out = tmp_path / "report.html"
    write_performance_explorer(
        out,
        "test",
        predictions,
        slices,
        np.array([[1, 1], [1, 1]]),
        ["A", "B"],
        {"accuracy": 0.5, "ece": 0.1, "mce": 0.2},
    )

    html = out.read_text(encoding="utf-8")
    assert "test Performance Explorer" in html
    assert "Plotly.newPlot" in html
    assert "row-normalized" in html
    assert "SNR 0-4 dB" in html
    assert "autorange\":\"reversed" in html
    assert "Highest-Lift Error Slices" in html
    assert "Metadata Histogram" in html
    assert "Highest-Confidence Errors" in html


if __name__ == "__main__":
    import tempfile

    test_build_prediction_table_preserves_metadata_index_order()
    test_error_slice_table_ranks_metadata_correlated_errors()
    with tempfile.TemporaryDirectory() as tmp:
        test_write_performance_explorer_smoke(Path(tmp))
