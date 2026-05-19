import numpy as np
import polars as pl

from modreczoo.evaluation import accuracy_by_osr_snr_levels, information_by_snr, information_summary


def test_information_summary_perfect_binary_confusion() -> None:
    summary = information_summary(
        confusion=np.array([[2, 0], [0, 2]]),
        nll_bits=np.array([0.25, 0.25, 0.25, 0.25]),
    )
    assert summary["n"] == 4
    assert np.isclose(summary["label_entropy_bits"], 1.0)
    assert np.isclose(summary["pred_label_mi_bits"], 1.0)
    assert np.isclose(summary["pred_label_mi_fraction"], 1.0)
    assert np.isclose(summary["conditional_entropy_true_given_pred_bits"], 0.0)
    assert np.isclose(summary["nll_bits"], 0.25)
    assert np.isclose(summary["mi_nll_lower_bound_bits"], 0.75)


def test_information_by_snr_bins_examples() -> None:
    df = information_by_snr(
        metadata=pl.DataFrame({"snr_db": [0.0, 1.0, 4.0, 5.0]}),
        test_idx=np.array([0, 1, 2, 3]),
        y_true=np.array([0, 1, 0, 1]),
        y_pred=np.array([0, 1, 1, 1]),
        nll_bits=np.array([0.2, 0.3, 1.5, 0.4]),
        n_classes=2,
        bin_width=4.0,
    )
    assert df["snr_bin_db"].to_list() == [0.0, 4.0]
    assert df["n"].to_list() == [2, 2]
    assert np.isclose(df["pred_label_mi_fraction"][0], 1.0)


def test_accuracy_by_osr_snr_levels_picks_low_mid_high_bins() -> None:
    df = accuracy_by_osr_snr_levels(
        metadata=pl.DataFrame(
            {
                "snr_db": [0.0, 0.5, 4.0, 4.5, 8.0, 8.5, 12.0, 12.5],
                "osr": [1.5, 2.5, 1.5, 2.5, 1.5, 2.5, 1.5, 2.5],
            }
        ),
        test_idx=np.arange(8),
        y_true=np.array([0, 0, 0, 0, 0, 0, 0, 0]),
        y_pred=np.array([0, 1, 1, 1, 0, 0, 0, 1]),
        bin_width=4.0,
    )
    assert df["snr_bin_db"].unique().sort().to_list() == [0.0, 8.0, 12.0]
    assert df["osr"].to_list() == [1.5, 2.5, 1.5, 2.5, 1.5, 2.5]
    assert df["accuracy"].to_list() == [1.0, 0.0, 1.0, 1.0, 1.0, 0.0]


if __name__ == "__main__":
    test_information_summary_perfect_binary_confusion()
    test_information_by_snr_bins_examples()
    test_accuracy_by_osr_snr_levels_picks_low_mid_high_bins()
