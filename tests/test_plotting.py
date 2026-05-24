import numpy as np
import polars as pl

from modreczoo.plotting import iq_explorer_figure


def test_iq_explorer_figure_aligns_axes() -> None:
    signals = np.exp(1j * 2 * np.pi * 0.08 * np.arange(512, dtype=np.float32))[np.newaxis, :]
    metadata = pl.DataFrame(
        {
            "signal_id": [0],
            "modulation": ["2PSK"],
            "snr_db": [10.0],
            "osr": [2.0],
            "n_samples": [512],
        }
    )

    fig = iq_explorer_figure(signals, metadata, 0, nperseg=64, noverlap=48, nfft=128, width=900, height=640)

    assert len(fig.data) == 5
    assert fig.layout.xaxis.matches == "x3"
    assert fig.layout.yaxis2.matches == "y3"
    assert fig.layout.xaxis3.title.text == "sample index"
    assert fig.layout.yaxis3.title.text == "cycles/sample"
    assert fig.layout.width == 900
    assert fig.layout.height == 640
    assert "2PSK" in fig.layout.title.text


if __name__ == "__main__":
    test_iq_explorer_figure_aligns_axes()
