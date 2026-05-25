import numpy as np

from modreczoo.simulation import (
    DEFAULT_PARAMS,
    apply_pulse_shape,
    generate_dataset,
    generate_symbols,
    in_band_noise_fraction,
    rational_resample_factors,
    rng_from_seed,
    sample_parameter_design,
)


def test_parameter_design_samples_requested_osr_and_rational_factors() -> None:
    params = dict(DEFAULT_PARAMS)
    params["seed"] = 3
    rows = sample_parameter_design(64, ("2PSK", "MSK"), params, rng_from_seed(3))

    assert all(row["symbol_period"] * row["osr"] >= 2 for row in rows)
    assert all(row["osr"] == row["upsample_factor"] / row["downsample_factor"] for row in rows)
    assert all(np.isclose(row["symbol_rate"], 1.0 / (row["symbol_period"] * row["osr"])) for row in rows)
    assert all("target_osr" not in row for row in rows)


def test_parameter_design_samples_osr_across_range() -> None:
    params = dict(DEFAULT_PARAMS)
    params.update(
        {
            "seed": 11,
            "sampler": "sobol",
            "osr_range": (3.0, 9.0),
        }
    )
    rows = sample_parameter_design(512, ("2PSK",), params, rng_from_seed(11))
    osr = np.asarray([row["osr"] for row in rows])

    assert osr.min() >= 3.0
    assert osr.max() <= 9.0
    assert len(set(osr)) >= 8
    assert np.quantile(osr, 0.1) <= 3.5
    assert np.quantile(osr, 0.9) >= 8.0


def test_rational_resample_factors_uses_coarse_approximation() -> None:
    upsample_factor, downsample_factor = rational_resample_factors(
        target_ratio=0.72,
        max_factor=5,
    )

    assert (upsample_factor, downsample_factor) == (3, 4)


def test_rational_resample_factors_respects_min_ratio() -> None:
    upsample_factor, downsample_factor = rational_resample_factors(
        target_ratio=0.2,
        min_ratio=0.5,
        max_factor=5,
    )

    assert upsample_factor / downsample_factor >= 0.5


def test_apply_pulse_shape_resamples_in_one_step_and_skips_srrc_for_msk() -> None:
    rng = rng_from_seed(4)
    symbols, _ = generate_symbols("4PSK", 32, rng)
    shaped = apply_pulse_shape(symbols, "4PSK", symbol_period=2, upsample_factor=5, downsample_factor=2, ebw=0.35)

    assert shaped.dtype == np.complex128
    assert len(shaped) > len(symbols)
    assert np.isclose(np.mean(np.abs(shaped) ** 2), 1.0)

    msk_symbols, _ = generate_symbols("MSK", 32, rng)
    msk = apply_pulse_shape(msk_symbols, "MSK", symbol_period=2, upsample_factor=4, downsample_factor=1, ebw=0.35)
    assert np.count_nonzero(np.abs(msk) > 1e-12) / len(msk) > 0.9

    # symbol_period>1: two-stage path — SRRC at symbol_period then pure resample
    shaped2 = apply_pulse_shape(symbols, "4PSK", symbol_period=4, upsample_factor=2, downsample_factor=1, ebw=0.35)
    assert shaped2.dtype == np.complex128
    assert len(shaped2) > len(symbols)
    assert np.isclose(np.mean(np.abs(shaped2) ** 2), 1.0)


def test_apply_pulse_shape_requires_two_samples_per_symbol() -> None:
    symbols = np.ones(4, dtype=np.complex128)
    try:
        apply_pulse_shape(symbols, "2PSK", symbol_period=2, upsample_factor=1, downsample_factor=3, ebw=0.35)
    except ValueError as exc:
        assert ">= 2" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_in_band_noise_fraction_uses_srrc_bandwidth() -> None:
    narrow = in_band_noise_fraction(osr=8.0, ebw=0.1)
    wide = in_band_noise_fraction(osr=8.0, ebw=1.0)

    assert np.isclose(narrow, 1.1 / 8.0)
    assert np.isclose(wide, 2.0 / 8.0)
    assert narrow < wide


def test_generate_dataset_metadata_records_rational_osr_approximation() -> None:
    params = dict(DEFAULT_PARAMS)
    params.update(
        {
            "n_samples": 128,
            "seed": 5,
            "sampler": "random",
            "symbol_period_range": (4, 5),
            "osr_range": (3.0, 4.0),
        }
    )

    signals, metadata, _ = generate_dataset(("2PSK",), 2, params, num_workers=1)

    assert signals.shape == (2, 128)
    rows = metadata.to_dicts()
    assert all(row["symbol_period"] == 4 for row in rows)
    assert all(row["osr"] == row["upsample_factor"] / row["downsample_factor"] for row in rows)
    assert all("target_osr" not in row for row in rows)
    assert all(np.isclose(row["symbol_rate"], 1.0 / (row["symbol_period"] * row["osr"])) for row in rows)


if __name__ == "__main__":
    test_parameter_design_samples_requested_osr_and_rational_factors()
    test_parameter_design_samples_osr_across_range()
    test_rational_resample_factors_uses_coarse_approximation()
    test_rational_resample_factors_respects_min_ratio()
    test_apply_pulse_shape_resamples_in_one_step_and_skips_srrc_for_msk()
    test_apply_pulse_shape_requires_two_samples_per_symbol()
    test_in_band_noise_fraction_uses_srrc_bandwidth()
    test_generate_dataset_metadata_records_rational_osr_approximation()
