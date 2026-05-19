import numpy as np

from modreczoo.simulation import (
    DEFAULT_PARAMS,
    apply_pulse_shape,
    generate_dataset,
    generate_symbols,
    in_band_noise_fraction,
    rng_from_seed,
    sample_parameter_design,
)


def test_parameter_design_samples_rational_osr_above_one() -> None:
    params = dict(DEFAULT_PARAMS)
    params["seed"] = 3
    rows = sample_parameter_design(64, ("2PSK", "MSK"), params, rng_from_seed(3))

    assert all(row["upsample_factor"] >= 2 for row in rows)
    assert all(row["downsample_factor"] >= 1 for row in rows)
    assert all(row["upsample_factor"] / row["downsample_factor"] > 1 for row in rows)
    assert all(row["osr"] == row["upsample_factor"] / row["downsample_factor"] for row in rows)
    assert any(not float(row["osr"]).is_integer() for row in rows)


def test_apply_pulse_shape_resamples_in_one_step_and_skips_srrc_for_msk() -> None:
    rng = rng_from_seed(4)
    symbols, _ = generate_symbols("4PSK", 32, rng)
    shaped = apply_pulse_shape(symbols, "4PSK", upsample_factor=5, downsample_factor=2, ebw=0.35)

    assert shaped.dtype == np.complex128
    assert len(shaped) > len(symbols)
    assert np.isclose(np.mean(np.abs(shaped) ** 2), 1.0)

    msk_symbols, _ = generate_symbols("MSK", 32, rng)
    msk = apply_pulse_shape(msk_symbols, "MSK", upsample_factor=4, downsample_factor=1, ebw=0.35)
    assert np.count_nonzero(np.abs(msk) > 1e-12) / len(msk) > 0.9


def test_apply_pulse_shape_requires_real_oversampling() -> None:
    symbols = np.ones(4, dtype=np.complex128)
    try:
        apply_pulse_shape(symbols, "2PSK", upsample_factor=1, downsample_factor=1, ebw=0.35)
    except ValueError as exc:
        assert "upsample_factor" in str(exc)
    else:
        raise AssertionError("expected ValueError")

    try:
        apply_pulse_shape(symbols, "2PSK", upsample_factor=2, downsample_factor=2, ebw=0.35)
    except ValueError as exc:
        assert "greater than 1" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_in_band_noise_fraction_uses_srrc_bandwidth() -> None:
    narrow = in_band_noise_fraction(osr=8.0, ebw=0.1)
    wide = in_band_noise_fraction(osr=8.0, ebw=1.0)

    assert np.isclose(narrow, 1.1 / 8.0)
    assert np.isclose(wide, 2.0 / 8.0)
    assert narrow < wide


def test_generate_dataset_metadata_includes_up_down_and_float_osr() -> None:
    params = dict(DEFAULT_PARAMS)
    params.update(
        {
            "n_samples": 128,
            "seed": 5,
            "sampler": "random",
            "upsample_factor_range": (4, 5),
            "downsample_factor_range": (3, 4),
        }
    )

    signals, metadata, _ = generate_dataset(("2PSK",), 2, params, num_workers=1)

    assert signals.shape == (2, 128)
    assert metadata["upsample_factor"].to_list() == [4, 4]
    assert metadata["downsample_factor"].to_list() == [3, 3]
    assert metadata["osr"].to_list() == [4 / 3, 4 / 3]


if __name__ == "__main__":
    test_parameter_design_samples_rational_osr_above_one()
    test_apply_pulse_shape_resamples_in_one_step_and_skips_srrc_for_msk()
    test_apply_pulse_shape_requires_real_oversampling()
    test_in_band_noise_fraction_uses_srrc_bandwidth()
    test_generate_dataset_metadata_includes_up_down_and_float_osr()
