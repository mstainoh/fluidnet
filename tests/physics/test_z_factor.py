"""Sanity + vectorization tests for the Z-factor correlations."""

import numpy as np
import pytest

from fluidnet.physics.gas_correlations import z_dranchuk_abou_kassem, z_hall_yarborough


def test_hall_yarborough_dranchuk_abou_kassem_agree() -> None:
    """Both correlations fit the same Standing-Katz chart, so at a typical
    natural-gas condition they should land close to each other."""
    z_hy = z_hall_yarborough(1.5, 1.8)
    z_dak = z_dranchuk_abou_kassem(1.5, 1.8)

    assert z_hy == pytest.approx(z_dak, rel=2e-2)
    assert 0.5 < z_hy < 1.2


def test_hall_yarborough_vectorized_matches_scalar_loop() -> None:
    Ppr = np.array([0.5, 1.0, 1.5, 2.0, 3.0])
    Tpr = np.array([1.3, 1.5, 1.8, 2.0, 2.5])

    vectorized = z_hall_yarborough(Ppr, Tpr)
    looped = np.array([z_hall_yarborough(p, t) for p, t in zip(Ppr, Tpr, strict=True)])

    assert np.asarray(vectorized).shape == Ppr.shape
    np.testing.assert_allclose(vectorized, looped, rtol=1e-6)


def test_dranchuk_abou_kassem_vectorized_matches_scalar_loop() -> None:
    Ppr = np.array([0.5, 1.0, 1.5, 2.0, 3.0])
    Tpr = np.array([1.3, 1.5, 1.8, 2.0, 2.5])

    vectorized = z_dranchuk_abou_kassem(Ppr, Tpr)
    looped = np.array([z_dranchuk_abou_kassem(p, t) for p, t in zip(Ppr, Tpr, strict=True)])

    assert np.asarray(vectorized).shape == Ppr.shape
    np.testing.assert_allclose(vectorized, looped, rtol=1e-6)


def test_hall_yarborough_rejects_invalid_temperature_scalar() -> None:
    with pytest.raises(ValueError):
        z_hall_yarborough(1.0, 0.9)


def test_hall_yarborough_rejects_invalid_temperature_array() -> None:
    """`np.any` guard: a single invalid element in an array input must
    still raise, not silently produce a NaN/garbage array."""
    Ppr = np.array([1.0, 1.0, 1.0])
    Tpr = np.array([1.5, 0.9, 1.8])

    with pytest.raises(ValueError):
        z_hall_yarborough(Ppr, Tpr)
