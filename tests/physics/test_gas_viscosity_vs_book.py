"""Golden test for Lee-Gonzalez-Eakin viscosity against a published example."""

import pytest
import scipy.constants as spc

from fluidnet.physics.gas_correlations import lee_gonzalez_eakin_viscosity
from fluidnet.physics.gas_correlations.viscosity import _lee_gonzalez_eakin_detailed


def test_ahmed_example_2_14() -> None:
    """Ahmed, T.H., "Reservoir Engineering Handbook", Example 2-14.

    P=2000 psia, T=600 R, MW=20.85 lb/lb-mol, Z=0.78 -> rho=8.3 lb/ft3,
    mu=0.0173 cP. Book intermediates: K=119.72, X=5.35, Y=1.33.

    rtol=1e-2 throughout: the book works from rho and K/X/Y already
    rounded to 2-3 significant figures, so a tighter tolerance would be
    testing the book's rounding, not the correlation.
    """
    molecular_weight = 20.85e-3  # kg/mol
    temperature = spc.convert_temperature(600, "Rankine", "Kelvin")
    density = 8.3 * spc.lb / spc.foot**3  # kg/m3, from the book's rho=8.3 lb/ft3

    calc = _lee_gonzalez_eakin_detailed(
        pressure=0.0,  # unused by this correlation
        temperature=temperature,
        density=density,
        molecular_weight=molecular_weight,
    )

    assert calc["K"] == pytest.approx(119.72, rel=1e-2)
    assert calc["X"] == pytest.approx(5.35, rel=1e-2)
    assert calc["Y"] == pytest.approx(1.33, rel=1e-2)
    assert calc["viscosity"] == pytest.approx(0.0173e-3, rel=1e-2)  # cP -> Pa.s


def test_public_wrapper_matches_detailed() -> None:
    """`lee_gonzalez_eakin_viscosity` must be exactly the `viscosity` key of
    `_lee_gonzalez_eakin_detailed` — same book case, no golden values here,
    just internal consistency of the wrapper/detailed pair."""
    molecular_weight = 20.85e-3
    temperature = spc.convert_temperature(600, "Rankine", "Kelvin")
    density = 8.3 * spc.lb / spc.foot**3

    detailed = _lee_gonzalez_eakin_detailed(
        pressure=0.0, temperature=temperature, density=density, molecular_weight=molecular_weight
    )
    wrapped = lee_gonzalez_eakin_viscosity(
        pressure=0.0, temperature=temperature, density=density, molecular_weight=molecular_weight
    )

    assert wrapped == detailed["viscosity"]
