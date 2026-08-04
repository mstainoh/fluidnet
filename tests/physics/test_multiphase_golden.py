"""Golden tests for Beggs & Brill against published results."""

import logging

import numpy as np
import scipy.constants as SPC

from fluidnet.physics.multiphase import _beggs_brill_detailed

logger = logging.getLogger(__name__)


def test_kermit_brown_example_4_7() -> None:
    """Example 4.7, Kermit Brown. Book reports Darcy-Weisbach f (= 4 * Fanning)."""
    qos = 10000 * SPC.barrel / SPC.day
    qgs = 10e6 * SPC.foot**3 / SPC.day
    D = 6 * SPC.inch
    P = 1700 * SPC.psi
    T = SPC.convert_temperature(180, "F", "K")

    sigma = 8.41
    eps = 6e-6 * SPC.foot
    mu_liquid = 0.97e-3
    mu_gas = 0.016e-3
    Bo = 1.197
    Bg = 0.0091
    Rs = 281 * SPC.foot**3 / SPC.barrel
    z = 0.853
    dos = 141.5 / (131.5 + 33) * 1000
    dgs_free = 0.70 * 1.225
    dgs_diss = 0.88 * 1.225

    rho_liquid = (dos + dgs_diss * Rs) / Bo
    rho_gas = dgs_free / z * (288.15 / T) * (P / SPC.atm)

    qo = qos * Bo
    qg = (qgs - qos * Rs) * Bg

    calc = _beggs_brill_detailed(
        liquid_mass_rate=qo * rho_liquid,
        gas_mass_rate=qg * rho_gas,
        rho_liquid=rho_liquid,
        rho_gas=rho_gas,
        mu_liquid=mu_liquid,
        mu_gas=mu_gas,
        D=D,
        inclination=1.0,
        roughness=eps,
        sigma=sigma,
    )

    # (expected, rel_tol); book gives Darcy-Weisbach f (= 4 * Fanning) and
    # rounds intermediates, hence the looser tolerance on f (same ~6.5%
    # discrepancy observed with the 2018 prototype).
    book = {
        "NFr": (3.81, 0.05),
        "ReNs": (3.15e5, 0.05),
        "f": (0.0228 / 4, 0.08),
        "fNs": (0.0155 / 4, 0.08),  # book reads f from a Moody chart
        "liquid_holdup": (0.530, 0.05),
    }

    logger.info("flow_regime: calc=%s book=intermittent", calc["flow_regime"])
    assert calc["flow_regime"] == "intermittent"

    for key, (expected, tol) in book.items():
        rel_err = abs(calc[key] - expected) / expected
        logger.info(
            "%s: calc=%.5g book=%.5g rel_err=%.4f (tol=%.2f)",
            key, calc[key], expected, rel_err, tol,
        )
        assert rel_err < tol, f"{key}: got {calc[key]:.5g}, book {expected:.5g}"

    dpg_book = 28 * SPC.psi / 144 / SPC.foot
    dpf_book = 1.17 * SPC.psi / 144 / SPC.foot

    gravity_err = abs(-calc["gradient"].gravity - dpg_book) / dpg_book
    logger.info(
        "gradient.gravity: calc=%.5g book=%.5g rel_err=%.4f (tol=0.05)",
        -calc["gradient"].gravity, dpg_book, gravity_err,
    )
    assert gravity_err < 0.05

    friction_err = abs(-calc["gradient"].friction - dpf_book) / dpf_book
    logger.info(
        "gradient.friction: calc=%.5g book=%.5g rel_err=%.4f (tol=0.10)",
        -calc["gradient"].friction, dpf_book, friction_err,
    )
    assert friction_err < 0.10


def test_checalc_case_no_payne() -> None:
    """checalc.com Beggs & Brill sample (no Payne correction)."""
    calc = _beggs_brill_detailed(
        liquid_mass_rate=4.75 / SPC.hour * 613.8,
        gas_mass_rate=9 / SPC.hour * 141.3,
        rho_liquid=613.8,
        rho_gas=141.3,
        mu_liquid=0.5e-3,
        mu_gas=0.02e-3,
        D=50e-3,
        inclination=np.sin(np.deg2rad(90)),
        roughness=0.0018e-3,
        sigma=28.0,
        payne_correction=False,
    )

    logger.info(
        "gradient.total=%.5g (expected < 0, vertical upflow)",
        calc["gradient"].total,
    )
    assert calc["gradient"].total < 0

    logger.info(
        "liquid_fraction=%.5g liquid_holdup=%.5g (expected Cl < Hl <= 1)",
        calc["liquid_fraction"], calc["liquid_holdup"],
    )
    assert calc["liquid_fraction"] < calc["liquid_holdup"] <= 1