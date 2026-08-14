"""Golden tests for the Z-factor correlations against a digitized
Standing-Katz chart.

Source: digitized Standing & Katz (1942) chart, dataset originally
prepared for and provided by the authors of:

  Kamyab, M., Sampaio Jr., J.H.B., Qanbari, F., Eustes III, A.W. (2010).
  "Using artificial neural networks to estimate the Z-Factor for natural
  hydrocarbon gases." Journal of Petroleum Science and Engineering, 73,
  248-257. DOI: 10.1016/j.petrol.2010.07.006

Redistributed as SK_data.xls in:
  https://github.com/f0nzie/zFactor.DL/blob/master/inst/extdata/SK_data.xls

Values below are linearly interpolated (np.interp) from the raw digitized
curve at each Tpr onto round Ppr targets. Interpolation error is
negligible relative to the ~0.3-1% digitization/reading error inherent
to any chart-derived Z-factor source (see Dranchuk & Abou-Kassem 1975,
JCPT, DOI: 10.2118/75-03-03, who report 0.585% avg. abs. error fitting
the same chart).

Scope: physics layer only (``z_hall_yarborough``/``z_dranchuk_abou_kassem``
in ``physics/gas_correlations/z_factor.py``), capa cero — no ``RealGas``/
``StateModel`` involved. ``rtol=3e-2`` throughout: digitization +
interpolation error of the source dataset, not correlation error (see
module docstring above).

Known near-critical limitation (``Tpr`` in ``{1.05, 1.10}``, ``xfail``):
both correlations are documented as valid for ``Tpr > 1.0`` and both rows
sit inside that nominal domain, but right against its lower edge, where the
Standing-Katz isotherm has the steep non-monotonic dip characteristic of
the near-critical region (e.g. ``Tpr=1.05``: ``Z`` drops to 0.26 then rises
back to 0.66 across the same isotherm). Neither correlation was fit to
resolve that shape at the stated tolerance — Dranchuk & Abou-Kassem's own
0.585% *average* error is not uniform across the chart, and
``scipy.optimize.newton`` (secant, no analytic derivative) fails to
converge outright for two of the DAK points there. The exact failing
``(Tpr, Ppr)`` sets differ slightly between the two correlations, so each
is marked separately below rather than sharing one set.
"""

import pytest

from fluidnet.physics.gas_correlations import z_dranchuk_abou_kassem, z_hall_yarborough

GOLDEN_Z_STANDING_KATZ = [
    # (Tpr, Ppr, Z_expected)
    (1.05, 0.5, 0.8315), (1.05, 1.0, 0.5923), (1.05, 1.5, 0.2553),
    (1.05, 2.0, 0.2794), (1.05, 3.0, 0.4055), (1.05, 5.0, 0.6621),

    (1.10, 0.5, 0.8575), (1.10, 1.0, 0.6694), (1.10, 1.5, 0.4272),
    (1.10, 2.0, 0.3698), (1.10, 3.0, 0.4398), (1.10, 5.0, 0.6712),

    (1.30, 0.5, 0.9181), (1.30, 1.0, 0.8387), (1.30, 1.5, 0.7582),
    (1.30, 2.0, 0.6870), (1.30, 3.0, 0.6244), (1.30, 5.0, 0.7207),

    (1.50, 0.5, 0.9500), (1.50, 1.0, 0.9021), (1.50, 1.5, 0.8596),
    (1.50, 2.0, 0.8256), (1.50, 3.0, 0.7764), (1.50, 5.0, 0.8099),

    (2.00, 0.5, 0.9837), (2.00, 1.0, 0.9704), (2.00, 1.5, 0.9581),
    (2.00, 2.0, 0.9475), (2.00, 3.0, 0.9384), (2.00, 5.0, 0.9542),

    (3.00, 0.5, 1.0036), (3.00, 1.0, 1.0066), (3.00, 1.5, 1.0100),
    (3.00, 2.0, 1.0142), (3.00, 3.0, 1.0229), (3.00, 5.0, 1.0492),
]


# (Tpr, Ppr) pairs where the correlation misses the 3% tolerance (or, for
# two Dranchuk-Abou-Kassem points, scipy.optimize.newton fails to converge
# outright) — see the near-critical note in the module docstring.
_HY_NEAR_CRITICAL = {(1.05, 1.5), (1.05, 2.0), (1.05, 3.0), (1.10, 1.5), (1.10, 2.0), (1.10, 3.0)}
_DAK_NEAR_CRITICAL = {
    (1.05, 1.5), (1.05, 2.0), (1.05, 3.0), (1.05, 5.0), (1.10, 1.5), (1.10, 3.0),
}

_NEAR_CRITICAL_REASON = (
    "Known near-critical limitation (Tpr close to 1.0) — see module docstring."
)


def _golden_cases(near_critical: set[tuple[float, float]]) -> list[object]:
    return [
        pytest.param(
            Tpr, Ppr, Z_expected,
            marks=pytest.mark.xfail(strict=True, reason=_NEAR_CRITICAL_REASON),
        )
        if (Tpr, Ppr) in near_critical
        else pytest.param(Tpr, Ppr, Z_expected)
        for Tpr, Ppr, Z_expected in GOLDEN_Z_STANDING_KATZ
    ]


@pytest.mark.parametrize("Tpr, Ppr, Z_expected", _golden_cases(_HY_NEAR_CRITICAL))
def test_hall_yarborough_vs_standing_katz(Tpr: float, Ppr: float, Z_expected: float) -> None:
    z = z_hall_yarborough(Ppr, Tpr)
    assert z == pytest.approx(Z_expected, rel=3e-2)


@pytest.mark.parametrize("Tpr, Ppr, Z_expected", _golden_cases(_DAK_NEAR_CRITICAL))
def test_dranchuk_abou_kassem_vs_standing_katz(Tpr: float, Ppr: float, Z_expected: float) -> None:
    z = z_dranchuk_abou_kassem(Ppr, Tpr)
    assert z == pytest.approx(Z_expected, rel=3e-2)
