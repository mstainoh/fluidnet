"""Cross-validation of Beggs & Brill against ``fluids`` (ChEDL), v1.3.1.

Reconstruido 2026-08-07 tras confirmar (``git log`` sin resultados) que el
archivo original nunca llegó a commitear pese a estar documentado en
``docs/design/physics-single-multiphase.md`` §3.2 y en el ROADMAP como
"Capa 0 completada". Ver ``docs/session-log.md`` 2026-08-07.

Alcance de la comparación (importante, no es un port 1:1 de todo
``fluids.two_phase.Beggs_Brill``):

- ``liquid_holdup`` y ``mixture_density`` — mismas fórmulas de literatura
  (mismos coeficientes a/b/c por régimen, mismas correcciones B(theta)),
  se comparan a tolerancia ajustada (``rtol=1e-6``).
- ``gradient.gravity`` y ``gradient.friction`` — se comparan contra
  ``fluids`` con ``acceleration=False`` (ver más abajo por qué) a
  ``rtol=1.5%``, tolerancia que absorbe la diferencia Chen (fluidnet) vs.
  Colebrook-White (fluids) en el factor de fricción.
- ``gradient.momentum`` — **no se cross-valida contra fluids**. fluidnet
  no usa la fórmula de aceleración de fluids (``Ek = Vsg*Vm*rhos/P``, que
  depende de presión absoluta); usa un modelo propio vía
  ``compressibility`` [1/Pa] como propiedad termodinámica de la mezcla
  (``eh = compressibility * rho_mix * v_mix**2``). Con
  ``compressibility=0.0`` (default de ``_beggs_brill_detailed``, y el
  valor usado en todos los casos de este archivo) el momentum da
  exactamente 0, así que ``gradient.total == gradient.gravity +
  gradient.friction`` y la comparación es directa. Llamamos a
  ``fluids.Beggs_Brill(..., acceleration=False)`` para excluir su término
  de aceleración y comparar manzanas con manzanas. Esto es una diferencia
  de diseño documentada, no un gap de cobertura — ver ADR §1
  (compressibility es estado, no flag).
- ``payne_correction=False`` en todos los casos: la corrección de Payne es
  exclusiva de fluidnet, ``fluids`` no la implementa (convención ya fijada
  en ``CLAUDE.md``).

Todos los casos golden usan el mismo sistema de fluido (D=100mm, aceite
liviano / gas) variando únicamente rates y ángulo para cubrir los 4
regímenes (segregated, transition, intermittent, distributed) en distintas
orientaciones (horizontal, uphill, vertical, downhill).
"""

import logging
from typing import cast
import numpy as np
import pytest
import scipy.constants as spc

from fluidnet.physics.multiphase import _beggs_brill_detailed

logger = logging.getLogger(__name__)

# Sistema de fluido común a los 8 casos golden.
RHO_LIQUID = 850.0     # kg/m3
RHO_GAS = 25.0         # kg/m3
MU_LIQUID = 3.0e-3     # Pa.s
MU_GAS = 1.3e-5        # Pa.s
SIGMA = 0.020          # N/m
ROUGHNESS = 4.6e-5     # m
D = 0.1                # m

# Valores generados corriendo fluids v1.3.1 localmente (ver
# docs/session-log.md 2026-08-07 para el script de generación). Formato:
# (mass_rate_liquid, mass_rate_gas, angle_deg) -> valores esperados de
# fluids con acceleration=False, payne_correction=False.
GOLDEN = {
    "segregated_horizontal": {
        "mass_rate_liquid": 0.04069 * (np.pi * D**2 / 4) * RHO_LIQUID,
        "mass_rate_gas": 0.23125 * (np.pi * D**2 / 4) * RHO_GAS,
        "angle_deg": 0.0,
        "flow_regime": "segregated",
        "liquid_holdup": (0.48851467545964866, 1e-6),
        "mixture_density": (428.02460725421014, 1e-6),
        "grad_gravity": (-0.0000, 1e-6),
        "grad_friction": (-2.3474, 0.015),
    },
    "segregated_downhill": {
        "mass_rate_liquid": 0.04069 * (np.pi * D**2 / 4) * RHO_LIQUID,
        "mass_rate_gas": 0.23125 * (np.pi * D**2 / 4) * RHO_GAS,
        "angle_deg": -8.0,
        "flow_regime": "segregated",
        "liquid_holdup": (0.1426873926758196, 1e-6),
        "mixture_density": (142.71709895755117, 1e-6),
        "grad_gravity": (194.7834, 0.015),
        "grad_friction": (-3.3973, 0.015),
    },
    "transition_horizontal": {
        "mass_rate_liquid": 0.11923 * (np.pi * D**2 / 4) * RHO_LIQUID,
        "mass_rate_gas": 0.30541 * (np.pi * D**2 / 4) * RHO_GAS,
        "angle_deg": 0.0,
        "flow_regime": "transition",
        "liquid_holdup": (0.5674935380966901, 1e-6),
        "mixture_density": (493.1821689297694, 1e-6),
        "grad_gravity": (-0.0000, 1e-6),
        "grad_friction": (-9.0300, 0.015),
    },
    "transition_downhill": {
        "mass_rate_liquid": 0.11923 * (np.pi * D**2 / 4) * RHO_LIQUID,
        "mass_rate_gas": 0.30541 * (np.pi * D**2 / 4) * RHO_GAS,
        "angle_deg": -15.0,
        "flow_regime": "transition",
        "liquid_holdup": (0.07246239792911707, 1e-6),
        "mixture_density": (84.78147829152158, 1e-6),
        "grad_gravity": (215.1879, 0.015),
        "grad_friction": (-21.8837, 0.015),
    },
    "intermittent_uphill": {
        "mass_rate_liquid": 1.13998 * (np.pi * D**2 / 4) * RHO_LIQUID,
        "mass_rate_gas": 3.53123 * (np.pi * D**2 / 4) * RHO_GAS,
        "angle_deg": 45.0,
        "flow_regime": "intermittent",
        "liquid_holdup": (0.37651458225699197, 1e-6),
        "mixture_density": (335.6245303620184, 1e-6),
        "grad_gravity": (-2327.3375, 0.015),
        "grad_friction": (-698.2258, 0.015),
    },
    "intermittent_vertical": {
        "mass_rate_liquid": 1.13998 * (np.pi * D**2 / 4) * RHO_LIQUID,
        "mass_rate_gas": 3.53123 * (np.pi * D**2 / 4) * RHO_GAS,
        "angle_deg": 90.0,
        "flow_regime": "intermittent",
        "liquid_holdup": (0.37651458225699197, 1e-6),
        "mixture_density": (335.6245303620184, 1e-6),
        "grad_gravity": (-3291.3523, 0.015),
        "grad_friction": (-698.2258, 0.015),
    },
    "distributed_horizontal": {
        "mass_rate_liquid": 1.41345 * (np.pi * D**2 / 4) * RHO_LIQUID,
        "mass_rate_gas": 12.00744 * (np.pi * D**2 / 4) * RHO_GAS,
        "angle_deg": 0.0,
        "flow_regime": "distributed",
        "liquid_holdup": (0.20901283106906945, 1e-6),
        "mixture_density": (197.4355856319823, 1e-6),
        "grad_gravity": (-0.0000, 1e-6),
        "grad_friction": (-2688.8865, 0.015),
    },
    "distributed_uphill": {
        "mass_rate_liquid": 1.41345 * (np.pi * D**2 / 4) * RHO_LIQUID,
        "mass_rate_gas": 12.00744 * (np.pi * D**2 / 4) * RHO_GAS,
        "angle_deg": 30.0,
        "flow_regime": "distributed",
        "liquid_holdup": (0.20901283106906945, 1e-6),
        "mixture_density": (197.4355856319823, 1e-6),
        "grad_gravity": (-968.0908, 0.015),
        "grad_friction": (-2688.8865, 0.015),
    },
}


def _rel_err(calc: float, expected: float) -> float:
    if expected == 0.0:
        return abs(calc - expected)
    return abs(calc - expected) / abs(expected)


@pytest.mark.parametrize("name", list(GOLDEN.keys()))
def test_golden_vs_fluids_pinned(name: str) -> None:
    """8 casos, valores pinneados generados una vez contra fluids v1.3.1.

    Corre sin tener `fluids` instalado (no importa el paquete). Cubre los
    4 regimenes (3 puros + transition) en horizontal/uphill/vertical/
    downhill. Ver docstring del módulo para qué se compara y qué no
    (momentum queda afuera, es modelo propio).
    """
    case = GOLDEN[name]
    calc = _beggs_brill_detailed(
        mass_rate_liquid=cast(float, case["mass_rate_liquid"]),
        mass_rate_gas=cast(float, case["mass_rate_gas"]),
        density_liquid=RHO_LIQUID,
        density_gas=RHO_GAS,
        viscosity_liquid=MU_LIQUID,
        viscosity_gas=MU_GAS,
        D=D,
        inclination=np.sin(np.deg2rad(cast(float, case["angle_deg"]))),
        roughness=ROUGHNESS,
        sigma=SIGMA,
        payne_correction=False,

        # compressibility=0.0 (default) -> momentum=0, total=gravity+friction
        compressibility_gas=0.0,
        compressibility_liquid=0.0,
    )

    logger.info("[%s] flow_regime: calc=%s fluids=%s",
                name, calc["flow_regime"], case["flow_regime"])
    assert calc["flow_regime"] == case["flow_regime"]

    for key, attr in [("liquid_holdup", "liquid_holdup"),
                       ("mixture_density", "mixture_density")]:
        expected, tol = cast(tuple[float, float], case[attr])
        err = _rel_err(calc[key], expected)
        logger.info("[%s] %s: calc=%.6g fluids=%.6g rel_err=%.2e (tol=%.2e)",
                    name, key, calc[key], expected, err, tol)
        assert err < tol, f"{name}/{key}: got {calc[key]:.6g}, fluids {expected:.6g}"

    g = calc["gradient"]
    for key, attr, val in [("gravity", "grad_gravity", g.gravity),
                            ("friction", "grad_friction", g.friction)]:
        expected, tol = cast(tuple[float, float], case[attr])
        err = _rel_err(val, expected)
        logger.info("[%s] gradient.%s: calc=%.6g fluids=%.6g rel_err=%.4f (tol=%.3f)",
                    name, key, val, expected, err, tol)
        assert err < tol, f"{name}/gradient.{key}: got {val:.6g}, fluids {expected:.6g}"

    # compressibility=0 -> invariante: total == gravity + friction exacto
    assert g.total == pytest.approx(g.gravity + g.friction)
    assert g.momentum == 0.0


@pytest.mark.parametrize("name", list(GOLDEN.keys()))
def test_against_fluids_live(name: str) -> None:
    """Mismos 8 casos, llamando a fluids en tiempo real.

    `pytest.importorskip` la saltea si `fluids` no está instalado. Detecta
    si una versión futura de `fluids` cambia el resultado y los valores
    pinneados de arriba quedaron desactualizados. Usa la API pública
    (`Beggs_Brill`, dP total con `acceleration=False`) en vez de funciones
    internas — no depende de que fluids mantenga estable su API privada.
    """
    fluids_tp = pytest.importorskip("fluids.two_phase")
    case = GOLDEN[name]

    calc = _beggs_brill_detailed(
        mass_rate_liquid=cast(float, case["mass_rate_liquid"]),
        mass_rate_gas=cast(float, case["mass_rate_gas"]),
        density_liquid=RHO_LIQUID,
        density_gas=RHO_GAS,
        viscosity_liquid=MU_LIQUID,
        viscosity_gas=MU_GAS,
        D=D,
        inclination=np.sin(np.deg2rad(cast(float, case["angle_deg"]))),
        roughness=ROUGHNESS,
        sigma=SIGMA,
        payne_correction=False,
    )

    m = cast(float, case["mass_rate_liquid"]) + cast(float, case["mass_rate_gas"])
    x = cast(float, case["mass_rate_gas"]) / m
    dp_fluids = fluids_tp.Beggs_Brill(
        m=m, x=x, rhol=RHO_LIQUID, rhog=RHO_GAS, mul=MU_LIQUID, mug=MU_GAS,
        sigma=SIGMA, P=1e5, D=D, angle=case["angle_deg"], roughness=ROUGHNESS,
        L=1.0, acceleration=False,
    )
    # fluids: dP = p_up - p_down (positivo = pérdida). fluidnet:
    # gradient.total = p_down - p_up por longitud (pérdida -> negativo).
    expected_total = -dp_fluids

    err = _rel_err(calc["gradient"].total, expected_total)
    logger.info("[%s] live: calc=%.6g fluids=%.6g rel_err=%.4f (tol=0.015)",
                name, calc["gradient"].total, expected_total, err)
    assert err < 0.015, (
        f"{name}: pinned value may be stale, got {calc['gradient'].total:.6g}, "
        f"live fluids {expected_total:.6g}"
    )


def test_holdup_within_physical_bounds() -> None:
    """Caso downhill empinado + Cl alto: fluids da holdup negativo, no clippea.

    fluidnet sí lo mantiene en [0, 1] (decisión propia, documentada en
    physics-single-multiphase.md §4 como desviación intencional). No es
    parte del set golden (los valores no son comparables 1:1 -- uno clippea
    y el otro no) sino un guard de invariante físico.
    """
    fluids_tp = pytest.importorskip("fluids.two_phase")

    mass_rate_liquid = 0.04069 * (np.pi * D**2 / 4) * RHO_LIQUID
    mass_rate_gas = 0.23125 * (np.pi * D**2 / 4) * RHO_GAS
    angle_deg = -30.0

    calc = _beggs_brill_detailed(
        mass_rate_liquid=mass_rate_liquid,
        mass_rate_gas=mass_rate_gas,
        density_liquid=RHO_LIQUID,
        density_gas=RHO_GAS,
        viscosity_liquid=MU_LIQUID,
        viscosity_gas=MU_GAS,
        D=D,
        inclination=np.sin(np.deg2rad(angle_deg)),
        roughness=ROUGHNESS,
        sigma=SIGMA,
        payne_correction=False,
    )

    logger.info("liquid_holdup (fluidnet, clipped): %.5f", calc["liquid_holdup"])
    assert 0.0 <= calc["liquid_holdup"] <= 1.0

    # Confirmar que fluids efectivamente da negativo con estos inputs (si
    # esto empieza a fallar, fluids cambió de comportamiento y el caso ya
    # no sirve como guard -- hay que buscar otro).
    m = mass_rate_liquid + mass_rate_gas
    x = mass_rate_gas / m
    qg, ql = x * m / RHO_GAS, (1 - x) * m / RHO_LIQUID
    area = np.pi * D**2 / 4
    Vsl, Vsg = ql / area, qg / area
    Vm = Vsl + Vsg
    Fr = Vm**2 / (spc.g * D)
    lambda_L = Vsl / Vm
    LV = Vsl * (RHO_LIQUID / (spc.g * SIGMA)) ** 0.25
    hl_fluids_raw = fluids_tp._Beggs_Brill_holdup(
        0, lambda_L, Fr, np.deg2rad(angle_deg), LV  # regime 0 = segregated
    )
    logger.info("liquid_holdup (fluids, unclipped): %.5f (expected < 0)", hl_fluids_raw)
    assert hl_fluids_raw < 0.0, (
        "fluids ya no da holdup negativo para este caso -- buscar otro "
        "ángulo/Cl para el guard test"
    )