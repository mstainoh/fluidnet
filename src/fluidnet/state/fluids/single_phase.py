"""Single-phase ``FluidState`` + ``Fluid`` implementations.

Layer zero, stateless (``CLAUDE.md`` #4): ``Fluid`` maps
``(composition, P, T) -> FluidState``, never freezes a density. Receives
composition as raw data, never a ``Rate``. Multi-phase counterparts live in
a sibling module (``multiphase_fluids.py``, not yet written).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import NamedTuple, cast

import numpy as np
import numpy.typing as npt

from fluidnet._types import ArrayLike
from fluidnet.state.protocol import BoundStateModel


class SinglePhaseFluidState(NamedTuple):
    """Mono-phase ``FluidState``, bare field names (#19).

    Field names match the canonical kwargs of
    ``fluidnet.physics.single_phase.single_phase_gradient`` (#21) — the
    filtering-by-signature in ``loss_func`` depends on this.
    """

    density: ArrayLike
    viscosity: ArrayLike
    compressibility: ArrayLike

    def as_physics_kwargs(self) -> dict[str, ArrayLike]:
        return self._asdict()


class IncompressibleFluid:
    """Constant-property fluid — MVP baseline (ROADMAP Capa 0 bis, v0.2 scope).

    Ignores composition and temperature: properties are fixed at
    construction, not derived from an EOS. The bound state does not depend
    on ``x``/``across`` — the degenerate, scalar branch of #26.
    """

    def __init__(self, *, density: float, viscosity: float, compressibility: float = 0.0) -> None:
        self._state = SinglePhaseFluidState(
            density=density, viscosity=viscosity, compressibility=compressibility
        )

    def bind(
        self,
    ) -> BoundStateModel[SinglePhaseFluidState]:
        state = self._state

        def bound(*, x: float, across: ArrayLike) -> SinglePhaseFluidState:
            return state

        return bound


class CompressibleFluidBase(ABC):
    """Pressure-dependent fluid: density/viscosity/compressibility are an EOS
    evaluated per bound state, not fixed at construction (#4).

    Composition is bound as early as the case allows. A fluid whose EOS
    parameters are fixed for the network resolves them in ``__init__``
    (e.g. ``IdealGas(molecular_weight=...)``) and its ``bind`` takes no
    composition at all. A compositional fluid (v1.5, composition comes from
    ``propagate_rates`` at runtime) overrides ``bind`` to accept it and
    resolves the EOS parameters once there, before returning the closure.
    Either way the hot loop never sees it: composition is constant along the
    edge in steady state with no mass exchange (#28). The base class does
    not know which case applies, so it declares neither.

    ``temperature`` in ``bind`` follows #26: ``None``/a scalar and a
    ``Callable[[float], float]`` are the two sibling cases, discriminated
    once in ``bind`` by ``callable(temperature)`` — no intermediate wrapper
    type. A callable is re-evaluated at ``x`` on every step (prescribed
    profile); ``None``/a scalar is captured once and ``x`` is ignored.

    ``viscosity`` is concrete here, not abstract (#21): every subclass
    delegates to a constructor-supplied ``viscosity_fn(pressure,
    temperature, **injectables) -> mu`` correlation, and the property
    injection (``density``, ``molecular_weight``, and any reduced
    properties) is identical machinery regardless of the EOS. A subclass
    sets ``self.molecular_weight``/``self.viscosity_fn`` in its
    ``__init__`` and overrides :meth:`_reduced_injectables` (and
    :attr:`uses_reduced_properties`) only if it has reduced properties to
    offer — ``RealGas`` does, ``IdealGas`` doesn't.
    """

    molecular_weight: float
    viscosity_fn: Callable[..., ArrayLike]

    @abstractmethod
    def density(self, *, pressure: ArrayLike, temperature: float | None = None) -> ArrayLike:
        """Return density at given P, T (EOS)."""
        ...

    @abstractmethod
    def compressibility(
        self, *, pressure: ArrayLike, temperature: float | None = None
    ) -> ArrayLike:
        """Return isothermal compressibility at given P, T (EOS)."""
        ...

    @property
    def uses_reduced_properties(self) -> bool:
        """bool: Whether this EOS's correlations are evaluated at reduced
        properties (``P/Pc``, ``T/Tc``) rather than absolute ``(P, T)``.
        ``False`` unless a subclass that has pseudo-critical properties
        overrides it (e.g. ``RealGas`` when both ``Pc`` and ``Tc`` are
        given).
        """
        return False

    def _reduced_injectables(
        self, *, pressure: ArrayLike, temperature: float
    ) -> dict[str, ArrayLike]:
        """Reduced-property injectables for ``viscosity_fn`` (#21).

        Hook for subclasses that support reduced properties. Empty by
        default — a subclass overrides it together with
        :attr:`uses_reduced_properties`.

        Parameters
        ----------
        pressure : ArrayLike
            Pressure [Pa].
        temperature : float
            Temperature [K].

        Returns
        -------
        dict[str, ArrayLike]
            Empty unless overridden.
        """
        return {}

    def _viscosity_injectables(
        self, *, pressure: ArrayLike, temperature: float
    ) -> dict[str, ArrayLike]:
        """Properties this class already knows or has computed, offered to
        ``viscosity_fn`` by keyword (``CLAUDE.md`` #21).

        Parameters
        ----------
        pressure : ArrayLike
            Pressure [Pa].
        temperature : float
            Temperature [K].

        Returns
        -------
        dict[str, ArrayLike]
            ``density`` and ``molecular_weight`` always, plus whatever
            :meth:`_reduced_injectables` adds (e.g. ``pressure_reduced``,
            ``temperature_reduced`` when :attr:`uses_reduced_properties`).
        """
        injectables: dict[str, ArrayLike] = {
            "density": self.density(pressure=pressure, temperature=temperature),
            "molecular_weight": self.molecular_weight,
        }
        injectables.update(self._reduced_injectables(pressure=pressure, temperature=temperature))
        return injectables

    def viscosity(self, *, pressure: ArrayLike, temperature: float | None = None) -> ArrayLike:
        """Viscosity, delegated to the constructor's ``viscosity_fn``
        correlation (#21).

        An ideal EOS does not imply a temperature-only viscosity model:
        ``mu = f(T)`` (e.g. Sutherland) holds in the dilute-gas limit,
        which is a property of the chosen correlation, not of the EOS.
        ``IdealGas`` paired with a density-dependent correlation (e.g.
        Lee-Gonzalez-Eakin) is a valid, correct combination — ``density``
        here is still the ideal-gas density from :meth:`density`.

        Parameters
        ----------
        pressure : ArrayLike
            Pressure [Pa].
        temperature : float, optional
            Temperature [K]. Required — ``None`` raises ``ValueError``.

        Returns
        -------
        ArrayLike
            Dynamic viscosity [Pa.s], from ``viscosity_fn(pressure,
            temperature, **injectables)`` — see
            :meth:`_viscosity_injectables` for what is injected.
        """
        if temperature is None:
            raise ValueError(
                f"Temperature must be provided for {type(self).__name__} viscosity calculation."
            )
        injectables = self._viscosity_injectables(pressure=pressure, temperature=temperature)
        return self.viscosity_fn(pressure, temperature, **injectables)

    def _state_at(self, *, pressure: ArrayLike, temperature: float | None) -> SinglePhaseFluidState:
        return SinglePhaseFluidState(
            density=self.density(pressure=pressure, temperature=temperature),
            viscosity=self.viscosity(pressure=pressure, temperature=temperature),
            compressibility=self.compressibility(pressure=pressure, temperature=temperature),
        )

    def bind(
        self,
        *,
        temperature: float | Callable[[float], float] | None = None,
    ) -> BoundStateModel[SinglePhaseFluidState]:
        if callable(temperature):
            profile = temperature

            def bound_callable(*, x: float, across: ArrayLike) -> SinglePhaseFluidState:
                P = cast(npt.NDArray[np.float64], across)[0]
                return self._state_at(pressure=P, temperature=profile(x))

            return bound_callable

        T = temperature

        def bound_fixed(*, x: float, across: ArrayLike) -> SinglePhaseFluidState:
            P = cast(npt.NDArray[np.float64], across)[0]
            return self._state_at(pressure=P, temperature=T)

        return bound_fixed
