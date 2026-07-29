# fluidnet — contexto para Claude Code

## Qué es esto

Librería Python **graph-first** para simulación steady-state de redes de
fluido. El usuario trae su física (`Rate` polimórfico + `loss_func`
pluggables); la librería aporta topología (`nx.DiGraph`), propagación,
solvers y calibración.

Objetivo del proyecto: **portfolio profesional** → publicación en JOSS o
SoftwareX. Esto pesa en cada decisión: API limpia y documentada > cantidad
de features. Repo: `github.com/mstainoh/fluidnet`, versión activa
`0.2.0.dev0` (rewrite; el prototipo viejo vive taggeado en `legacy-0.1`).

Antes de tocar código, leé `ROADMAP.md` y `docs/session-log.md` (estado
actual y último punto donde se quedó).

## Arquitectura (4 capas, no te saltees el orden de dependencias)

```
Solvers (forward_propagation | mass_balance | fitting)
Network (wrapper nx.DiGraph, sin física)
Rate (polimórfico) │ loss_func (Protocol: AlgebraicLoss / IntegralLoss)
Result (contrato de salida, solo lo produce el solver)
```

- `Rate` y `loss_func` **no conocen la red**.
- `Network` **no conoce la física** (solo plumbing de grafo).
- `Result` lo produce **solo el solver**, nunca una loss function.

## Decisiones cerradas — no las reabras sin discutirlo primero

1. **SI estricto en el core.** Sin `pint`. Unidades de presentación solo en
   capa de I/O (pre/post-proceso), nunca dentro de `loss_func` ni solvers.
2. **`loss_func` recibe el `Rate` entero**, no atributos sueltos
   (`loss_func(rate: Rate, **edge_attrs) -> float`). Para el caso trivial
   hay helpers que extraen escalares.
3. **Densidad/viscosidad viven en el `Rate`**, no como atributo de red
   (corrige deuda de mineplanner: ahí `density` estaba mal puesto en la red).
4. **Dos protocolos de loss, no uno**: `AlgebraicLoss` (dp no depende de P
   absoluta) e `IntegralLoss` (integra dp/dL, necesita `p_boundary`).
   `IntegralLoss` se **declara** en v0.2 (interfaz documentada) pero se
   **implementa recién en v1.0/v2**.
5. **Sin clases `Node`/`Edge`.** Atributos en los dicts nativos de
   networkx (`G.nodes[n]`, `G.edges[u,v]`). Vistas tabulares vía
   `to_frames()`.
6. **Diagnósticos vía decorador `@diagnostic`**, no `full_output` en la
   firma pública. El contrato público de `loss_func` es estrictamente
   `-> float`; el decorador registra intermedios (f, Re, v, régimen) en un
   canal lateral que el *solver* recolecta con `full_output=True`. La loss
   function nunca devuelve un dict.
7. **`pandas` es dependencia core** (no lazy-import): `Result.to_frames()`
   es scope de MVP.
8. **Argumentos keyword-only** en funciones donde un bug de orden de
   argumentos es estructuralmente probable (ya pasó una vez con
   `friction_factor` en el prototipo 2018 — no repetir).
9. **Versionado**: `version = "0.2.0.dev0"` no cambia durante desarrollo
   activo. Solo se bumpea a `0.2.0` en el momento del release, con
   `git tag -a v0.2` + `git push origin v0.2` explícito.

## Convenciones de testing (capa physics)

- `fluids` (ChEDL) es oráculo de cross-validation, **dependencia de
  `[dev]` únicamente**, nunca runtime. Usar `pytest.importorskip` para
  skips limpios si no está instalado.
- `payne_correction=False` al comparar contra `fluids` (fluidnet la tiene,
  `fluids` no).
- Holdup: tolerancia `rtol 1e-6` (tight, independiente de fricción).
  Gradiente total: `rtol 1.5%` (absorbe diferencia Chen vs. Colebrook).
- `xfail(strict=True)` se usa como **spec ejecutable de roadmap** (p. ej.
  vectorización de `_beggs_brill_detailed` para v0.5) — no lo borres para
  "arreglar" un test rojo; es rojo a propósito hasta que se implemente.
- Golden tests: preferir literatura (Kermit Brown ej. 4.7) sobre valores
  pinneados de `fluids` cuando ambos estén disponibles.

## Cómo trabajar en una sesión de código

1. Empezá leyendo `ROADMAP.md` (scope de la etapa actual) y
   `docs/session-log.md` (último estado, próximo paso concreto).
2. **No implementes nada que no tenga firma/decisión cerrada.** Si algo no
   está en "Decisiones cerradas" de este archivo ni en el ADR
   correspondiente en `docs/design/`, para y preguntá — no asumas.
3. Commits atómicos, chicos, contra `main` limpia.
4. Al cerrar la sesión: actualizá `docs/session-log.md` (qué se cerró, qué
   quedó abierto, próximo paso en una frase) antes de terminar.

## Dónde está cada cosa

- `ROADMAP.md` — plan de 3 etapas (MVP v0.2 / v1 / v2), scope in/out por etapa.
- `docs/design/` — ADRs de arquitectura detallada (`architecture-v0.2.md`,
  `physics-single-multiphase.md`).
- `docs/session-log.md` — bitácora de sesiones, más reciente arriba.
- `legacy-0.1` (branch) — prototipo viejo, código a rescatar (ver mapa de
  rescate en el ADR de arquitectura), no a extender directamente.
