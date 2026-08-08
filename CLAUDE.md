# fluidnet — contexto para Claude Code

## Qué es esto

Librería Python **graph-first** para simulación steady-state de redes de
fluido. El usuario trae su física (`Rate` polimórfico + `Fluid` + `loss_func`
pluggables); la librería aporta topología (`nx.DiGraph`), propagación,
solvers y calibración.

Objetivo del proyecto: **portfolio profesional** → publicación en JOSS o
SoftwareX. Esto pesa en cada decisión: API limpia y documentada > cantidad
de features. Repo: `github.com/mstainoh/fluidnet`, versión activa
`0.2.0.dev0` (rewrite; el prototipo viejo vive taggeado en `legacy-0.1`).

Antes de tocar código, leé `ROADMAP.md` y `docs/session-log.md` (estado
actual y último punto donde se quedó).

## Arquitectura (capas, no te saltees el orden de dependencias)

```
Solvers (forward_propagation | mass_balance | fitting)
Network (wrapper nx.DiGraph, sin física)
Rate (polimórfico) │ loss_func (Protocol: AlgebraicLoss / IntegralLoss)
Fluid (fábrica de FluidState) │ physics/ (gradientes puros)   ← capa cero
Result (contrato de salida, solo lo produce el solver)
```

- `Rate`, `Fluid` y `loss_func` **no conocen la red**.
- `physics/` y `Fluid` son **capa cero**: funciones puras SI→SI, no importan
  nada del paquete.
- `Network` **no conoce la física** (solo plumbing de grafo).
- `Result` lo produce **solo el solver**, nunca una loss function.

**Cadena de evaluación** (la ruta canónica de un dp):

```
Rate ──► composición (intensiva)
             │
             ▼
     Fluid(composición, P, T) ──► FluidState(ρ, μ, β, σ)
             │
             ▼
     physics.gradient(**kwargs) ──► GradientResult (Pa/m)
             │
             ▼
     loss_func integra sobre L ──► dp (Pa)
```

## Decisiones cerradas — no las reabras sin discutirlo primero

1. **SI estricto en el core.** Sin `pint`. Unidades de presentación solo en
   capa de I/O (pre/post-proceso), nunca dentro de `loss_func` ni solvers.
2. **`loss_func` recibe el `Rate` entero**, no atributos sueltos
   (`loss_func(rate: Rate, fluid: Fluid, **edge_attrs) -> float`). Para el
   caso trivial hay helpers que extraen escalares.
3. **`Rate` = extensivo + composición intensiva. Sin propiedades de fluido.**
   *(Reemplaza la versión previa de esta decisión, que ponía densidad y
   viscosidad adentro del `Rate`. Sesión de diseño 2026-08-07.)*
   Un `Rate` contiene solo cantidades **lineales bajo las operaciones de
   red**: una magnitud extensiva (`mass_rate`) y una composición intensiva.
   Razón: `Rate` debe soportar `__mul__` (scaling del optimizer) y `__add__`
   (mezcla en nodos). Con `density` adentro, `rate * 2` es ambiguo — habría
   que escribir una regla ad-hoc "escalá extensivos, dejá intensivos" dentro
   del álgebra, y eso se rompe con el tercer tipo de rate. Además mantiene
   `Rate` homogéneo numéricamente: un `ComplexRate` (fasores, red AC) entra
   sin tocar nada.
   - `__add__`: suma extensivos, promedia composición ponderada por caudal.
   - `__mul__`: escala extensivos, composición intacta.
   - El diferencial de mezcla de salmueras **no se pierde**: lo que se
     propaga por la red es composición, no propiedades. Las propiedades se
     derivan, nunca se almacenan.
4. **`Fluid` es una fábrica stateless de `FluidState`.** Capa cero, hermana
   de `physics/`. No es un objeto con estado ni con densidad congelada: es
   el **modelo** que sabe mapear `(composición, P, T) → (ρ, μ, β, σ)`.
   - Firma: `Fluid.get_state(*, pressure, temperature=None, composition=...)
     -> FluidState`. Kw-only, por coherencia con `physics/` (#10).
   - **Recibe la composición como dato crudo** (mapping / array), **nunca un
     objeto `Rate`**. Si recibiera el `Rate` dejaría de ser capa cero.
   - `physics/` nunca ve `T` ni composición: recibe propiedades ya evaluadas.
5. **`FluidState` = `NamedTuple(density, viscosity, compressibility, sigma)`,
   todos los campos requeridos.** Nada de `float | None`. `β = 0` es el valor
   físico exacto del agua, no una ausencia (ver la nota de `compressibility`
   como estado, no flag). Un `sigma=None` filtrándose a B&B es el mismo
   problema que un default físico invisible, y rompe el filtrado de kwargs
   por `signature`. Un fluido monofásico puede levantar `NotImplementedError`
   en `sigma` si se quiere ser explícito — pero no devolver `None`.
6. **`temperature` en la firma desde v0.2, default `None` = "no
   suministrado", nunca un valor implícito.** Todo fluido es en general
   función de `(P, T)`; se deja opcional para los casos en que `T` es
   irrelevante o ya está fijada. Reglas:
   - Fluido incompresible (agua): **ignora** `T`. `None` es legítimo porque
     el parámetro no aplica.
   - Gas isotérmico: `T` se fija **en construcción**
     (`IsothermalGas(T=323.15)`), no en el call site.
   - Fluido que necesita `T` en la llamada y recibe `None` → **`ValueError`
     explícito**. Jamás un default tipo `T = 288.15`.
   Consecuencia: v2 (perfiles de temperatura prescritos) es aditivo — el
   solver pasa `T` desde un atributo de nodo/edge en vez de `None`. Cero
   cambio de protocolo.
7. **El régimen `AlgebraicLoss`/`IntegralLoss` lo declara el `Fluid`**, no el
   usuario al registrar la loss ni un chequeo del solver. El `Fluid` es dueño
   del EOS, así que es el único que sabe si `∂ρ/∂P = 0`. Un fluido
   incompresible expone propiedades sin necesitar `P` y habilita
   `AlgebraicLoss`; un fluido compresible exige `P` y por lo tanto solo es
   compatible con `IntegralLoss`. La `loss_func` hereda el régimen del fluido
   que recibe. Coherente con "ningún solver adivina".
8. **Dueño del `Fluid`: red hasta v1.0, nodo en v1.5 — mismo mecanismo.**
   Resolución: `G.nodes[n].get('fluid', network.fluid)`. Un modelo de fluido
   por red (v0.2–v1.0) es el **caso degenerado** del fluido por nodo (v1.5,
   precalculado en la misma pasada topológica que los rates), no un mecanismo
   distinto que después se reemplaza.
   **Un edge toma el fluido de su nodo `upstream`.** La mezcla ocurre *en* el
   nodo receptor, nunca dentro del caño. En v1 es una identidad, pero la
   semántica queda escrita desde ahora.
9. **Dos protocolos de loss, no uno**: `AlgebraicLoss` (dp no depende de P
   absoluta) e `IntegralLoss` (integra dp/dL, necesita `p_boundary`).
   `IntegralLoss` se **declara** en v0.2 (interfaz documentada) pero se
   **implementa recién en v1.0**.
10. **Sin clases `Node`/`Edge`.** Atributos en los dicts nativos de
    networkx (`G.nodes[n]`, `G.edges[u,v]`). Vistas tabulares vía
    `to_frames()`.
11. **Diagnósticos vía decorador `@diagnostic`**, no `full_output` en la
    firma pública. El contrato público de `loss_func` es estrictamente
    `-> float`; el decorador registra intermedios (f, Re, v, régimen) en un
    canal lateral que el *solver* recolecta con `full_output=True`. La loss
    function nunca devuelve un dict.
12. **`pandas` es dependencia core** (no lazy-import): `Result.to_frames()`
    es scope de MVP.
13. **Argumentos keyword-only** en funciones donde un bug de orden de
    argumentos es estructuralmente probable (ya pasó una vez con
    `friction_factor` en el prototipo 2018 — no repetir).
14. **Versionado**: `version = "0.2.0.dev0"` no cambia durante desarrollo
    activo. Solo se bumpea a `0.2.0` en el momento del release, con
    `git tag -a v0.2` + `git push origin v0.2` explícito.
15. **Firmas de `physics/`: keyword-only completo.** Toda función de
    gradiente (`single_phase_gradient`, `beggs_brill_gradient`,
    `_beggs_brill_detailed`) es enteramente kw-only, sin excepción por
    cantidad de rates (monofásico 1 rate, B&B 2). Razón: `loss_func`
    despacha genéricamente (`gradient_fn(**kwargs)` filtrado por
    `signature`); cualquier variación de forma entre modelos obligaría a
    ramificar en el consumidor. Corolario sobre defaults: solo van en
    flags de modelo (`payne_correction`, `holdup_adj`), nunca en estado
    físico (`roughness`, `inclination`, `sigma`, densidades, viscosidades,
    `compressibility`) — un default físico es una hipótesis de modelado
    invisible.
16. **Nombre del módulo: `fluidnet/fluid.py` (singular), no `fluids.py`.**
    Colisiona visualmente con `fluids` (ChEDL), que es el oráculo de
    cross-validation importado en tests.
17. **La restricción DAG es del solver, no de la física.**
    `forward_propagation` necesita orden topológico para propagar; eso no
    implica que las redes de flujo sean acíclicas. Las redes de agua potable
    y los circuitos de edificios son mallados y pasivos, y un loop pasivo con
    losses monótonas sigue siendo un problema convexo con solución única
    (content de Millar). Loops llegan en v2.0. Al escribir docs o README, no
    presentar DAG como una propiedad del dominio.

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
- **Tests de comportamiento vectorial** (`test_flowmap_vectorized`,
  `test_detailed_scalar_contract_today`, `test_detailed_vectorized_over_rates`
  xfail) viven en `tests/physics/test_multiphase_vector_1.py`, archivo
  separado — **no** en `test_beggs_brill_vs_book.py` como decía originalmente
  la ADR §3.3. Decisión tomada en sesión de código 2026-07-31; ADR
  actualizada en la misma sesión.

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
- `tests/physics/test_multiphase_vector_1.py` — batería de tests de
  comportamiento vectorial (flowmap, contrato escalar, xfail de roadmap
  v0.5). Ver nota en "Convenciones de testing" arriba.
- `legacy-0.1` (branch) — prototipo viejo, código a rescatar (ver mapa de
  rescate en el ADR de arquitectura), no a extender directamente.

## Tipado (housekeeping, no arquitectura)

**Esta sección tiene precedencia sobre "Cómo trabajar §2" para errores de
mypy.** Los errores de tipado (`no-any-return`, `no-untyped-def`,
`import-untyped`, `arg-type` por float/array) son mecánicos, no decisiones
de diseño. No pares a preguntar ni greppees tests/docs antes de tipar —
aplicá directo:

- `ArrayLike` vive en `fluidnet/physics/types.py`. Importalo de ahí; no
  redefinas el alias por módulo ni uses `numpy.typing.ArrayLike` (es laxo,
  pensado para inputs).
- Funciones que devuelven `ArrayLike` con expresión aritmética numpy →
  `cast(ArrayLike, expr)` en el return.
- Parámetros sin anotar → anotá con el tipo obvio del uso (`float`,
  `ArrayLike`, etc.), sin buscar la firma "perfecta".
- Nivel funcional, no exhaustivo: no hace falta `@overload` ni generics
  finos. `float | npt.NDArray[np.float64]` alcanza.

**Excepciones — acá SÍ pará y preguntá** (son contrato, no housekeeping):

- `_beggs_brill_detailed`: firma escalar a propósito hasta v0.5. No la
  toques — ver `test_detailed_scalar_contract_today` y el
  `xfail(strict=True)` que es su spec de roadmap. Excepción dentro de la
  excepción: pasar sus parámetros a kw-only (decisión cerrada #15) **sí**
  es housekeeping autorizado — no toca el contrato de forma escalar, solo
  cómo se invoca. No lo tomes como pie para vectorizar de paso.
- Campos de `GradientResult` y de `FluidState`: pasar `float` → `ArrayLike`
  cambia el contrato de retorno de toda la capa cero. Decisión de diseño
  abierta.

## Entorno (resuelto, no re-investigar)

- Paquete tipado PEP 561: `py.typed` en `src/fluidnet/`, empaquetado vía
  `[tool.setuptools.package-data]`, con `packages`/`mypy_path` fijados en
  `[tool.mypy]`.
- Comando de chequeo canónico: **`python -m mypy`** (sin argumentos). No
  usar paths sueltos (`mypy src/...`): dan falsos `import-untyped` por el
  editable install. Si aparece uno, el problema es la invocación, no el
  código.
- scipy: `ignore_missing_imports` vía override en `pyproject.toml`. No
  instalar `scipy-stubs`.
