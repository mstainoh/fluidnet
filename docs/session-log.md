# Session log

> Bitácora de sesiones (diseño y código). Entrada más reciente arriba.
> Formato por entrada: fecha, modo (diseño/código), qué se cerró, qué
> quedó abierto, próximo paso concreto. En sesiones de código la escribe
> Claude Code al cierre; en sesiones de diseño la actualiza Marcelo o
> Claude (Desktop/chat) a mano.

---

## Plantilla (copiar para cada entrada nueva)

```
## AAAA-MM-DD — [diseño|código]

**Cerrado:**
-

**Abierto:**
-

**Próximo paso:**
-
```

---

## 2026-08-14 — código: `BaseRate` → ABC (`ScalarRateBase`/`VectorRateBase`, #35–#37)

**Cerrado:**

- `rate/base.py`: `BaseRate` pasa a `ABC`, `as_physics_kwargs` como
  `@abstractmethod`, sin `physics_key`/`physics_keys` propio. Dos
  hermanos sin herencia entre sí (#35): `ScalarRateBase` (`physics_key:
  ClassVar[str]`, contrato de un solo nombre) y `VectorRateBase`
  (`physics_keys: ClassVar[tuple[str, ...]]`, `__init__` valida
  `ndim == 0` o `shape[0] != len(physics_keys)` con `ValueError`
  explícito, `as_physics_kwargs` vía `dict(zip(physics_keys, value,
  strict=True))` — sin invertir el zip). Convención de eje `(n_cantidades,
  *shape_escenario)` documentada en el docstring de la clase (#36).
- `MassRate`, `VolumetricRate` (`rate/fluids/single_phase.py`) y
  `BrineRate` (`rate/fluids/brine.py`) migradas de `BaseRate` a
  `ScalarRateBase`. Ningún consumidor anotaba contra `BaseRate` fuera del
  re-export de `rate/__init__.py`, así que no hubo nada que reportar.
- Sin subclase concreta de `VectorRateBase` en el package todavía (spec
  #37, implementación pendiente).
- `tests/test_rate_base.py` nuevo (mirror de `test_rate_algebra.py`):
  dummies `_ScalarTestRate`/`_TwoPhaseTestRate` definidos en el test, no
  en el package. Cobertura: mapeo clave↔fila explícito contra el zip
  invertido, validación de construcción (escalar y `shape[0]` incorrecto),
  broadcast de split contra el eje de escenario (`(2,5) * (5,)`),
  `__add__` cruzado (`NotImplemented` directo + `TypeError` en ambos
  sentidos), `__array_ufunc__ = None` (`ndarray * rate`), `_rebuild`
  preserva tipo concreto en `__mul__`/`__neg__`/`__add__` para ambos
  hermanos.
- `python -m mypy` limpio (24 archivos, tests fuera de scope — mismo
  criterio que `test_rate_algebra.py`), suite completa en verde
  (`xfail(strict=True)` de roadmap incluidos), `lint-imports` 3/3.

**Abierto** → `ROADMAP §Abiertas`: primera subclase concreta de
`VectorRateBase` (rate multifásico real, #22).

**Próximo paso:** ninguno bloqueado por esta sesión.

---

## 2026-08-14 — Diseño: protocolo `Rate` y `BrineRate`

**Cerrado.** `Protocol` con `Self`; `physics_key` como `ClassVar`; sin
`__radd__`, `__sub__`, `mix()` ni `@runtime_checkable`. `BaseRate` como
conveniencia opcional (#34) con `__slots__`, `__array_ufunc__ = None` y el
par `_rebuild`/`_combine`. `MassRate`, `VolumetricRate` y `BrineRate` en
`rate/fluids/` (`single_phase.py` + `brine.py`, espejando
`state/protocol.py` + `state/fluids/`, #33). Composición como
`dict[str, ArrayLike]`, trazador pasivo, vacía es `{}`. Positividad
validada en `BrineRate.__init__`. Balance de nodo vía `reduce(add, rates)`.

**Desbloqueado.** `propagate_rates` — el álgebra de `Rate` está cerrada y
testeada.

**Abierto** → `ROADMAP §Abiertas`.

---

## 2026-08-14 — Diseño: protocolo `Rate`

**Cerrado.** Protocolo con `Self` en ambas posiciones de `__add__`;
`physics_key` como `ClassVar`; sin `__radd__`, sin `mix()`, sin
`@runtime_checkable`. `BaseRate` como conveniencia opcional (#34) en
`rate/base.py`, con `__slots__`, `__array_ufunc__ = None` y el par
`_rebuild`/`_combine`. Balance de nodo vía `reduce(add, rates)`.

**Desbloqueado.** `ScalarRate` — el spec está cerrado, se puede implementar.

**Abierto (a `ROADMAP §Abiertas`).** Si la mezcla ponderada de v1.5 necesita
una pasada única con acumuladores separados de extensivo e intensivo, el
fold por pares de `reduce` hay que revisarlo. Para suma ponderada estándar
no pierde precisión, así que no bloquea v0.2.

---

## 2026-08-14 — código: `rate/protocol.py` (protocolo `Rate`)

> Arquitectura resuelta en sesión de diseño con Claude web (no en esta
> sesión de código): forma del protocolo `Rate` — genérico con `Self` en
> vez de con `Rate`, nombre canónico como `ClassVar`, `__array_ufunc__ =
> None`, sin herencia de `float`/`ndarray`. El snippet de
> `rate/protocol.py` llegó ya escrito y el usuario lo aplicó directo al
> archivo antes de esta sesión; el trabajo acá fue mecánico: crear el
> paquete, transcribir el protocolo tal cual sin reinterpretarlo ni
> extenderlo, aplicar las correcciones dadas a `CLAUDE.md` #21/#22, y
> verificar.

**Cerrado:**

- `src/fluidnet/rate/` creado: `__init__.py` (exporta solo `Rate`) +
  `protocol.py` (protocolo `Rate`, ya transcripto por el usuario antes de
  esta sesión).
- **`__add__`/`__radd__`/`__mul__`/`__rmul__`/`__neg__` tipados con `Self`,
  no con `Rate`.** Razón: los parámetros de estos métodos son
  contravariantes — un protocolo que promete `other: Rate` obliga a toda
  implementación a aceptar cualquier `Rate`, y `BrineRate.__add__(other:
  BrineRate)` dejaría de satisfacerlo bajo `--strict`. Con `Self` el
  contrato es *me sumo con otro de mi propio tipo*: mypy rechaza
  `ScalarRate + BrineRate` estáticamente y el tipo concreto sobrevive
  `sum()` sin cast.
- **Nombre canónico (`physics_key`) como `ClassVar`, no campo de
  instancia.** Evita la tabla de renombres distribuida que #21 ya prohíbe
  centralizada — el nombre es propiedad del tipo (`MassRate` siempre
  entrega `mass_rate`), no de cada instancia.
- **`CLAUDE.md` #21**: línea cruzada agregada — corolario en `Rate`, el
  nombre canónico es `ClassVar` de la subclase (#22), nunca argumento de
  `__init__`.
- **`CLAUDE.md` #22**: cuatro sub-bullets agregados (firma con `Self`,
  `ClassVar` del nombre canónico, `__array_ufunc__ = None` en la base
  concreta, no heredar de `float`/`ndarray`) — texto dado por el usuario,
  pegado sin reformular.
- `python -m mypy`: limpio (21 archivos). `lint-imports`: 3/3 contratos en
  verde (`rate/` solo importa de `fluidnet._types`, ya declarado como capa
  opcional en `pyproject.toml`).

**Abierto:**

- **`CLAUDE.md` #22, referencia rota**: el bloque de "no heredar de
  `float`/`ndarray`" tal como se dio termina citando "#56" como la
  decisión que sostiene la frontera `Rate` vs. propiedad de fluido —
  `CLAUDE.md` no pasa de #34, así que la referencia no apunta a nada. Se
  pegó tal cual (instrucción explícita de no reformular); pendiente que el
  usuario confirme cuál era el número real.
- **Tres decisiones sin cerrar, señaladas por el propio chat de diseño,
  bloquean congelar el protocolo:**
  1. `__radd__` vs. `sum()`: `sum()` arranca en `int` (`0`), y `__radd__`
     tipado a `Self` no cubre ese caso — `sum(rates)` no tipa tal como
     está. Alternativas sobre la mesa: ensuciar la firma
     (`Self | Literal[0]`) o sacar `__radd__`/`sum()` y exponer
     `Rate.mix(iterable)` (preferencia del chat de diseño, sin cerrar) —
     que además sería el gancho natural para la mezcla ponderada de v1.5
     cuando el intensivo deje de sumar linealmente.
  2. `@runtime_checkable`: solo verifica presencia de métodos, no firmas —
     sirve para un `assert` de test, no como validación real. Sacarlo si
     no se va a usar en tests.
  3. `__slots__`: no va en el `Protocol` (no tiene instancias) — va en la
     ABC de conveniencia (`BaseRate`), que por #33 vive en el subpaquete
     concreto (`rate/scalar.py` o similar), no en `protocol.py`.
- No se creó ninguna implementación concreta (`ScalarRate`, `BrineRate`):
  fuera de alcance explícito de esta tarea.

**Próximo paso:**

- Cerrar la referencia rota en `CLAUDE.md` #22 y las tres decisiones de
  arriba (probablemente en sesión de diseño, no de código) antes de
  escribir `ScalarRate`.

---

## 2026-08-14 — código: pre-commit (ruff + lint-imports)

**Cerrado:**

- `.pre-commit-config.yaml` creado: `ruff-check --fix` + `ruff-format` (rev
  `v0.16.0`, la versión ya instalada) y hook local `lint-imports`.
  `pre-commit` agregado a `dev` en `pyproject.toml`. No se corrió
  `pre-commit install` (acción del operador, no de la sesión).
- `README_CLAUDE.md` creado (protocolo de trabajo, separado de `CLAUDE.md`):
  sección "Verificación" — automático por commit (`pre-commit`) vs. manual al
  cerrar sesión (`mypy`, `pytest`).
- `CLAUDE.md` #32: línea agregada — los contratos de `import-linter` corren
  automáticamente vía `pre-commit`; el comando canónico sigue siendo
  `lint-imports` sin argumentos.
- `ROADMAP.md`, ítem "CI mínimo": nota de puente — `pre-commit` local ya
  cubre `ruff` y `lint-imports` desde v0.2; lo que CI agrega y el puente no
  puede dar es el ambiente limpio y la matriz de versiones.
- `pre-commit run --all-files`: `lint-imports` pasó limpio (3/3). `ruff check
  --fix` + `ruff format` resolvieron la deuda de formato que la sesión de
  organización de imports (entrada de abajo) había dejado explícitamente
  fuera de alcance — 13 archivos reformateados, solo cosmético (indentación
  de firmas kw-only, newlines finales, líneas en blanco), sin cambios de
  lógica. Verificado aparte con `mypy` (18 archivos, limpio) y `pytest`
  (mismo resultado que antes del reformateo).
- Split en dos commits atómicos: setup de pre-commit + docs por un lado, el
  reformateo resultante de ruff por otro.

**Abierto:**

- CI (`.github/workflows/`) sigue sin existir — `pre-commit` es el puente
  declarado en `ROADMAP.md`, no el reemplazo.

**Próximo paso:**
- Retomar el roadmap de física/estado (v0.2) — ver entrada de golden
  Z-factor más abajo.

---

## 2026-08-14 — código: organización de módulos e imports

> Sesión mixta: instrucción mecánica inicial a Claude Code (`ArrayLike` a
> `fluidnet._types`, rename `CompressibleFluid` → `CompressibleFluidBase`,
> `single_phase_fluids.py` → `single_phase.py`, fachadas de `state/`,
> `import-linter`), pero buena parte de la ejecución terminó siendo del
> usuario en el editor en paralelo — Claude Code verificó/cerró lo que
> quedaba suelto en cada pasada (mypy, pytest, lint-imports).

**Cerrado:**

- `CLAUDE.md` #32-34: alias de tipo transversales en `fluidnet/_types.py`
  (capa −1); `physics/`/`state/` hermanos sin import lateral; política de
  `__init__.py` (fachada re-exporta, código interno importa del módulo que
  define); abstracto (`protocol.py`) vs. concreto (`state/fluids/`) como eje
  de profundidad, no de hermandad; `StateModel` es el contrato, las ABC
  (`CompressibleFluidBase`) son conveniencia opcional para implementadores.
- `ArrayLike` movido a `fluidnet._types` (capa −1), `physics/types.py` solo
  conserva `GradientResult`. Sweep completo de imports en `src/` y `tests/`
  (incluye un descuido real del primer commit del usuario: `physics/types.py`
  había quedado con `ArrayLike` duplicado en vez de importado).
- `CompressibleFluid` → `CompressibleFluidBase` en definición, subclases,
  tests y docstrings.
- `state/fluids/single_phase_fluids.py` → `single_phase.py`.
- Fachadas: `state/fluids/__init__.py` exporta también `CompressibleFluidBase`;
  `state/__init__.py` exporta el set completo (+ `IdealGas`/`RealGas`) y su
  docstring ya no menciona `IsothermalGas` (nombre que nunca existió).
  `fluidnet/__init__.py` reducido a `__version__` vía `importlib.metadata`.
  Dos tests (`test_single_phase_fluids.py`, `test_gas.py`) importaban de la
  fachada `fluidnet.state.fluids` en vez del módulo que define — corregido.
- `import-linter` agregado a `dev`, tres contratos en `pyproject.toml`
  (`layers` top-down con capas opcionales entre paréntesis, `independence`
  physics/state, `forbidden` capa cero vs. `networkx`/`pandas`).
  **Rojo→verde real, pero no por lo que se anticipaba**: la sintaxis de
  `layers` dada originalmente (`"state : physics"` sin `containers`) no
  valida contra `import-linter` 2.13 instalado — sin `containers` cada
  nombre de layer se busca como módulo top-level literal (`state`, no
  `fluidnet.state`); tocó pausar y reportar en vez de reescribir el
  contrato por cuenta propia (condición de parada explícita de la
  instrucción). El usuario probó una variante con `containers=["fluidnet"]`
  y capas obligatorias (creó `network/`, `rate/`, `solvers/` vacíos para
  que existieran) — falló por `fluidnet.losses` faltante. Se resolvió
  volviendo a capas opcionales (`"(solvers)"`, `"(network) : (losses)"`,
  `"(rate)"`) + `containers=["fluidnet"]` (necesario, no estaba en la
  sintaxis original) + `include_external_packages = true` (lo exige el
  contrato `forbidden` sobre paquetes externos) — sin materializar paquetes
  vacíos. Verde final: `Contracts: 3 kept, 0 broken.`
- Los cinco checks en verde: `mypy` (18 archivos), `pytest` (127 passed, 14
  xfailed), `lint-imports` (3/3). `ruff check`/`ruff format --check` tienen
  deuda preexistente sin relación con esta sesión (1 import sin ordenar en
  `test_beggs_brill_vs_fluids.py`, 15 archivos sin formatear) — no tocada,
  fuera de alcance.

**Abierto:**

- `_beggs_brill_detailed` sigue cruzando `physics/multiphase/__init__.py`
  tal cual estaba — pendiente de la decisión de diseño de diagnósticos
  nivel 1 (#25), no se tocó a propósito.
- No hay CI (`.github/workflows/`) donde enganchar `lint-imports` — queda
  como ítem de la sección "Infraestructura de repositorio" de `ROADMAP.md`.

**Próximo paso:**
- Retomar el roadmap de física/estado (v0.2) donde estaba antes de esta
  sesión de limpieza — ver la entrada de abajo (golden Z-factor).

---

## 2026-08-14 — código: golden Z-factor vs. Standing-Katz (Kamyab et al. 2010) + `RealGas.z()` bonus

> Continuación directa de la entrada de abajo (mismo día): cierra el ítem
> abierto "golden dataset real pendiente". El usuario dio el dataset
> completo (`GOLDEN_Z_STANDING_KATZ`, 36 puntos Tpr/Ppr/Z) con instrucciones
> mecánicas explícitas: crear el golden test siguiendo el patrón de
> `tests/test_multiphase_golden.py` — archivo que ya no existe con ese
> nombre (`git log` confirma el rename a `tests/physics/
> test_beggs_brill_vs_book.py` en el commit `a346df1`, 2026-08-07); se usó
> ese archivo actual como referencia de patrón. Instrucción explícita de
> parar sin inventar si `RealGas`/`StateModel` no expone `Z` como output de
> `bind()`/`__call__` — se paró, se reportó el estado real (ver hallazgo
> abajo), y el usuario redirigió el alcance: golden solo contra la capa
> `physics` (opción 1), más un test aparte en `state/` para `RealGas.z()`.

**Cerrado:**

- **Hallazgo reportado antes de escribir nada** (`CLAUDE.md` no tiene
  cerrado un campo `Z`/`compressibility_factor` en `FluidState`):
  `SinglePhaseFluidState` (#19/#21) solo tiene `density`/`viscosity`/
  `compressibility` — `Z` no es accesible vía `bind()`/`__call__`.
  `RealGas.z(*, pressure, temperature)` sí existe como método público
  directo (no vía `StateModel`), llama a `z_fn` en reducidas si hay
  `Pc`/`Tc`. No se inventó un wrapper para forzar `Z` al protocolo — el
  usuario decidió el alcance en base a este reporte.
- **`tests/physics/test_z_factor_vs_book.py`** (nuevo): golden real, capa
  `physics` pura (`z_hall_yarborough`/`z_dranchuk_abou_kassem`
  directamente, sin `RealGas`/`StateModel`), parametrizado sobre las 36
  filas de `GOLDEN_Z_STANDING_KATZ` (dataset + cita bibliográfica completa
  en el docstring del módulo, tal cual lo dio el usuario — Kamyab et al.
  2010, JPSE, digitalización del chart Standing-Katz 1942), `rtol=3e-2`
  (error de digitalización/interpolación del dataset fuente, no del
  modelo).
- **Región cercana al crítico (`Tpr` 1.05/1.10): 12 de 72 casos en rojo la
  primera corrida.** Patrón consistente, no ruido: las dos filas de `Tpr`
  más bajas del dataset, ambas `> 1.0` (dentro del dominio nominal
  declarado por las dos correlaciones — no es un caso fuera de rango
  colándose). Dos sub-tipos de falla: fuera de `rtol=3e-2` (la mayoría), y
  2 casos de Dranchuk-Abou-Kassem donde `scipy.optimize.newton` (secante)
  no converge en 50 iteraciones (`RuntimeError`) porque la raíz real
  (`Z≈0.68-0.95`) queda lejos del guess inicial `Z0=1.0` en esa zona no
  lineal. Confirmado con el usuario que ningún caso que falla tiene
  `Tpr <= 1.0` antes de proceder. Los 12 casos (sets distintos entre HY y
  DAK) quedan `xfail(strict=True)` con motivo documentado en el módulo —
  mismo patrón que otros `xfail` del repo, spec ejecutable de una
  limitación conocida de literatura (ninguna correlación fue ajustada para
  resolver la forma no monótona de la isoterma cerca del crítico al 3%).
  **60 pasan, 12 xfail.**
- **Bonus, `tests/state/test_gas.py`**: nueva clase
  `TestRealGasZWhenZIsNotOne` (hermana de
  `TestRealGasMatchesIdealGasWhenZIsOne`), dos tests separados (uno por
  correlación) que confirman `RealGas.z()` es pass-through exacto a
  `z_hall_yarborough`/`z_dranchuk_abou_kassem` en `(Ppr, Tpr)` reducidas
  (metano, `Pc=4.599 MPa`, `Tc=190.6 K`, lejos de la zona conflictiva de
  arriba). Test independiente del golden — verifica el wrapper de
  `RealGas`, no la precisión de la correlación.
- `pytest` verde (72 passed/xfailed en total sumando ambos archivos
  nuevos + el resto de la suite), `python -m mypy` limpio.

**Abierto:**

- Sin cambios respecto de la entrada de abajo (el resto del roadmap de
  gas compresible sigue igual).

**Próximo paso:**

- Sin instrucciones nuevas del usuario todavía.

---

## 2026-08-14 — código: refactor Z-factor (`gas_correlations/z_factor.py`) + vectorización

> Continuación de la sesión de viscosidad de gas (ver entrada de abajo, mismo
> día). `z_factor.py` existía como dos correlaciones (Hall-Yarborough,
> Dranchuk-Abou-Kassem) con docstrings en castellano, `math` en vez de
> `numpy`, y sin exportar en `__init__.py` — pedido inicial: refactor de
> forma (numpydoc inglés, `numpy`, export). Sobre la marcha, dos vueltas más
> pedidas por el usuario: sacar los loops de Newton hechos a mano (usar
> `scipy.optimize.newton`) y, ya que la implementación es toda `numpy`,
> asegurar que las firmas fueran `ArrayLike`/vectorizables en vez de
> `float`-only. Sesión no cerrada: sigue con la validación contra el golden
> dataset Kamyab et al. (2010) JPSE (36 puntos Tpr/Ppr/Z, digitalización del
> chart Standing-Katz), a instrucciones del usuario en el próximo paso.

**Cerrado:**

- **Refactor de forma**: docstrings numpydoc en inglés, `math.exp` →
  `np.exp`, exportadas en `gas_correlations/__init__.py` (`__all__`
  explícito junto a las de viscosidad).
- **Loops de Newton manuales → `scipy.optimize.newton`**: Hall-Yarborough
  usa la derivada analítica que ya estaba escrita (`fprime=dF`, sin cambio
  de fórmula); Dranchuk-Abou-Kassem tenía una derivada aproximada por
  diferencias finitas *manuales* (`dZ=1e-6`) — se saca esa aproximación y se
  deja el método de secante interno de `scipy.optimize.newton` resolver.
- **Firmas a `ArrayLike`** (`pressure_reduced`/`temperature_reduced`/
  return), consistente con `physics/types.py`. Tres ajustes necesarios para
  que de verdad vectorice y no solo tipe:
  1. el guard `Tpr <= 1.0` pasa a `np.any(np.asarray(Tpr) <= 1.0)` — con
     array, un solo elemento inválido sigue lanzando `ValueError` en vez de
     comparar ambiguamente o dejar pasar NaN;
  2. los `return float(...)` se sacan (rompían con arrays de tamaño > 1),
     reemplazados por `cast(ArrayLike, ...)`;
  3. en Dranchuk-Abou-Kassem, el guess inicial `Z0` pasa de constante
     `1.0` a `np.ones_like(Ppr * Tpr)` — necesario para que el guess tenga
     la forma broadcast correcta y dispare el path array de
     `scipy.optimize.newton` (que decide por `np.size(x0)`, no por el
     tamaño de lo que devuelven `F`/`f`).
  Housekeeping de tipos: dos `# type: ignore[arg-type]` puntuales en las
  llamadas a `newton()` — el stub de scipy tipa el overload array-only como
  `(ndarray, /, *Any, **Any) -> ndarray` y las closures internas (`F`, `dF`,
  `f`, marcadas positional-only) no calzan textualmente esa forma aunque
  son correctas en runtime para ambos casos (escalar y array).
- `tests/physics/test_z_factor.py` (nuevo, 5 tests): Hall-Yarborough y
  Dranchuk-Abou-Kassem coinciden entre sí en una condición típica (sanity de
  fórmula, no golden); llamada vectorizada == loop escalar elemento a
  elemento para ambas correlaciones; guard de `Tpr` dispara `ValueError`
  tanto en escalar como cuando un solo elemento de un array es inválido.
- `pytest` verde, `python -m mypy` limpio.

**Abierto:**

- **Golden dataset real pendiente**: Kamyab et al. (2010) JPSE, 36 puntos
  Tpr/Ppr/Z digitalizados del chart Standing-Katz — mencionado por el
  usuario, instrucciones todavía no dadas. Objetivo: reemplazar/complementar
  el sanity check cruzado HY-vs-DAK de `test_z_factor.py` con un golden real
  de literatura, mismo patrón que `test_beggs_brill_vs_book.py` /
  `test_gas_viscosity_vs_book.py`.
- La nota "Abierto" de la entrada de abajo (mismo día, sesión de viscosidad)
  decía que `z_factor.py` era "un placeholder vacío" — quedó desactualizada
  por este trabajo; no se reescribe la entrada histórica, se deja
  constancia acá.

**Próximo paso:**

- Validar Hall-Yarborough/Dranchuk-Abou-Kassem contra el golden
  Kamyab et al. (2010) cuando el usuario dé las instrucciones (dataset,
  tolerancias, formato del test).

---

## 2026-08-14 — código: inyección de propiedades en viscosidad de gas + fix `1e-4` en LGE

> Continuación de la sesión de `RealGas`/`IdealGas` (ver más abajo). Pedido
> inicial: actualizar la firma de `RealGas.viscosity` para que el callable
> reciba `(pressure, temperature, **injectables)` en vez de solo
> `(pressure, temperature)` (`CLAUDE.md` #21, nuevo). Al revisar el alcance
> el usuario pidió subir el mecanismo a `IdealGas` también (no era una
> decisión de excluirlo, era el alcance acotado del pedido original), y de
> paso surgieron dos bugs preexistentes.

**Cerrado:**

- **`CLAUDE.md` #21**: `viscosity_fn(pressure, temperature, **injectables)`.
  `CompressibleFluid` inyecta `density`, `molecular_weight`, y
  `pressure_reduced`/`temperature_reduced` si `uses_reduced_properties`.
  Kw-only sin default en la correlación para esos nombres — un typo falla
  con `TypeError`, no con un default silencioso. `**kwargs` catch-all
  válido para ignorar injectables no usados.
- **Mecanismo único en `CompressibleFluid`** (`state/fluids/
  single_phase_fluids.py`), no duplicado por subclase: `viscosity()` pasa
  de abstracto a concreto ahí; `uses_reduced_properties` (default `False`)
  y el hook `_reduced_injectables` (default `{}`) son lo único que
  `RealGas` overridea e `IdealGas` no. Nota física agregada al docstring:
  un EOS ideal no implica una correlación de viscosidad `T`-only —
  `IdealGas` + LGE (density-dependent) es combinación válida.
- **Bug real en `IdealGas`/`RealGas`, dos:** `IdealGas.viscosity` llamaba a
  `self._viscosity_fn`, atributo que no existía (el constructor guardaba
  `self._viscosity_fn` pero el nombre correcto tras el cambio es
  `viscosity_fn`); `RealGas._viscosity_injectables` usaba `self.Pr`/
  `self.Tr`, que tampoco existían (correcto: `self.Pc`/`self.Tc`). Ambos
  tiraban `AttributeError` en cualquier llamada a `.viscosity()`.
- **Bug real, `RealGas.compressibility`:** `dz_fn` se llamaba con
  `(pressure, temperature)` absolutos aun cuando `z_fn` se evaluaba en
  reducidas (`Pc`/`Tc` dados) — el par `z_fn`/`dz_fn` de la misma
  correlación veía convenciones de coordenadas distintas. Fix: helper
  `_reduced_pt` compartido por `z()` y `compressibility()`.
- **Rename `molar_weight` → `molecular_weight`** en `IdealGas`/`RealGas`
  (constructor + atributo): el injectable ya se llamaba `molecular_weight`
  (y también las correlaciones reales en `physics/gas_correlations/
  viscosity.py`) — mantener `molar_weight` en el constructor era la tabla
  de renombres que #29 prohíbe.
- **Bug real en `lee_gonzalez_eakin_viscosity`** (`physics/
  gas_correlations/viscosity.py`): faltaba el prefactor `1e-4` de la Ec.
  2-63 de Ahmed (`mu[cP] = 1e-4 * K * exp(X * rho^Y)`) — sin él el
  resultado quedaba 4 órdenes de magnitud de más. Encontrado al armar el
  golden test contra Ahmed, *Reservoir Engineering Handbook*, Example
  2-14 (verificado también fuera de sesión por el usuario). Se expuso
  `_lee_gonzalez_eakin_detailed` (K/X/Y + viscosity), mismo patrón que
  `_beggs_brill_detailed` (#25), y `lee_gonzalez_eakin_viscosity` pasa a
  ser un wrapper de una línea sobre ese dict.
  Golden test: `tests/physics/test_gas_viscosity_vs_book.py`
  (`rtol=1e-2` — el libro reporta rho e intermedios ya redondeados;
  K=119.72, X=5.35, Y=1.33, mu=0.0173 cP, todos verificados).
- `pytest` verde (65 tests), `python -m mypy` limpio (17 archivos).

**Abierto:**

- **Invariante de signo `∂μ/∂T` cruzando ~7 MPa** (GPSA/PetroSkills Fig.
  23-23), propuesto como test estructural adicional para LGE — descartado
  por ahora: requiere densidad de gas real (`Z < 1` en ese rango), y el
  repo no tiene un Z-factor implementado todavía (`physics/
  gas_correlations/z_factor.py` es un placeholder vacío). Con densidad
  ideal (`Z=1`) el cruce de signo existe pero aparece ~17-18 MPa, no ~7 —
  no vale la pena documentar ese número como si fuera el de la fuente. Se
  retoma cuando exista un Z-factor real en el repo.
- Sin cambios respecto de las entradas de abajo: `Rate`/`ScalarRate`
  pendiente.

**Próximo paso:**

- Sin cambios respecto de la entrada de abajo.

---

## 2026-08-13 — código: `BoundStateModel` genérico (fix `state.density` no tipaba)

> El usuario renombró `BoundState` → `BoundStateModel` a mano en
> `state/protocol.py` y estaba armando en paralelo
> `tests/state/test_gas.py` (ver entrada de abajo, escrita por esa sesión
> paralela). Reportó mypy en rojo: `state` no tenía `.density` donde debería.

**Cerrado:**

- **Import roto en `state/__init__.py`**: seguía con `from .protocol import
  BoundState, ...` tras el rename — `ImportError` en cualquier `import
  fluidnet.state.fluids` (bloqueaba `pytest` entero, no solo mypy).
  Corregido a `BoundStateModel`.
- **Causa real del "`state` no tiene `density`": `BoundStateModel`/
  `StateModel` no eran genéricos.** Todo `bind()` concreto devolvía el
  `BoundStateModel` desnudo; su `__call__` tipaba `-> State` (el `Protocol`
  neutro, solo `as_physics_kwargs()`). El tipo concreto se perdía en el
  camino aunque en runtime el objeto devuelto siempre fue
  `SinglePhaseFluidState` — reproducido con
  `python -m mypy tests/state/test_gas.py`: 7 errores
  `"State" has no attribute "density"/"viscosity"/"compressibility"`
  (`python -m mypy` sin argumentos no lo mostraba: por config
  `[tool.mypy] packages = ["fluidnet"]` no cubre `tests/`).
  Fix: `StateModel`/`BoundStateModel` pasan a `Protocol[S_co]` con
  `TypeVar("S_co", bound="State", covariant=True)`; `IncompressibleFluid.bind`
  y `CompressibleFluid.bind` anotan `-> BoundStateModel[SinglePhaseFluidState]`.
  Los 7 errores desaparecen sin tocar el cuerpo de ningún `bind`/`bound` —
  es un fix puramente de tipos.
- Docs actualizadas con el rename + la parametrización genérica:
  `CLAUDE.md` (#18, #26, #30), `ROADMAP.md` ("Capa 0 bis" y "Cerradas"),
  `docs/design/architecture-v0.2.md` §2.1bis (protocolo completo con el
  `TypeVar`). De paso, ese mismo ADR tenía el mismo desfasaje `float`/
  `ArrayLike` para `FluidState`/`GradientResult` que ya se había corregido
  en `CLAUDE.md`/ROADMAP/`physics-single-multiphase.md` en la sesión
  anterior — corregido ahí también.
- `python -m mypy` limpio (14 archivos) y `python -m mypy tests/state/
  test_gas.py tests/state/test_single_phase_fluids.py` limpio (2 archivos,
  0 errores — antes 7). `ruff check`/`format` limpios. `pytest tests/`
  verde.

**Abierto:**

- Sin cambios respecto de la entrada de abajo (`RealGas`/`IdealGas`):
  `RealGas.compressibility` sin verificar con `dZ/dP != 0` real,
  `Rate`/`ScalarRate` pendiente.

**Próximo paso:**

- Sin cambios respecto de la entrada de abajo.

---

## 2026-08-13 — código: test de `RealGas`/`IdealGas` (metano) + fix `RealGas.density` sin `molar_weight`

> Continuación directa de la sesión anterior del mismo día (`CompressibleFluid`
> EOS). Pedido: agregar tests de gas verificando densidad del metano a
> condiciones estándar (0°C, 1 atm) ≈0.717 kg/m³, viscosidad constante
> 10.84e-6 Pa·s, y que un `RealGas` con `z=1` constante (`dZ/dP=0`) dé
> resultados idénticos a `IdealGas`. Al armar la comparación se encontró que
> `RealGas` no tenía cómo dar ese resultado.

**Cerrado:**

- **Bug real en `RealGas.density` corregido: faltaba `molar_weight`.**
  Usaba `R_specific = spc.R` (constante universal, sin dividir por peso
  molecular) con el comentario "assuming molar weight is incorporated in
  z_fn" — con `z=1` eso devolvía densidad *molar* (44.6 mol/m³ a
  condiciones estándar), no másica. Verificado numéricamente: multiplicar
  ese resultado por el peso molecular del metano (0.016043 kg/mol) da
  0.7158 kg/m³, coincide con la densidad másica esperada — confirma que
  faltaba exactamente ese factor, no un error de otra naturaleza. Se agregó
  `molar_weight: float` al constructor de `RealGas` (mismo parámetro que
  `IdealGas`) y `density()` pasa a `R_specific = spc.R / self.molar_weight`,
  misma fórmula que `IdealGas`. **Cierra el ítem "Abierto" de la entrada
  anterior** ("`RealGas.density`: usa `R_specific = spc.R` ...
  dimensionalmente sospechoso"). `compressibility()` no necesitó cambios:
  β = 1/P − (1/Z)(dZ/dP) no depende de M (se cancela en la derivada
  logarítmica de ρ).
  Antes de tocar el código se preguntó al usuario cómo reconciliar el bug
  (CLAUDE.md #2, no implementar sin decisión cerrada, en vez de asumir un
  fix o un workaround solo en el test) — eligió arreglar `RealGas`.
- **`tests/state/test_gas.py` (nuevo).** `TestIdealGasMethane`: densidad
  del metano a 0°C/1 atm ≈0.717 kg/m³ (`rel=1e-2`, la fuente da valor
  redondeado), viscosidad constante, compresibilidad = 1/P.
  `TestRealGasMatchesIdealGasWhenZIsOne`: `RealGas(z_fn=1, dz_fn=0)`
  reproduce exactamente densidad/compresibilidad/viscosidad de `IdealGas`
  con el mismo `molar_weight`, y también da ≈0.717 kg/m³ standalone.
  Nota de implementación: `bound(x=..., across=...)` indexa `across[0]`
  (`CompressibleFluid.bind`, `single_phase_fluids.py`), así que los tests
  pasan `across=np.array([spc.atm])`, no un escalar.
- `python -m pytest tests/` verde (63 recolectados incl. los 7 nuevos, 2
  xfail sin cambios), `python -m mypy` limpio (14 archivos), `ruff
  check`/`format` limpios (`test_gas.py` reformateado por `ruff format`;
  los 3 errores preexistentes de `test_multiphase_vector_1.py` no se
  tocaron).

**Abierto:**

- **`RealGas.compressibility` sigue sin verificar contra literatura/`fluids`
  con `dZ/dP != 0`.** Hoy solo se probó el caso degenerado (`dZ/dP=0`),
  donde el término nuevo se anula y coincide con `IdealGas` por
  construcción — no prueba la fórmula completa.
- **`RealGas` con un z-factor de literatura real (Standing-Katz,
  Dranchuk-Abou-Kassem, etc.) sin probar.** Todo lo hecho hoy usa `z=1`
  constante como caso degenerado de verificación cruzada, no un fluido real.
- Sigue sin `Rate`/`ScalarRate` (bloqueante real de v0.2 según ROADMAP,
  postergado otra vez).

**Próximo paso:**

- Sesión próxima: (1) probar `RealGas` con una correlación de z-factor real
  (no `z=1` constante) contra un caso de literatura/`fluids`; (2) arrancar
  el esquema multifásico (`MultiPhaseState`, sufijos de fase — #19).

---

## 2026-08-13 — código: `FluidState` a `ArrayLike` + `IncompressibleFluid.bind()` sin parámetros

> Continuación directa de la sesión anterior del mismo día
> (`CompressibleFluid` + composición fuera de `bind`). El usuario cambió a
> mano `SinglePhaseFluidState` de `float` a `ArrayLike` (mismo criterio que
> `Fluid.bind`/`_state_at` ya usaban con `cast`) y sacó `composition`/
> `temperature` de `IncompressibleFluid.bind()` — extendiendo a
> `IncompressibleFluid` la misma lógica que ya se había aplicado a
> `CompressibleFluid.bind` (#28: si no lo usa, no lo declara). Pidió cerrar
> la decisión en los docs, no solo en el código.

**Cerrado:**

- **`SinglePhaseFluidState` campos `ArrayLike` (cierra CLAUDE.md #5).**
  `_state_at` ya no necesita los `cast(float, ...)` — se sacaron. Ver
  `CLAUDE.md` #5 y "Tipado", y ROADMAP "Cerradas" para el texto final.
- **Hallazgo al verificar el pedido de "mover `GradientResult` en la misma
  pasada": `GradientResult` nunca fue `float`.** Es `ArrayLike` desde su
  creación (`physics/types.py`, commit `9780d43`, nunca modificado desde
  entonces). La docs (`CLAUDE.md` "Tipado", el corolario de #30, ROADMAP
  "Abiertas", y el ADR `physics-single-multiphase.md` §1) describían un
  estado que el código no tenía hace rato — ni el tipo (`float` en vez de
  `ArrayLike`) ni la ubicación (`single_phase.py` en vez de
  `physics/types.py`, movido en el mismo commit `9780d43`). No fue
  causado por esta sesión, pero es exactamente el modo de falla que motivó
  el pedido: corregido en la misma pasada.
- **`IncompressibleFluid.bind()` pasa a tomar cero parámetros** (antes
  aceptaba `composition`/`temperature` y los ignoraba). Rompía 3 tests en
  `tests/state/test_single_phase_fluids.py` que se los pasaban
  explícitamente — actualizados:
  - `test_bind_ignores_composition_and_temperature` → renombrado
    `test_bind_takes_no_composition_or_temperature`: ahora verifica
    `TypeError` en vez de "ignorado en silencio" (el contrato cambió de
    "acepta y descarta" a "no declara").
  - `test_bind_is_pure_repeated_binds_are_independent` y
    `test_end_to_end_single_phase_gradient`: sacado `composition=` de las
    llamadas a `bind()`.
  - `test_bind_is_keyword_only` → renombrado
    `test_bind_takes_no_positional_args`: con cero parámetros ya no hay
    nada "kw-only" que probar; el test que queda es un guardia mínimo
    contra un positional-arg futuro.
- `python -m mypy` limpio (14 archivos, incluye `tests/`), `ruff
  check`/`format` limpios, `pytest tests/` verde (53 passed + 2 xfail).

**Abierto:**

- **`self._asdict()` devuelve `dict[str, Any]`, no `dict[str, ArrayLike]`.**
  Pasa por compatibilidad con `Any` sin que mypy lo verifique — la firma
  declarada en `as_physics_kwargs()` no está chequeada por nadie en este
  punto de contacto con `physics/`. Arreglarlo (construir el dict a mano
  con los tres nombres) cuesta repetir los nombres; no se hizo porque no
  vale la pena todavía.
- Sin cambios: tests de `CompressibleFluid`/`gas.py` sin escribir,
  `RealGas.compressibility` sin implementar, `Rate`/`ScalarRate` pendiente.

**Próximo paso:**

- Sin cambios respecto de la entrada anterior: decidir entre tests de
  `CompressibleFluid`/`gas.py` o `Rate`/`ScalarRate`.

---

## 2026-08-13 — código: `CompressibleFluid` (EOS) + composición fuera de `bind`

> Continuación directa de la sesión de código anterior del mismo día
> (`StateModel` protocol + `IncompressibleFluid` MVP). Arrancó revisando
> un WIP de `CompressibleFluid`/`temperature_profile.py` antes de seguir
> codeando; terminó en `src/fluidnet/state/fluids/gas.py`
> (`IdealGas`/`RealGas`) y en sacar `composition` de `bind` del todo.

**Cerrado:**

- **`temperature_profile.py` eliminado.** Introducía una jerarquía
  (`TemperatureProfile`/`FixedTemperatureProfile`/`NullTemperatureProfile`/
  `EdgeTemperatureProfile`) no especificada en ningún lado — #26 ya cierra
  la discriminación por `callable(field)` directo, sin wrapper
  intermedio. Además tenía un bug real: la rama de `isinstance(temperature,
  float)` capturaba antes que la de `TemperatureProfile`, y `None` caía al
  `else` y tiraba `ValueError` pese a ser el default documentado.
- **`CompressibleFluid` ahora hereda `ABC`** — antes `@abstractmethod` sin
  `ABC` no bloqueaba la instanciación directa. Los tres métodos EOS pasan
  de `return NotImplemented` (singleton de dunders, no de abstractos) a
  `...`.
- **`CompressibleFluid.bind` discrimina `temperature` por
  `callable(temperature)` (#26)**, dos closures hermanas
  (`bound_callable`/`bound_fixed`) decididas una vez en `bind`, sin tipo
  intermedio.
- **`composition` sale de `bind` (y de las tres firmas EOS) por completo**,
  no solo de las llamadas por-paso. Razón (#28 + corrección #18 del mismo
  día): la composición no depende de `x`, así que no tiene nada que hacer
  en el hot loop; pero además la clase base no sabe si el caso es "EOS fijo
  para toda la red" (se liga en `__init__`, como `IdealGas(molar_weight=...)`)
  o "composicional" (v1.5, sale de `propagate_rates` en runtime — resuelve
  ahí quien haga `override` de `bind`). Como no sabe cuál aplica, no declara
  ninguna. Extraído `_state_at(*, pressure, temperature)` como helper
  privado para no duplicar los tres `cast` entre las dos closures.
- **`gas.py` (`IdealGas`/`RealGas`) alineado a la firma nueva** y tres bugs
  reales corregidos de paso:
  - `IdealGas.__init__` nunca guardaba el `viscosity` del constructor;
    `viscosity()` se llamaba a sí mismo (`self.viscosity(...)`) →
    `RecursionError` garantizado en el primer uso. Guardado en
    `self._viscosity_fn`.
  - `IdealGas.compressibility` devolvía `1/(R_specific·T)` — unidades de
    ρ/P, no de 1/Pa. Corregido a `1/pressure` (para gas ideal β = 1/P
    exacto, se cancela todo lo demás).
  - `RealGas.z()` no anidaba bien el chequeo de `Pr`/`Tr` para que mypy
    angostara `float | None` — reescrito con el `if` directo sobre
    `self.Pr`/`self.Tr` en vez de pasar por la property
    `uses_reduced_properties`.
- **`RealGas.compressibility` deja de devolver `None` en silencio.**
  Delegaba a `super().compressibility(...)` (el stub abstracto), violando
  #5 (`FluidState` nunca `float | None`). Convertido a
  `raise NotImplementedError` explícito — la fórmula real (β = 1/P −
  (1/Z)(dZ/dP)_T, usando `self.dz_fn`, hoy guardado y sin usar) no se
  implementó porque no hay convención cerrada de qué devuelve `dz_fn`.
- `python -m mypy` limpio (14 archivos), `ruff check`/`format` limpios,
  `pytest tests/` verde (53 passed + 2 xfail, sin regresiones). Smoke test
  manual de `IdealGas.bind` en ambas ramas de #26 (temperatura fija y
  perfil callable) — no hay todavía tests formales de `gas.py`/
  `CompressibleFluid` en `tests/`.

**Abierto:**

- **Tests de `CompressibleFluid`/`gas.py` sin escribir.** `IncompressibleFluid`
  tiene su batería en `tests/state/test_single_phase_fluids.py`; `gas.py` no
  tiene ninguna todavía.
- **`RealGas.compressibility`**: fórmula sin implementar (ver arriba),
  bloqueada en la convención de `dz_fn`.
- **`RealGas.density`**: usa `R_specific = spc.R` (constante universal, no
  específica) con el comentario "assuming molar weight is incorporated in
  z_fn" — dimensionalmente sospechoso, no se tocó porque no está claro el
  diseño intencional (¿de dónde sale el peso molecular para un `RealGas`
  sin `molar_weight` en el constructor?). Candidato a resolverse junto con
  el ítem de composición→parámetros EOS de v1.5.
- **Scope**: `CompressibleFluid`/`IdealGas`/`RealGas` son v1.0 según el
  ROADMAP (`Scope v0.2: IncompressibleFluid ... es suficiente`), sin sesión
  de diseño que cierre su firma formalmente — se avanzó igual porque venía
  como "próximo paso" declarado en la entrada anterior del mismo día. Si se
  sigue construyendo sobre esto (p. ej. `IsothermalGas`), conviene una
  entrada de diseño que lo cierre explícito.
- `Rate`/`ScalarRate` sigue sin implementar.

**Próximo paso:**

- Decidir: ¿tests de `CompressibleFluid`/`gas.py` primero, o volver a
  `Rate`/`ScalarRate` (bloqueante real de v0.2 según el ROADMAP)?

---

## 2026-08-13 — código: `StateModel` protocol + `IncompressibleFluid` MVP

**Cerrado:**

- **Corrección de sesión de diseño 2026-08-13, aplicada a `CLAUDE.md` #18/#30
  y ADR §2.1bis** (commit `71af608` en `dev`, mergeado a esta rama sin
  conflictos): `composition` sale del `Protocol` neutro; la distinción
  propagado/prescrito se muda al solver; `across: ArrayLike` (no `float`,
  porque `solve_ivp` siempre entrega `ndarray`); `State.as_physics_kwargs()
  -> dict[str, ArrayLike]`. Ver también la tabla de los tres casos de
  vectorización (`vectorized=True` / escenarios apilados / potencial
  acoplado) que cierra esa sección del ADR.
- **`state/protocol.py`**: `StateModel`, `BoundState`, `State` como
  `Protocol`, ya con la firma corregida.
- **`state/fluids/single_phase_fluids.py`**: `SinglePhaseFluidState`
  (`NamedTuple` con `density`/`viscosity`/`compressibility` — nombres
  canónicos de `single_phase_gradient`, #21) y `IncompressibleFluid`
  (props constantes fijadas en `__init__`, ignora `composition`/
  `temperature`; `bind()` devuelve un `BoundState` que ignora `x`/`across`
  — la rama degenerada/escalar de #26). Es el MVP de Capa 0 bis declarado
  en el ROADMAP para v0.2.
- **Hallazgo técnico**: `typing.NamedTuple` no admite agregar campos vía
  subclase, ni combinarse con un mixin plano en el mismo `class` — se probó
  y ambos casos tiran `TypeError`. Se descartó el patrón `BaseState`/
  `BaseFluid` (que intentaba compartir `as_physics_kwargs()` por herencia);
  cada `State` concreto lo implementa inline sobre `self._asdict()`, que ya
  es gratis.
- **`get_state` sacado de `IncompressibleFluid`**: se había agregado
  siguiendo la firma de la decisión #4 (`Fluid.get_state(*, pressure,
  temperature=None, composition=...)`), pero no estaba conectado a
  `bind`/`bound` — el estado se cachea en `__init__`, no se deriva. Vuelve
  el día que exista un `Fluid` que sí derive `(P, T, composición) ->
  FluidState` (p. ej. `IsothermalGas`).
- **Batería de tests: `tests/state/test_single_phase_fluids.py` (10
  tests)**. Qué prueban:
  - `SinglePhaseFluidState`: construcción por keyword, inmutabilidad
    (`NamedTuple`), que `as_physics_kwargs()` emita exactamente los nombres
    canónicos `{density, viscosity, compressibility}`, y que ese dict entre
    sin fricción a `single_phase_gradient` (integración real, no mock).
  - `IncompressibleFluid`: `compressibility` default `0.0`; `bind()` ignora
    `composition`/`temperature` sin importar qué se le pase; el estado
    ligado es el mismo sin importar `x`/`across` (rama degenerada de #26);
    binds independientes no interfieren entre sí (`bind` es aplicación
    parcial pura, no muta el `Fluid`, #18); `bind` es keyword-only (#13);
    y un smoke test end-to-end `bind → BoundState → as_physics_kwargs() →
    single_phase_gradient`.
  - No cubierto todavía: la rama *callable* de #26 (perfil `T(x)`) —
    `IncompressibleFluid` ignora `temperature` por completo, así que no la
    ejercita. Queda pendiente para el próximo `Fluid` no constante.
- `python -m mypy` limpio (13 archivos), `ruff check`/`format` limpios,
  suite completa (`pytest tests/`) en verde: 53 passed + 2 xfail esperados.

**Abierto:**

- **`state/fluids/` (plural) vs. decisión #16.** Ver ROADMAP §Abiertas —
  no hay colisión de import real, pero sí de lectura (`import fluids` de
  ChEDL en tests vs. `fluidnet.state.fluids`). Falta decisión: reabrir #16
  para el caso anidado, o renombrar a `state/fluid/`.
- `MultiPhaseFluidState` / `multiphase_fluids.py` (mencionado en el
  docstring del módulo) — no empezado.
- `Rate`/`ScalarRate` sigue sin implementar (siguiente pieza según
  Secuencia inmediata #5).

**Próximo paso:**
- Decidido en cierre de sesión: siguiente `Fluid` a implementar es
  **compresible** (deriva `density` de `P`, ejercita la rama callable de
  #26 y el discriminador `AlgebraicLoss`/`IntegralLoss` de #7) y
  **multifásico** (`MultiPhaseFluidState`, convención de sufijos #19).
  `Rate`/`ScalarRate` queda pendiente en paralelo.

---

## 2026-08-10 — diseño: `StateModel` ligado por eje

**Cerrado:**

- **Nombre del `Protocol`: `StateModel`.** — cierra el ítem abierto del
  2026-08-09. Descartados `Medium` (sólo tiene sentido en continuos) y
  `ConstitutiveModel` (colisiona con `LossFunc`, que *es* la constitutiva —
  usar ese nombre acá vuelve ilegible el argumento de por qué son dos capas).
- **El `StateModel` tiene dos métodos, en dos tiempos distintos.**
  `bind(*, composition, **fields) -> BoundState`, una vez por eje; y
  `BoundState.__call__(*, x, across) -> State`, una vez por evaluación del
  gradiente. La "cadena anidada `comp → T → P`" cerrada el 2026-08-09 se
  concreta como **binding time**, no como nesting sintáctico: el orden es por
  frecuencia de cambio (composición constante en el eje → campos prescritos
  ligados por eje → `across` variable en cada paso).
- **`x` asciende al protocolo; los campos físicos no.** La coordenada del eje
  es domain-neutral: existe en todo dominio, ya está en el vocabulario del
  integrador (`t_span`), y no dice nada sobre qué física corre adentro. Es el
  único argumento que se puede agregar sin comprometer el diferencial
  physics-agnostic. Las dos alternativas se descartan por motivo explícito:
  `temperature` en la firma reabre la decisión #18; `**fields` en `__call__`
  muere con `mypy strict` y obliga a `loss_func` a saber qué campos existen en
  el dominio, contradiciendo "`loss_func` compone, no traduce".
- **`bind` es aplicación parcial, no construcción — por ownership, no por
  performance.** Si la composición entrara por `Fluid.__init__`, el `Fluid`
  dejaría de poder ser atributo declarativo de la red: la composición sale de
  `propagate_rates`, o sea de runtime. Con `bind` hay **un** `StateModel`
  declarativo por red/nodo y **un objeto liviano ligado por eje**.
- **Dos clases hermanas de `BoundState`, discriminadas por `callable(field)`
  en `bind`.** El corte no es "campo fijo vs. campo variable" sino **si el
  objeto ligado necesita `x`**: `None` y un escalar son el mismo caso (se
  guarda el valor, `x` se descarta); un callable es el otro. La rama se decide
  una vez, en `bind`. Motivo: evita que `None` viaje disfrazado de función a
  través de la capa más caliente del código. El ahorro de la llamada de Python
  es marginal en v0.2 (con `β = 0` corre `AlgebraicLoss`: **una** evaluación
  por eje, no 10²–10³).
- **Los campos prescritos son datos, no cantidades conservadas.** No se impone
  balance sobre ellos en los nodos. Si dos corrientes a distinta `T` se
  mezclan, reconciliar el perfil es hipótesis del usuario. Va declarado en
  README y ADR: un límite escrito se lee como criterio de modelado;
  descubierto por un reviewer, se lee como bug.
- **Forma de `FluidState`: `NamedTuple`, construido por llamada, con
  `as_physics_kwargs()` propio.** — cierra el ítem abierto del 2026-08-09.
  Lo que destraba la sub-pregunta es que **el número de fases es propiedad de
  la implementación del `StateModel`, no del valor**: un flash puede cruzar el
  punto de burbuja a lo largo de `x`, pero un modelo multifásico igual emite
  las dos fases (una con fracción cero). La *clase* del estado es fija por
  modelo, así que el parseo a kwargs es estático y correcto del lado del
  estado, sin contradecir "`loss_func` compone": `loss_func` sigue componiendo
  el extensivo con la fracción de fase; el estado sólo se autodescribe.
  Se mantiene la subdivisión `SinglePhaseState` / `MultiPhaseState`.
- **Regla que ordena tres decisiones sueltas en una sola**: *todo lo que no
  depende de `x` se liga antes del integrador; lo que depende de `x` entra por
  el `bound`.* Es el mismo criterio detrás de `bind`, del hoisting de
  `rate.as_physics_kwargs()` y de las dos clases hermanas.
- **Vocabulario: `gradient_fn` (función de `physics/`) vs. `loss_func` (la
  capa).** Ya está implícito en el filtrado por `signature(gradient_fn)`; se
  fija como convención de documentación porque el uso informal de `loss_func`
  para referirse al gradiente vuelve ilegible el rationale de #18.
- **`as_physics_kwargs()` es un contrato con varios dueños, no un método
  repetido.** Nombre canónico + valor en SI. Hoy lo implementan `Rate` y
  `State`; en v1.5 se suma un tercer productor (atributos de eje calculados).
  El `rhs` es literalmente el merge de esos dicts filtrado por `signature`.

**Corregido en el entendimiento, no en el diseño:**

- El objeto ligado **no se enchufa al integrador**. Lo que entra a `solve_ivp`
  es el `rhs`, que devuelve `dp/dx`. El `bound` se llama *adentro* del `rhs` y
  devuelve un `State`. `gradient_fn` es hermano del `bound`, no su consumidor.
- **La integración en `x` no es vectorizable** y no lo va a ser: el paso `n+1`
  depende del `n`. El eje vectorizable es el de **escenarios** (`y0` como
  array de N presiones iniciales independientes → una evaluación vectorizada
  del `rhs` por paso en vez de N). Es la forma exacta del loop interno del
  fitting de v0.5.

**Abierto:**

- **`across` vs. `pressure` como nombre del argumento del potencial.** Único
  ítem que toca firmas de `physics/` ya implementadas. La vectorización por
  escenarios lo vuelve más urgente de lo que parecía: ya no es una hipótesis
  de v2 acoplado, está en el camino directo a v0.5.

**Próximo paso**: sesión de código — implementar `StateModel`, `BoundState`,
`Fluid`, `FluidState`. Spec cerrada, alcance mecánico.

---

## 2026-08-10 — código: guard de Mach verificado + tests

**Cerrado:**

- **Guard de Mach (`Ek = compressibility * rho_mix * v_mix**2 → 1`).**
  Verificado: `single_phase_gradient` y `beggs_brill_gradient` ya levantan
  `ValueError("Supersonic flow encountered")` en `eh >= 1`, con
  `warnings.warn` en `eh > 0.9` como banda de tolerancia. El guard ya estaba
  implementado en ambos módulos (idéntico patrón); solo faltaba test de B&B.
- Agregados `test_supersonic_raises` y `test_close_to_supersonic_warns` en
  `tests/physics/test_beggs_brill_vs_book.py` (`single_phase` ya tenía
  `test_supersonic_raises`). Suite completa + mypy en verde.

**Abierto:**

- Documentar el límite físico en el README junto con los otros límites
  declarados (scope formal del dominio, DAG del solver, etc.).

**Próximo paso:**

- Sin cambios respecto de la entrada de diseño de abajo.

---

## 2026-08-10 — diseño: canal `@diagnostic`

> Pregunta de la sesión: cerrar el mecanismo de `@diagnostic`, último ítem
> listado como bloqueante de `darcy_weisbach`.

**Cerrado:**

- **`@diagnostic` es post-proceso sobre la solución convergida.** La
  disyuntiva histórica (contextvar vs. colector) estaba mal planteada:
  presuponía captura en vuelo. Replayeando `detailed` sobre el `P(x)` ya
  integrado, en grilla declarada, desaparecen los pasos rechazados de
  `solve_ivp`, desaparece la ambigüedad de reducción punto→eje, y el costo
  durante el fitting es cero exacto. Idea de Marcelo.
- **`GradientResult` no cambia**: sigue `NamedTuple`. Rechazados `extra:
  dict[str, Any]` y `__array__ → total`.
- **Dos niveles**: descomposición (nivel 0, universal, gratis) e intermedios
  de correlación (nivel 1, `detailed_fn` opcional y **explícita**, nunca por
  convención de nombre — no funciona para correlaciones de terceros).
- **`diagnose()` como tercer método del protocolo `LossFunc`**, con el patrón
  ya cerrado de `solve_rate` (`NotImplementedError` por default).
- **Overhead medido**: dict de 8 claves ≈187 ns, llamada extra ≈96 ns,
  correlación escalar proxy ≈1515 ns → ~19% peor caso. Se divide por N al
  vectorizar. No justifica firma polimórfica.
- **`@diagnostic` deja de bloquear `darcy_weisbach` y v0.2.**
- **Momentum explícito, derivación escrita** (`d(ρv²)` con `ρv` constante →
  todo el diferencial sobre `d(1/ρ)`). `total` como suma pasa a ser identidad
  verificable; `β = 0` da cero exacto.

**Abierto:**

- **Firma de `diagnose()` y declaración de grilla** — qué variables se piden y
  en qué puntos del trayecto. Diseño de v0.5.
- Ítems previos sin cambios: forma de `FluidState` bajo fases, nombre del
  `Protocol` neutro, ubicación de `GradientResult`, Jacobiano sparse,
  anidación de métodos numéricos.

**Próximo paso:**

- Sin cambios: sesión de código con la firma por fase de `compressibility` en
  B&B; después, forma de `FluidState` y nombre del `Protocol`.

**Pendiente (higiene):**

- Issue "B&B — firma por fase de `compressibility`" (milestone v0.2), sigue
  sin crear.

---

## 2026-08-09 — código: firma de B&B por fase

> Continuación directa del "Próximo paso" de la sesión de diseño del mismo
> día: aplicar la convención de sufijos de fase (`CLAUDE.md` #19) a
> `beggs_brill_gradient`/`_beggs_brill_detailed`.

**Cerrado:**

- **Renombrado completo a `{propiedad}_{fase}` en la firma de B&B**, en
  `_beggs_brill_detailed` y `beggs_brill_gradient` (`src/fluidnet/physics/multiphase/beggs_brill.py`):
  - `rho_liquid`/`rho_gas` → `density_liquid`/`density_gas`
  - `mu_liquid`/`mu_gas` → `viscosity_liquid`/`viscosity_gas`
  - `gas_compressibility`/`liquid_compressibility` → `compressibility_gas`/
    `compressibility_liquid` (estaban en orden `fase_propiedad`, invertido
    respecto a la convención recién cerrada — corregido de paso, no era
    parte del pedido original pero contradecía #19 directamente).
  - `liquid_mass_rate`/`gas_mass_rate` → `mass_rate_liquid`/`mass_rate_gas`
    (mismo criterio de consistencia: todo kwarg de fase sigue
    `{propiedad}_{fase}`, sin excepción para los rates).
  - Variables internas (`rho_ns`, `rho_mix`, `mu_ns`) **no** se tocaron: no
    son parte del contrato público, son locals de mezcla que ya calcula
    `physics` (coherente con #19 — el `StateModel` entrega por fase, la
    mezcla la computa `physics`).
- Docstrings (`Parameters`) de ambas funciones actualizadas con los nombres
  nuevos; nota agregada en `compressibility_gas`/`compressibility_liquid`
  aclarando que la ponderación por holdup es interna (no se recibe un valor
  de mezcla).
- Tests actualizados con el mismo mapeo de nombres: `test_beggs_brill_vs_fluids.py`,
  `test_beggs_brill_vs_book.py`, `test_multiphase_vector_1.py`.
- `python -m pytest tests/physics/` verde (incluidos los 2 `xfail(strict=True)`
  de vectorización, sin regresiones) y `python -m mypy` limpio tras el cambio.
- Cierra el ítem `physics-single-multiphase.md` §4 ("Firma por fase de
  `compressibility` — pendiente") y el "Próximo paso" de la sesión de diseño
  2026-08-09 de arriba.

**Abierto:**

- **Issue de GitHub "B&B — firma por fase de `compressibility`" (milestone
  v0.2): nunca se creó** (`gh` no disponible en la sesión de diseño previa;
  quedó anotado como pendiente de creación manual). Con el cambio ya
  implementado y verde, no tiene sentido crearla ahora solo para cerrarla —
  si en algún momento se quiere el registro en GitHub, se puede crear
  directamente como cerrada (o simplemente omitir, ya que este log cumple
  la misma función de trazabilidad).
- Sin cambios en los demás ítems abiertos de la sesión de diseño: forma de
  `FluidState` bajo fases, nombre del `Protocol` neutro, resto de
  `ROADMAP §Abiertas`.

**Próximo paso:**

- Cerrar la forma de `FluidState` bajo la convención de fases y el nombre
  concreto del `Protocol` neutro (`StateModel`/`Medium`/`ConstitutiveModel`);
  recién ahí, diseño del caso demo (wellfield sintético).

---

## 2026-08-09 — diseño: contrato de `LossFunc` y `StateModel`

> Pregunta de la sesión: cerrar el contrato de `loss_func` y la relación
> `Rate` ↔ modelo de propiedades. Derivó en la formalización del scope
> matemático del dominio de problemas — la definición que va al README.

**Cerrado:**

- **Contrato de `LossFunc`: dos métodos, no tres modos.**
  `solve_dp(rate, state, **edge_attrs) -> dp` abstracto;
  `solve_rate(dp, state, **edge_attrs) -> rate` en el protocolo integral con
  `NotImplementedError` por default. Las direcciones `Q→P1→P2` y `Q→P2→P1`
  **no son modos distintos**: es el signo del `t_span`, ya cerrado como
  responsabilidad del integrador (2026-08-04). Los ejes ortogonales reales
  son: régimen algebraico/integral (lo declara el `StateModel`) × cuál es la
  incógnita (`dp` o `rate`).
  Rationale de `NotImplementedError` en vez de un default por root-find:
  declara la capacidad en el vocabulario del protocolo y habilita **resolver
  la red sobre campo de potenciales** — no es el régimen de fluidos, pero sí
  el natural de otros dominios. Generalidad barata, sin imponer un Newton
  heredado a toda loss.

- **Scope formal del dominio de problemas.** Potencial en nodos (*across*),
  flujo en ejes (*through*), relación constitutiva **monótona** expresable
  como `Δ = ∫ f(P, x, edge_attrs, node_attrs) dx`. Es un grafo lineal /
  bond graph con elementos disipativos.
  **Dos niveles de garantía, no confundirlos**: Lipschitz da
  existencia/unicidad de la ODE *en el eje*; el content de Millar la da del
  *problema de red*.
  Fuera de scope, declarado como **límite matemático y no de
  implementación**: histéresis, relaciones implícitas no reducibles a ODE,
  elementos activos (rompen la monotonía — por eso v2.0, y son un problema
  más duro que el loop pasivo).
  Consecuencia para el README: el límite del proyecto no es "fluidos" ni
  "DAG", es **"constitutivas monótonas expresables como ODE en la coordenada
  del eje"**. Un límite declarado se lee como criterio; "todavía no lo
  implementé" se lee como inmadurez.

- **`AlgebraicLoss` es el caso degenerado de `IntegralLoss`, no otra física.**
  Si `P` no entra en el integrando, éste es constante y la integral colapsa a
  `grad × L`. Coherente con que el discriminante sea `β == 0`, valor de
  runtime del `StateModel`. Rationale de mantener los dos protocolos: **no
  tiene sentido formular una ODE cuando la solución analítica existe y es
  simple de poner en código.** (Argumento académico, no de performance —
  corrige una formulación previa de esta sesión.)

- **`StateModel`: `Protocol` de nombre neutro; `Fluid` lo implementa.**
  Definición: transforma la variable *across* del nodo (más el estado
  propagado) en los argumentos de la función de gradiente. Fluidos:
  `(comp, T, P) → (ρ, μ, β, σ)`. AC: `(V) → impedancia`.
  **La cadena es anidada `comp → T → P`, no una tupla plana**: fijadas la
  composición y `T`, la parcial que queda es sólo función de `P` — que es lo
  que hace prolija la formulación de la ODE (una sola variable independiente
  en el integrando).
  **`temperature` NO asciende al `Protocol`**: lo contamina (en el demo AC no
  significa nada) y puede complicar la existencia. Queda en la firma de
  `Fluid`, que es donde ya estaba.
  Motivo del nombre neutro: `Fluid` en la firma del protocolo es evidencia
  *en contra* del diferencial physics-agnostic, justo en la fila más débil de
  la tabla de diferenciales. Costo de arreglarlo ahora: un `Protocol`. Costo
  después: romper firma pública.

- **Convención de sufijos de fase. `StateModel` entrega propiedades por fase;
  toda propiedad de mezcla es cómputo de `physics`.**
  `density_gas` / `density_liquid`, etc.; monofásico usa el nombre pelado.
  Razón: la compresibilidad de mezcla se pondera por holdup, y el holdup lo
  calcula B&B — el `StateModel` no puede entregarla porque no conoce el
  holdup. La línea de corte tiene que quedar fijada **antes** de que entre la
  segunda correlación (v0.5, fin del freeze), porque es exactamente donde se
  rompería.

- **Vocabulario canónico, no tabla de renombres.** Los nombres de `physics`
  son contrato. `loss_func` arma `{**rate_kwargs, **state_kwargs,
  **edge_attrs}` y filtra por `signature(gradient_fn)`. Un dict de override
  sólo donde un modelo tenga razón genuina para desviarse: un dict identidad
  es ritual, y una tabla que crece con cada modelo se desincroniza en
  silencio (el kwarg no pasa, entra un default, y revienta lejos del origen).
  Contrato de entrega: **nombre canónico + valor en SI**; `physics` maneja
  los números. Consecuencia buscada: `loss_func` queda como un `solve_ivp`
  wrappeando `physics`, replicable modelo a modelo.

- **`loss_func` compone, no traduce.** La aridad multifásica no es renaming
  sino cómputo: `Rate` aporta el extensivo, `StateModel` la fracción de fase,
  el producto da los rates por fase.

- **Definición de `Rate` por contrato, no por contenido**: *puedo sumarme con
  otros y dar balance cero, y puedo meterme en una loss func.* Eso es todo.
  Es lo que hace que `ComplexRate` entre sin tocar el álgebra.

- **Composición trivial de `ScalarRate`: atributo de subclase, no de clase.**
  ← cierra el ítem que venía abierto desde 2026-08-07 (a).
  `ScalarRate` **no tiene** el campo; no lo tiene vacío. La uniformidad va en
  el método, no en el dato. Un singleton `{"fluid": 1.0}` obligaría a iterar
  dict + división ponderada por nodo en el caso monofásico, que es el loop
  interno del optimizer de v0.5, y no vectoriza.

- **`as_physics_kwargs` se arma una vez por eje, fuera del `solve_ivp`.** Lo
  licencia que en steady-state sin intercambio de masa el mass rate sea
  constante a lo largo del eje. El RHS sólo actualiza `pressure`. Sin esto,
  con 10²–10³ evaluaciones del RHS por eje, el overhead del wrapper deja de
  ser despreciable.

- **Conversión de unidades fuera del contrato de `LossFunc`** — atributo de
  `Network`, capa de I/O. Mantiene la decisión #1 (SI estricto) como real y
  no nominal: es de las cosas que un reviewer chequea leyendo una firma.

**Cambiado:**

- **`as_physics_kwargs()` — ubicación afinada, no movida del todo.** El
  método **existe en `Rate`**: lo extensivo va a la loss function sí o sí.
  Lo que se movió a `loss_func` es la *composición con la fracción de fase*,
  no el acceso al extensivo. Una formulación previa de esta sesión colapsaba
  mal las dos cosas.
- **`MultiphaseRate` (fracciones) sale del roadmap** (`ROADMAP §v1.5` lo
  listaba como extensión de ejemplo). El split líquido/gas depende de un
  flash `(P, T, comp)` y cambia a lo largo del caño: es trabajo de EOS, no
  del rate. Lo que queda es un `StateModel` degenerado de fracción impuesta
  (usuario que declara "asumo 30 % vapor fijo") — hipótesis de modelado
  explícita y válida, pero es un fluido, no un rate.
- **`CLAUDE.md` #6 acotada**: `temperature` en la firma sigue válida para
  `Fluid`; no asciende al `Protocol` neutro.
- **`physics-single-multiphase.md` §1 queda contradicha** por la convención
  de sufijos: documenta la unificación `mix_compressibility →
  compressibility` como deliberada. Resolvía un problema real de su momento,
  pero elige el nombre equivocado bajo la regla nueva. Hay que corregir el
  ADR, no sólo el código.

**Abierto:**

- **Forma de `FluidState` / `StateState` bajo la convención de fases.**
  Consecuencia directa de la decisión de sufijos: `FluidState` **no puede ser
  genéricamente escalar**. Dirección propuesta (no cerrada): dividir en
  subclases `SinglePhaseState` (escalar) y `MultiPhaseState` (vectorial, con
  su dict de nombres `_liquid`/`_gas`), y que el parseo a kwargs lo haga el
  propio estado. Sub-pregunta abierta: si ese parseo es un método del
  `StateModel`/estado o queda del lado de `loss_func` — la decisión de que
  "`loss_func` compone" empuja hacia lo segundo, pero el conocimiento de
  cuántas fases hay vive en el estado. **Toca `CLAUDE.md` #5** (`FluidState`
  como `NamedTuple` de campos escalares requeridos).
- **Nombre concreto del `Protocol` neutro** (`StateModel` / `Medium` /
  `ConstitutiveModel`) — fijarlo antes de que aparezca en firmas públicas.
- **Jacobiano sparse estructural en `mass_balance`**: es la matriz de
  incidencia, conocida de antemano. Pasársela a `scipy.optimize.root` en vez
  de dejar que la estime por diferencias finitas es O(1) bloque vs. O(E)
  evaluaciones por iteración; pesa más en redes chicas. Diferido, no
  discutido en profundidad.
- **Anidación de métodos numéricos.** Resolver por presión da *tres* niveles:
  integral para `dp` vs. `Q`, solver para `Q = f(P)`, solver de balance de
  red. Es un problema matemático, no computacional. Conteo por etapa: v0.2 =
  1 nivel; v1.0 forward + `IntegralLoss` = 2 (no hace falta invertir por
  eje); 3 sólo en `mass_balance` + `IntegralLoss` + BC que exijan inversión.
  **El peor caso no está en el camino a JOSS.** Estrategias, más adelante.
- Los ítems previos sin cambios: mecanismo del canal `@diagnostic`,
  ubicación de `GradientResult`.

**Próximo paso:**

- Sesión de **código**: cambiar la firma de B&B — sacar `compressibility`,
  recibir `compressibility_gas` / `compressibility_liquid`, ponderación por
  holdup interna. Toca `test_beggs_brill_vs_fluids.py` y
  `test_beggs_brill_vs_book.py`, y corregir `physics-single-multiphase.md`
  §1 (que hoy documenta la unificación como decisión vigente). Mecánico,
  pero es lo que fija la convención de fases antes de la segunda correlación.
- Después: cerrar la forma de `FluidState` bajo fases y el nombre del
  `Protocol`; recién ahí, diseño del caso demo.

**Pendiente (higiene, no bloquea código):**

- **Issue de GitHub "B&B — firma por fase de `compressibility`" (milestone
  v0.2), sin crear.** `gh` no está disponible en el entorno de esta sesión;
  queda para creación manual desde la web. El work item en sí ya está
  documentado (ver "Próximo paso" arriba y `physics-single-multiphase.md`
  §4), así que no bloquea la sesión de código.

---

## 2026-08-07 — diseño (b): secuencia de etapas

> Continuación de la sesión (a) del mismo día. Surgió al discutir la tensión
> composición↔propiedades y derivó en una revisión de scope — de las que el
> `README_CLAUDE.md` llama "revisión periódica de alcance".

**Cerrado:**

- **Secuencia de cinco etapas por generalización progresiva**, reemplazando
  las tres anteriores (MVP / v1 / v2):

  | | Contenido | Justificación |
  |---|---|---|
  | v0.2 | DAG, single sink, nodos de paso, un `Fluid` de red | prueba estructural |
  | v0.5 | **fitting** + broadcasting + `@diagnostic` | JOSS — no se mueve |
  | v1.0 | BC en nodos intermedios (2-de-3) + `mass_balance` + `IntegralLoss` | acá aparece Picard |
  | v1.5 | `Fluid` por nodo (composición → propiedades) + temperatura como input | |
  | v2.0 | loops pasivos + elementos activos (bombas) + segundo dominio demo | |

  Criterio: primero el régimen del dominio objetivo y el diferencial, después
  las generalizaciones, al final lo que obliga a abandonar la propagación
  topológica.
- **BC en nodos intermedios van antes que composición variable.** Son lo que
  habilita `mass_balance`, que es donde la dependencia de datos se vuelve
  circular. Con composición primero, al llegar a `mass_balance` habría dos
  fuentes de dificultad mezcladas y no se sabría cuál rompe la convergencia.
- **Nomenclatura unificada a números de versión reales.** Se retiran las
  etiquetas "MVP / v1 / v2" que convivían con los números y generaban
  ambigüedad (el ROADMAP llegaba a decir "v1 — ≈ v0.5").
- **La restricción DAG es del solver, no de la física.** Corrección de una
  hipótesis que estaba implícita: las redes de agua potable y los circuitos
  de edificios son mallados y pasivos, y un loop pasivo con losses monótonas
  sigue siendo convexo con solución única. El argumento correcto de
  posicionamiento es de **dominio** (gathering upstream es DAG y su régimen
  natural es forward propagation), no de generalidad. Documentado en el ADR
  §1 y §2.3 y en `CLAUDE.md` #17, con nota explícita de que el README no
  debe presentar DAG como propiedad del dominio — un reviewer que venga de
  EPANET lo leería como desconocimiento del caso mallado.
- **Vectorización de B&B se queda en v0.5** (decisión de Marcelo). Afecta al
  optimizer y no se considera difícil de resolver. Consecuencia anotada en
  el ROADMAP: en v0.5 la correlación queda vectorizada **antes** de que
  exista un solver que pueda llamarla (multifásico necesita `IntegralLoss`,
  v1.0). No es un olvido de secuencia — es el contrato de capa cero
  funcionando, y se agregó como corolario explícito en la sección Capa 0.
- **Tensión composición↔propiedades documentada** (ver "Abierto"): con
  composición fija el problema hidráulico es convexo (content de Millar) y
  tiene solución única; cuando las propiedades dependen de la composición se
  pierde la función potencial y con ella la garantía de existencia/unicidad.
  **Aparece en DAG, no requiere loops**: en `mass_balance` los caudales son
  incógnita y la dependencia de datos se vuelve circular aunque el grafo sea
  acíclico.

**Abierto:**

- **Acoplamiento composición↔propiedades en `mass_balance` (v1.5).**
  Dirección propuesta: lazo externo de Picard — campo de composición
  congelado *por nodo* (no un fluido global) → solver convexo → re-propagar
  composición → repetir. Converge bajo baja sensibilidad de propiedades a la
  mezcla, que es el régimen del caso de uso objetivo. La no convergencia se
  reporta como resultado con significado físico, nunca como excepción
  silenciosa. El `Fluid` stateless es lo que hace el lazo barato (no hay
  caché que invalidar). **Abierto**: criterio de corte; y si la inversión de
  sentido de flujo en loops (v2.0) requiere regularización o basta con
  detectarla y abortar.
- Los ítems de la sesión (a) siguen abiertos sin cambios: composición trivial
  de `ScalarRate`, `as_physics_kwargs()`, ubicación de `FluidState`,
  mecanismo de `@diagnostic`.
- **¿Adelantar el segundo dominio demo de v2.0 a v0.5?** Surgió al armar la
  tabla de diferenciales: physics-agnostic queda declarado en v0.2 pero sin
  demostrar hasta v2.0, o sea sin evidencia en el momento del submit a JOSS.
  Un segundo dominio (térmico / eléctrico AC) es topológicamente idéntico y
  el costo es el notebook, no el core. Contra: agranda v0.5, que es la etapa
  que no conviene dilatar. A evaluar al diseñar el caso demo.

**Higiene documental (misma sesión):**

- **`VISION.md` creado** — objetivo personal del proyecto, qué es lo escaso
  (criterio de dominio, no código), diferenciales rankeados por
  defendibilidad, anclaje teórico para el paper, criterio de decisión de
  scope. **No va al repo público**: entrada en `.gitignore`, vive en el
  Project y localmente. Existe porque el nivel "para qué" no estaba escrito
  en ningún lado y se estaba sosteniendo de memoria entre sesiones.
- **ROADMAP: bloque "A dónde vamos" al tope**, con la **tabla
  diferencial↔versión** — el eslabón que faltaba entre los diferenciales
  (que vivían en el ADR §1) y las etapas (que vivían en el ROADMAP). Sin esa
  tabla, la pregunta "¿esto entra en v0.5?" se respondía por intuición.
- **`README_CLAUDE.md` actualizado**: `VISION.md` en la jerarquía de
  documentos, sección de lectura top-down numerada, y explicitada la
  distinción entre `ROADMAP §Abiertas` (backlog de diseño, persiste) y
  `session-log §Abierto` (estado de la sesión, migra o se cierra).

**Próximo paso:**

- Sin cambios respecto de la sesión (a): cerrar la composición trivial de
  `ScalarRate` y `as_physics_kwargs()`, después diseño del caso demo.
  Verificar de paso si `physics-single-multiphase.md` necesita ajuste — §4
  apunta a "roadmap v0.5" para vectorización, que sigue siendo correcto, y
  §5 no nombra versión, así que probablemente no haya nada que tocar.

---

## 2026-08-07 — diseño (a): `Rate` / `Fluid`

> Pregunta de la sesión: **cerrar la firma de `Rate`/`ScalarRate`**
> (`ROADMAP.md` → Secuencia inmediata #1), incluyendo quién es dueño del
> modelo de propiedades del fluido.

**Cerrado:**

- **`Rate` = magnitud extensiva + composición intensiva. Sin propiedades de
  fluido.** Es una **corrección** de la decisión #3 previa de `CLAUDE.md` y
  del ADR §2.1, que ponían densidad y viscosidad adentro del `Rate`. No es
  una decisión nueva conviviendo con la vieja: la reemplaza. Rationale:
  - El álgebra queda bien definida. Con `density` adentro, `rate * 2` es
    ambiguo (el caudal escala, la densidad no) y hay que escribir una regla
    ad-hoc extensivo/intensivo *dentro* del `__mul__`.
  - `Rate` se mantiene homogéneo numéricamente (float / array / complejo,
    optimizable en todos los casos). Habilita `ComplexRate` para el demo de
    red AC sin tocar el álgebra.
  - `Rate` es el **input** de la correlación (o el output si hay root
    solving); el fluido es lo que entra en la fórmula de gradiente. Roles
    distintos → tipos distintos.
  - El diferencial de mezcla de salmueras **no se pierde**: lo que se
    propaga es composición (lineal bajo mezcla, invariante bajo scaling),
    no propiedades. Las propiedades se derivan, nunca se almacenan — lo que
    además satisface el requisito de que el estado del fluido sea función de
    `P` y no una constante congelada.
- **`Fluid` = fábrica stateless de `FluidState`, capa cero hermana de
  `physics/`.** Lo que se declara en la red no es un fluido con densidad
  1000, es el **modelo** que mapea `(composición, P, T) → (ρ, μ, β, σ)`.
  Recibe la composición como **dato crudo**, nunca un objeto `Rate` (si
  recibiera el `Rate` dejaría de ser capa cero).
  - Cadena canónica: `Rate → composición → Fluid(P, T) → FluidState →
    physics.gradient → loss_func integra sobre L`.
  - `physics/` nunca ve `T` ni composición: recibe propiedades ya evaluadas.
- **`FluidState` = `NamedTuple(density, viscosity, compressibility, sigma)`,
  todos los campos requeridos.** Nada de `float | None`: `β = 0` es el valor
  físico exacto del agua (ya cerrado como "estado, no flag"), y un
  `sigma=None` filtrándose a B&B es el mismo problema que un default físico
  invisible — además de romper el filtrado de kwargs por `signature`.
- **`temperature` en la firma desde v0.2, default `None` = "no
  suministrado", jamás un valor implícito.** Fluido incompresible la ignora;
  gas isotérmico la fija **en construcción** (`IsothermalGas(T=323.15)`), no
  en el call site; fluido que la necesita y recibe `None` → `ValueError`.
  Consecuencia: v2 (perfiles de `T` prescritos) es aditivo — el solver pasa
  `T` donde hoy pasa `None`, sin cambio de protocolo.
- **El discriminador `AlgebraicLoss`/`IntegralLoss` lo declara el `Fluid`.**
  ← cierra el ítem que `ROADMAP.md` y el ADR §2.2 listaban como abierto. No
  es el usuario al registrar la loss ni un chequeo del solver: el `Fluid` es
  dueño del EOS, así que es el único que sabe si `∂ρ/∂P = 0`. Un fluido
  incompresible expone propiedades sin necesitar `P` → `AlgebraicLoss`; uno
  compresible exige `P` → solo `IntegralLoss`. La loss hereda el régimen del
  fluido que recibe. Esto resuelve además la tensión de que `AlgebraicLoss`
  no tiene `P` en la firma (es su definición) y por lo tanto no podría
  llamar a un `Fluid` que siempre la exigiera.
- **Dueño del `Fluid`: red en v1, nodo en v2, mismo mecanismo**
  (`G.nodes[n].get('fluid', network.fluid)`). v1 es el caso degenerado de
  v2, no un mecanismo distinto a reemplazar. En v2 el fluido de cada nodo se
  precalcula en la misma pasada topológica que ya hace `propagate_rates`.
  **Un edge toma el fluido de su nodo `upstream`**: la mezcla ocurre *en* el
  nodo receptor, nunca dentro del caño. En v1 es identidad, pero la
  semántica queda escrita.
- **Módulo `fluidnet/fluid.py` (singular), no `fluids.py`** — colisión
  visual con `fluids` (ChEDL), oráculo de cross-validation importado en tests.
- **Statement of need reforzado**: frente a pandapipes la distinción no es
  "un fluido por red" vs. "muchos", es que fluidnet declara un **modelo** de
  fluido y no propiedades congeladas; la composición local sale de la
  propagación. ADR §1 actualizado.
- Docs sincronizadas: `CLAUDE.md` (decisiones #3–#8 reescritas/agregadas,
  diagrama de capas con `Fluid`, cadena de evaluación), `ROADMAP.md` (Capa
  0 bis, migración Abiertas→Cerradas, scope v0.2/v2),
  `architecture-v0.2.md` (§1, §2.1, §2.1bis nuevo, §2.2, §2.3, §2.5, §3, §4,
  §5, §6).

**Abierto:**

- **Forma de la composición trivial en `ScalarRate`** — ¿vacía, o un
  singleton tipo `{"fluid": 1.0}` para que el álgebra de mezcla sea uniforme
  entre `ScalarRate` y `BrineRate`? Bloquea la firma final de `__add__`.
- **`as_physics_kwargs()`** — dónde vive el adaptador que traduce
  `(Rate, FluidState, edge_attrs)` a los kwargs exactos de cada función de
  gradiente. Candidato natural: la `loss_func`, única capa que ve las tres
  cosas a la vez. Pendiente confirmar contra el filtrado por `signature`.
- Dónde vive `FluidState` (mismo debate que `GradientResult`: módulo propio
  de tipos vs. junto a su productor).
- Mecanismo del canal `@diagnostic` (contextvar vs. colector) — sin cambios.
- Higiene documental: el ADR conserva una sección "Abiertas" que
  `README_CLAUDE.md` dice que no debería tener (el backlog vive en
  `ROADMAP.md`). Anotado en el propio ADR; candidato a limpieza futura.

**Próximo paso:**

- Sesión de **diseño**: cerrar la composición trivial de `ScalarRate` y
  `as_physics_kwargs()` — es lo último entre el diseño y la implementación
  de la capa `Rate`/`Fluid`. Después, diseño del caso demo (wellfield
  sintético), que tensiona las firmas contra un uso real.

---

## 2026-08-07 — código

**Cerrado:**

- Conversión de `sigma` a SI (N/m) en `multiphase/beggs_brill.py` y golden
  tests, cerrando el ítem pendiente de la sesión de diseño 2026-08-04. Al
  verificar el cambio (hecho por Marcelo) se encontraron dos puntos
  desincronizados y se corrigieron en la misma sesión:
  - `beggs_brill_gradient` (wrapper público) seguía con
    `sigma: float = 30.0` (dyn/cm) y docstring sin actualizar — solo
    `_beggs_brill_detailed` se había convertido.
  - `tests/physics/test_multiphase_vector_1.py::DETAILED_ARGS` seguía con
    `sigma=28.0`; el comentario del propio archivo la marca como "same case
    as `test_checalc_case_no_payne`", que ya usaba `28.0e-3`.
- **Precondición de signo en `_beggs_brill_detailed`/`beggs_brill_gradient`**
  (segunda mitad del "próximo paso" del 2026-08-04, ya cerrada como decisión
  ahí): reemplazada la rama de flujo reverso (`abs()` + inclinación
  invertida + flip de signo del gradiente al final) por
  `if liquid_mass_rate < 0 or gas_mass_rate < 0: raise ValueError(...)`.
  Dirección de flujo queda resuelta por el integrador, no por `physics`,
  como corolario ya cerrado. Docstrings de ambas funciones actualizados.
- `python -m pytest tests/physics/` y `python -m mypy` verdes tras los
  cambios.
- **`tests/physics/test_beggs_brill_vs_fluids.py` creado (Marcelo)**,
  implementando por fin la estrategia de cross-validation contra `fluids`
  documentada en el ADR §3.2 pero nunca comiteada (confirmado por `git log`
  sin resultados para ningún archivo `*vs_fluids*`, ver "Abierto" de la
  sesión 2026-07-31). Verificarlo destapó un **bug real, no un problema del
  test**:
  - **`_holdup` interpolaba mal el régimen `transition`**: usaba
    intermittent+distributed (`_holdup(1, ...)` + `_holdup(2, ...)`) en vez
    de segregated+intermittent. Confirmado incorrecto contra el propio
    código fuente de `fluids` (`Beggs_Brill`/`_Beggs_Brill_holdup`,
    instalado localmente): interpola siempre segregated+intermittent, igual
    que la fórmula publicada. Este es exactamente el bug que el ADR §4 ya
    daba por "corregido" — la corrección nunca había llegado al código.
    Fix de una línea (`beggs_brill.py`): `_holdup(0, ...)` +
    `_holdup(1, ...)`. Confirmado por `test_against_fluids_live`
    (comparación en vivo contra `fluids`, no valores pinneados) pasando
    para los 2 casos `transition_*` tras el fix; sin regresiones en
    `test_beggs_brill_vs_book.py` / `test_multiphase_vector_1.py` (14 tests).
  - **Precisión de los valores pinneados en `GOLDEN`**: `liquid_holdup` y
    `mixture_density` estaban copiados a 6 cifras significativas con
    `rtol=1e-6` (convención de este repo para holdup, ver "Convenciones de
    testing" en `CLAUDE.md`) — demasiado ajustado para el redondeo manual,
    fallaban 6 de 8 casos por error de ~1e-6–2e-6 aun siendo correctos.
    Regenerados con precisión completa llamando directamente a las
    funciones internas de `fluids` (mismo cálculo que hace
    `fluids.two_phase.Beggs_Brill` puertas adentro, replicado para poder
    extraer `Hl`/`rhos` en vez de solo el `dP` total que expone la API
    pública). `rtol=1e-6` se mantuvo sin tocar — no era un problema de
    tolerancia sino de cuántos dígitos se habían pegado a mano.
  - 17/17 tests pasando en el archivo tras ambos fixes.
- ADR (`physics-single-multiphase.md` §3.2 y la mención en §1) actualizada:
  el nombre real es `test_beggs_brill_vs_fluids.py`, no
  `test_multiphase_vs_fluids.py` como decía el documento.

**Cerrado (cont.):**

- **Unificación `compressibility` vs. `mix_compressibility` — resuelta,
  solo faltaba documentarlo.** Al revisar el pedido de unificar nombres se
  encontró que el código ya no tenía el problema: el commit `7142191`
  (2026-08-04, refactor kw-only) ya había renombrado
  `mix_compressibility` → `compressibility` en `_beggs_brill_detailed` y
  `beggs_brill_gradient`, igual que en `single_phase_gradient`. `grep` en
  `src/` y `tests/` confirma cero ocurrencias de `mix_compressibility`. Lo
  que quedaba desincronizado era la documentación: ADR §1
  (`physics-single-multiphase.md`) seguía describiendo el nombre viejo como
  "hoy vigente" y la unificación como abierta — corregido para reflejar que
  ya está cerrada.

**Abierto:** (ver también 2026-08-04)

- Función de estado (agrupar densidad/viscosidad/sigma) antes de `physics`.
  → **cerrado en la sesión de diseño del mismo día**: es `Fluid` →
  `FluidState`.
- `Rate.as_physics_kwargs()` como adaptador. → sigue abierto tras la sesión
  de diseño, pero acotado: el candidato es la `loss_func`, no el `Rate`.
- ADR §3.2 sigue describiendo `test_holdup_within_physical_bounds` /
  `test_against_fluids_live` / `test_golden_vs_fluids_pinned` como diseño;
  ahora coinciden con el archivo real — si en el futuro se los toca,
  actualizar ambos lados juntos. (Revisado 2026-08-07: siguen coincidiendo,
  nada que corregir.)

**Próximo paso:**

- Sesión de **diseño**: cerrar la firma de `Rate`/`ScalarRate`
  (`ROADMAP.md` → Secuencia inmediata #1) — bloqueante del MVP. Ya no
  incluye resolver la unidad de `sigma` (cerrado hoy); sigue pendiente
  definir quién es dueño del modelo de propiedades (densidad/viscosidad) y
  cómo encajan la unificación de `compressibility` y `as_physics_kwargs()`
  listadas en "Abierto". → **hecho**, ver la entrada de diseño de arriba.

---

## 2026-08-04 — diseño

**Cerrado:**

- **Firmas de `physics/`: keyword-only completo.** Toda función de
  gradiente (`single_phase_gradient`, `beggs_brill_gradient`,
  `_beggs_brill_detailed`) enteramente kw-only, sin excepción por cantidad
  de rates. Razón: `loss_func` despacha genéricamente
  (`gradient_fn(**kwargs)` filtrado por `signature`); una firma que varía
  de forma entre modelos obliga a ramificar en el consumidor. Añadida como
  decisión cerrada de `CLAUDE.md`.
- **Política de defaults en `physics/`.** Defaults solo en flags de modelo
  (`payne_correction`, `holdup_adj`); nunca en estado físico
  (`roughness`, `inclination`, `sigma`, densidades, viscosidades,
  `compressibility`) — un default físico es una hipótesis de modelado
  invisible.
- **`sigma` a SI (N/m).** Cierra la deuda documentada en
  `physics-single-multiphase.md` §4; queda pendiente sacar la conversión
  `sigma * 1e-3` de los tests de cross-validation (ver "Abierto").
- **Precondición de signo en `beggs_brill_gradient`/`_beggs_brill_detailed`.**
  `liquid_mass_rate >= 0` y `gas_mass_rate >= 0`, `ValueError` fuera de
  rango — reemplaza la rama de "flujo reverso" actual (`abs()` + inclinación
  invertida).
- **Dirección de flujo resuelta por el integrador, no por `physics`.**
  Caudal negativo no es un caso a soportar en `physics` ni en `loss_func`;
  el sentido de integración se invierte en el solver (`solve_ivp` con
  `t_span=(L, 0)` en vez de `(0, L)`).
- Docs sincronizadas con estas decisiones: `CLAUDE.md`,
  `physics-single-multiphase.md` (§2, §4), `ROADMAP.md` (Decisiones
  cerradas), `architecture-v0.2.md` (§2.2).
- **`compressibility` es estado, no flag.** β = (1/ρ)(∂ρ/∂P)_T en 1/Pa,
  entra en el término de momento vía
  `dP/dx = (grav + fric) / (1 − ρv²β)`; `β = 0` es el valor físico exacto
  de un líquido incompresible, no una aproximación.
- **`physics` no conoce el régimen algebraico/integral** — corolario
  explícito del contrato de capa cero: tampoco sabe si su resultado se va
  a integrar o a multiplicar por `L`; esa decisión es de `loss_func`.
- **Higiene**: `ROADMAP.md` (scope MVP) ya no dice que `darcy_weisbach`
  "fija `compressibility=0`" como si fuera una decisión de configuración.

**Abierto:**

- Implementación: el código de `_beggs_brill_detailed` todavía tiene la
  rama de flujo reverso; falta reemplazarla por la precondición `ValueError`.
- Sacar la conversión `sigma * 1e-3` de los tests de cross-validation.
- Unificar nombre `compressibility` vs. `mix_compressibility`.
- Definir una función de estado (qué agrupa densidad/viscosidad/sigma antes
  de llegar a `physics`, probablemente parte del diseño de `Rate`).
- `Rate.as_physics_kwargs()` (o nombre similar) como adaptador.

**Próximo paso:**

- Sesión de código: implementar la precondición de signo en
  `_beggs_brill_detailed` y sacar la conversión `sigma * 1e-3` de los tests
  de cross-validation.

---

## 2026-07-31 — código

**Cerrado:**
- Docstrings de toda la capa `physics/` (`dimensionless.py`, `friction.py`,
  `single_phase.py`, `multiphase/beggs_brill.py`, `types.py`) pasadas a
  formato numpydoc, con los tipos de cada parámetro alineados a los type
  hints reales de la firma (`ArrayLike` donde correspondía, no prosa tipo
  "float or array_like").
- Bug de tipado corregido en `friction_factor`: `fanning` estaba anotado
  `ArrayLike` pero se usa como flag booleano puro — cambiado a `bool`
  (decisión confirmada por Marcelo, no housekeeping automático).
- `# type: ignore` obsoleto eliminado en `multiphase/beggs_brill.py` (import
  de `scipy.constants`, ya cubierto por el override de `pyproject.toml`).
- Batería nueva `tests/physics/test_multiphase_vector_1.py`: contrato de
  forma (array vs. escalar) de `beggs_brill_flowmap`, `_holdup`,
  `_beggs_brill_detailed` y `beggs_brill_gradient`, recorridas en el orden
  de definición del módulo. Confirma que `beggs_brill_flowmap` y `_holdup`
  ya vectorizan (esto último no estaba documentado); fija el contrato
  escalar de `_beggs_brill_detailed`/`beggs_brill_gradient` y agrega 2
  `xfail(strict=True)` como spec ejecutable de la vectorización v0.5 (a
  nivel interno y público).
- `docs/design/physics-single-multiphase.md` §3.3 actualizada: apunta al
  archivo de tests nuevo (antes decía que vivían en
  `test_multiphase_golden.py`) y documenta la cobertura de `_holdup`.
- `CLAUDE.md` actualizado con la nota de dónde viven los tests
  vectoriales; el TODO de sincronización de docs que se había agregado
  quedó resuelto en la misma sesión (el ADR ya estaba al día).

**Abierto:**
- Vectorización real de `_beggs_brill_detailed`/`beggs_brill_gradient`
  sigue sin implementar — los 2 `xfail` son la spec, no el fix.
- `test_multiphase_vs_fluids.py`, mencionado en el ADR §3.2 como parte de
  la estrategia de testing, todavía no existe en el repo.

**Próximo paso:**
- Arrancar con la integral de gradiente (relacionado al protocolo
  `IntegralLoss`, declarado en v0.2 pero sin implementar) — definir alcance
  concreto al empezar la sesión.

---

## 2026-07-29 — diseño

**Cerrado:**
- Definido el framework de trabajo: sesiones de diseño (viernes, acá/chat)
  vs. sesiones de código (sábado, Claude Code en VS Code).
- Definidos los archivos de contexto: `CLAUDE.md` (raíz, para Claude Code)
  y este `session-log.md` (docs/, bitácora de estado entre sesiones).
- `CLAUDE.md` inicial armado con arquitectura de 4 capas, decisiones
  cerradas (SI estricto, Rate completo en loss_func, sin Node/Edge,
  @diagnostic, keyword-only args, versionado) y convenciones de testing.

**Abierto:**
- Confirmar destino del MVP de 26 tests (¿local en alguna máquina, o se
  reconstruye desde Draft 1 + ROADMAP como spec?) — decisión abierta
  heredada de Draft 0/1, sigue sin resolver.
- Mecanismo del canal `@diagnostic` (contextvar vs. objeto colector) — no
  bloquea arquitectura, pero hay que cerrarlo antes de implementarlo.
- Firmas públicas de `Rate`/`ScalarRate` — diseñadas conceptualmente en
  Draft 1, faltan a nivel de interfaz (docstrings + type hints).

**Próximo paso:**
- Primera sesión de código: confirmar si el MVP de 26 tests aparece en
  alguna carpeta local antes de empezar a reconstruir nada desde cero.

---

## 2026-08-14 — diseño

**Cerrado:**

- Housekeeping de `ROADMAP.md`/`CLAUDE.md` aplicado: referencias a
  `Rate`/`BaseRate` (#35–#37) y `BoundStateModel` sincronizadas, rename
  `state/fluids/single_phase.py` reflejado en todos los bullets que
  correspondía, ítem 1 de §Secuencia inmediata marcado **Implementado**.
- Infraestructura de repositorio (§v1.0) adelantada a v0.2: el scope de
  mypy sobre `tests/` necesita un ambiente limpio con registro, y la
  sesión de infra es autocontenida (no depende de ninguna decisión de
  diseño abierta).
- Tres entradas nuevas en `ROADMAP §Abiertas`: primera subclase concreta
  de `VectorRateBase`, scope de mypy sobre `tests/` (con dirección
  propuesta: `strict = true` global + override para `tests.*`), y rate
  variable en `x` para black-oil (mecanismo propuesto vía #21: `Rate`
  aporta el extensivo total, `StateModel` emite la fracción de fase).
- Fragmento de prompt de una sesión ya cerrada, pegado por error dentro
  de `CLAUDE.md` (sección "## 10. ROADMAP.md"), eliminado.

**Abierto** → `ROADMAP §Abiertas`: primera subclase concreta de
`VectorRateBase`; scope de mypy sobre `tests/`; rate variable en `x`
(black-oil).

**Próximo paso:**

- Sesión de código de infraestructura: `.github/workflows/checks.yml`
  (matriz 3.10/3.12), `CONTRIBUTING.md`, `CITATION.cff` + Zenodo.
- Después: sesión de diseño de `Result` + firmas de `LossFunc` + caso
  demo.
