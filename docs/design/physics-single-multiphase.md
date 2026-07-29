# Physics layer: `single_phase` y `multiphase` — comportamiento y testing

> Documento de referencia rápida para la capa `physics/`. Complementa
> `docs/design/architecture-v0.2.md` (ADR); acá el foco es qué hace cada
> módulo hoy y cómo se valida, no las decisiones de arquitectura.

---

## 1. `single_phase.py`

**Rol**: contratos base compartidos por toda la capa physics.

- `GradientResult`: `NamedTuple(total, gravity, friction, momentum)` — el
  tipo de retorno de cualquier función de gradiente (mono o multifásico).
  Reemplaza la tupla pelada del prototipo 2018; acceso por nombre
  (`result.friction`) en vez de por índice.
- Funciones de gradiente monofásico (Darcy-Weisbach) construidas sobre este
  mismo contrato — comparten `GradientResult` con `multiphase`, lo que
  permite que un solver trate ambos casos de forma uniforme aguas abajo.

**Testing**: golden tests contra casos de libro/referencia (mismo patrón que
`multiphase`, ver §3). 11 tests pasando entre los cuatro módulos de physics
(`dimensionless`, `friction`, `single_phase`, `multiphase`) al momento de
escribir esto — la cuenta crece con lo agregado en `test_multiphase_golden.py`
y `test_multiphase_vs_fluids.py`.

---

## 2. `multiphase.py`

**Rol**: correlación de Beggs & Brill (1973) para flujo bifásico
líquido-gas, cualquier inclinación. Único modelo multifásico implementado
hoy — ver §4 para lo que falta.

**API pública**: `beggs_brill_gradient(...) -> GradientResult`. Firma en SI
estricto (kg/s, kg/m³, Pa·s, m), con `sigma` como única excepción histórica
(dyn/cm, no N/m — a revisar si se corrige antes de v0.2 o se documenta como
deuda).

**API interna**: `_beggs_brill_detailed(...) -> dict` — mismo cálculo más
intermedios (`NFr`, `Cl`, `liquid_holdup`, `ReNs`, `f`, `fNs`,
`mixture_density`, `flow_regime`). Es lo que consumen los tests y,
eventualmente, el canal `@diagnostic`. No es parte del contrato público —
ningún solver debería importar `_beggs_brill_detailed` directamente.

**Contrato de forma (hoy)**: **escalar únicamente**. `int(flowmap(...))`
fuerza escalar en `_beggs_brill_detailed`, igual que los `if` de `_holdup` y
del multiplicador de fricción. `beggs_brill_flowmap` en cambio **sí** es
vectorial (usa máscaras booleanas, no `if`). Ver §4.4 y el test
`test_detailed_scalar_contract_today`.

**Convención de signos**: `dp = p_downstream − p_upstream` (pérdida →
negativo), consistente con el resto del paquete.

---

## 3. Estrategia de testing de `multiphase`

Tres archivos, cada uno con un rol distinto — no son redundantes:

### 3.1 `test_multiphase_golden.py` — validación contra literatura

- **Kermit Brown, ejemplo 4.7**: caso de libro con NFr, Re, f (Darcy,
  book/4 para Fanning), holdup y gradientes gravitacional/friccional
  publicados. Tolerancias flojas (5–10%) porque el libro redondea
  intermedios y lee `f` de un diagrama de Moody.
- **checalc.com**: caso sanity sin corrección de Payne — no valida números
  exactos, valida invariantes físicos (`gradient.total < 0` en upflow,
  `Cl < Hl <= 1`).
- Logging INFO antes de cada assert: valor calculado, valor esperado, error
  relativo — para debuguear sin tener que parchear el test.

### 3.2 `test_multiphase_vs_fluids.py` — cross-validation con `fluids` (ChEDL)

- 8 casos (`GOLDEN`) cubriendo los 3 regímenes puros + transition, en
  horizontal / uphill / vertical / downhill.
- **Capa pinneada** (`test_golden_vs_fluids_pinned`): valores hardcodeados,
  generados una vez corriendo `fluids.two_phase.Beggs_Brill` v1.3.1. Corre
  sin instalar `fluids`. Holdup a rtol 1e-6 (independiente de fricción — si
  falla, el bug está en flowmap/holdup); gradiente total a rtol 1.5% (deja
  margen a la diferencia Chen vs. Colebrook en el factor de fricción).
- **Capa live** (`test_against_fluids_live`): mismos casos llamando a
  `fluids` en tiempo real; `pytest.importorskip` la saltea si no está
  instalado. Detecta silenciosamente si una versión futura de `fluids`
  corrige algo y los valores pinneados quedaron desactualizados.
- **Guard de bounds** (`test_holdup_within_physical_bounds`): un caso
  downhill donde `fluids` da holdup negativo (no clippea); confirma que
  fluidnet sí lo mantiene en [0, 1].
- `fluids` vive en `[project.optional-dependencies].dev` — oráculo de
  validación, nunca dependencia de runtime.

### 3.3 Comportamiento de arrays (dentro de `test_multiphase_golden.py`)

- `test_flowmap_vectorized`: `beggs_brill_flowmap` con arrays de 5 puntos,
  regímenes verificados uno por uno contra los boundaries L1–L4.
- `test_flowmap_rejects_out_of_domain`: NaN debe levantar `ValueError`
  (falsea todas las máscaras); `Cl=0` **no** sirve como probe porque cae en
  la máscara distributed.
- `test_detailed_vectorized_over_rates`: **`xfail(strict=True)`** — spec
  ejecutable del comportamiento deseado en v0.5 (rates vectorizados →
  gradientes vectorizados). Falla hoy a propósito; el día que se
  implemente el broadcasting, este test **pasa** y el `strict=True` pone la
  suite en rojo como recordatorio de sacar el marker y pinnear valores
  reales.
- `test_detailed_scalar_contract_today`: fija el contrato actual (escalar
  entra, floats salen) para no dejar que un array 0-d se filtre al
  `GradientResult` en un estado semi-vectorizado.

---

## 4. Deuda / desviaciones documentadas (no bugs)

- **Bug corregido**: `_holdup` interpolaba mal el régimen `transition`
  (intermittent+distributed en vez de segregated+intermittent). Fix de una
  línea, validado contra `fluids` en las 8 celdas de transition.
- **Guard agregado**: `EL(0) >= Cl` (restricción original del paper,
  presente en `fluids`, faltaba en fluidnet). Sin el guard, `fluidnet`
  divergía de `fluids` en casos de alto `Cl`/bajo `Fr`.
- **Clip de holdup a [0, 1]**: decisión propia, más correcta que `fluids`
  (que propaga holdups fuera de rango físico en downhill empinado /
  Cl alto). Documentado como desviación intencional, no discrepancia.
- **Payne correction**: solo en fluidnet; `fluids` no la tiene — todos los
  tests cross-validation la desactivan (`payne_correction=False`).
- **Unidad de `sigma`**: dyn/cm en la firma de fluidnet vs. N/m en
  `fluids` — conversión explícita en los tests (`sigma * 1e-3`). Candidato
  a unificar a SI (N/m) antes de cerrar la firma pública de `Rate`.
- **Vectorización**: no implementada (`_beggs_brill_detailed` es escalar);
  `beggs_brill_flowmap` sí. Ver roadmap v0.5 (broadcasting `pd.Series`).

---

## 5. Roadmap — próximos modelos multifásicos

Beggs & Brill es hoy el único modelo. Antes de agregar modelos más
sofisticados (drift-flux, etc. — ya descartados como punto de partida en
`draft_0.MD`), se suman los dos modelos básicos de holdup como referencia y
piso de comparación:

- **No-slip (homogéneo)**: `Hl = Cl` — sin corrección de deslizamiento
  entre fases. Es el caso trivial contra el que `Beggs_Brill` (y cualquier
  correlación futura) debería converger cuando `y = Cl/Hl² → 1` (fricción
  sin multiplicador bifásico) y sirve como baseline de sanity check /
  límite inferior de complejidad, análogo a `constant_friction` en la capa
  de `loss_func` algebraica.
- **Constant slip**: velocidad de deslizamiento fase líquida/gas fija
  (parámetro), `Hl` derivado de esa velocidad de slip constante en vez de
  correlacionarla con `NFr`/`Cl` como hace B&B. Puente conceptual entre
  no-slip y los modelos empíricos completos — útil como modelo pedagógico
  en el notebook demo y como test de regresión barato (menos parámetros
  que B&B, holdup en forma cerrada).

Ambos entran como **protocolo `AlgebraicLoss`** (dp no depende de presión
absoluta) — a diferencia de B&B multifásico que hoy vive conceptualmente
como candidato a `IntegralLoss` una vez que se acople a `p_boundary`. Igual
criterio de testing que B&B: golden contra caso analítico/libro + —si
`fluids` u otra librería los implementa— cross-validation opcional vía
`importorskip`.
