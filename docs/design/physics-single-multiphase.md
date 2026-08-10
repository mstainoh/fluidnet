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

**`compressibility` es estado, no flag.** Es β = (1/ρ)(∂ρ/∂P)_T en 1/Pa, una
propiedad física del fluido — no un switch que activa o desactiva un modo de
cálculo. Entra en el término de aceleración (momentum) vía
`dP/dx = (grav + fric) / (1 − ρv²β)`: `β = 0` es el valor físico exacto de un
líquido incompresible, no una aproximación ni un caso especial. (Ver también
la nota sobre régimen algebraico/integral en `architecture-v0.2.md` §2.2:
`compressibility == 0` es el discriminador de runtime entre los dos
protocolos de loss.)

> ⚠️ **Revertido para multifásico (2026-08-09).** El commit `7142191`
> (2026-08-04, refactor kw-only) había unificado
> `mix_compressibility → compressibility` en `beggs_brill_gradient` para que
> el adaptador emitiera una sola clave sin ramificar por modelo. La sesión de
> diseño del 2026-08-09 cerró la regla contraria: **el `StateModel` entrega
> propiedades por fase y toda propiedad de mezcla es cómputo de `physics`.**
> La compresibilidad de mezcla se pondera por holdup, y el holdup lo calcula
> B&B — el `StateModel` no tiene cómo entregarla.
>
> Firma resultante: `compressibility_gas` / `compressibility_liquid`, con la
> ponderación interna a la correlación. El monofásico conserva el nombre
> pelado `compressibility`. Ver `CLAUDE.md` #19.
> **Estado: pendiente de implementación** (próxima sesión de código).

**El término de momentum es explícito, no un residuo.** Sale de
`d(ρv²) = d((1/ρ)·(ρv)²)`, donde `ρv` es constante por conservación de masa,
de modo que el diferencial recae íntegro sobre `d(1/ρ)`, que es función de la
compresibilidad y de `dP`. La forma cerrada
`dP/dx = (grav + fric)/(1 − ρv²β)` es el resultado de despejar `dP` de esa
identidad, no una decisión de diseño ni una aproximación.

Dos consecuencias que conviene tener escritas:

- `total = gravity + friction + momentum` es una **identidad algebraica
  verificable**, no una definición de conveniencia. Test barato y que vale la
  pena: comparar `result.total` contra la forma cerrada a tolerancia de punto
  flotante.
- Con `β = 0` el término da **cero exacto**, no despreciable. Coherente con
  que `β = 0` sea el valor físico del líquido incompresible y no un flag.
- Hay singularidad genuina en `ρv²β → 1` (flujo bloqueado, Mach unitario). Es
  un límite físico, no numérico. Ver `ROADMAP §Abiertas` (guard pendiente de
  verificar).

**Testing**: golden tests contra casos de libro/referencia (mismo patrón que
`multiphase`, ver §3). 11 tests pasando entre los cuatro módulos de physics
(`dimensionless`, `friction`, `single_phase`, `multiphase`) al momento de
escribir esto — la cuenta crece con lo agregado en `test_beggs_brill_vs_book.py`
y `test_beggs_brill_vs_fluids.py`.

---

## 2. `multiphase.py`

**Rol**: correlación de Beggs & Brill (1973) para flujo bifásico
líquido-gas, cualquier inclinación. Único modelo multifásico implementado
hoy — ver §4 para lo que falta.

**API pública**: `beggs_brill_gradient(*, ...) -> GradientResult`. Firma en
SI estricto (kg/s, kg/m³, Pa·s, m, N/m para `sigma` — ver §4), enteramente
keyword-only (`CLAUDE.md` decisión cerrada #10).

**Precondición de signo (decisión de diseño 2026-08-04, implementada
2026-08-07)**: `liquid_mass_rate >= 0` y `gas_mass_rate >= 0`, con
`ValueError` fuera de ese rango — reemplazó la rama que existía de "flujo
reverso" (`liquid_mass_rate <= 0 and gas_mass_rate <= 0` → `abs()` +
inclinación invertida). `beggs_brill_gradient`/`_beggs_brill_detailed` no
conocen la orientación del edge; la dirección de flujo no es una feature de
`physics`, la resuelve el integrador aguas arriba (`solve_ivp` con
`t_span=(L, 0)` en vez de `(0, L)` — ver `ROADMAP.md` §Decisiones cerradas).
`loss_func` es quien adapta el signo antes de llamar a `physics`, no al
revés.

**API interna**: `_beggs_brill_detailed(*, ...) -> dict` — mismo cálculo más
intermedios (`NFr`, `Cl`, `liquid_holdup`, `ReNs`, `f`, `fNs`,
`mixture_density`, `flow_regime`). Misma firma kw-only que la pública. Es lo
que consumen los tests y el canal `@diagnostic` de nivel 1.

**Cómo llega al canal de diagnóstico** (decisión 2026-08-10, `CLAUDE.md` #25):
ningún solver la importa, y tampoco se la descubre por convención de nombre.
Se pasa **explícitamente** como `detailed_fn` al construir la `loss_func`, que
la invoca en post-proceso sobre el `P(x)` ya convergido. Sin `detailed_fn` la
loss entrega igual el nivel 0 (la descomposición de `GradientResult`).

El costo del patrón wrapper-sobre-detailed está medido: ~19% sobre el tiempo
de la correlación en el peor caso (escalar, correlación barata), diluido a
irrelevancia al vectorizar en v0.5 — se construye un dict por llamada de
array, no uno por elemento. No se justifica una firma polimórfica
(`-> GradientResult | dict`) para recuperarlo.

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

### 3.1 `test_beggs_brill_vs_book.py` — validación contra literatura

- **Kermit Brown, ejemplo 4.7**: caso de libro con NFr, Re, f (Darcy,
  book/4 para Fanning), holdup y gradientes gravitacional/friccional
  publicados. Tolerancias flojas (5–10%) porque el libro redondea
  intermedios y lee `f` de un diagrama de Moody.
- **checalc.com**: caso sanity sin corrección de Payne — no valida números
  exactos, valida invariantes físicos (`gradient.total < 0` en upflow,
  `Cl < Hl <= 1`).
- Logging INFO antes de cada assert: valor calculado, valor esperado, error
  relativo — para debuguear sin tener que parchear el test.

### 3.2 `test_beggs_brill_vs_fluids.py` — cross-validation con `fluids` (ChEDL)

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

### 3.3 Comportamiento de arrays (`test_multiphase_vector_1.py`)

Archivo separado de `test_beggs_brill_vs_book.py`: golden testea corrección
física (valores contra literatura/oráculo), este archivo testea el
*contrato de forma* (qué soporta arrays y qué no) — son preocupaciones
distintas, no redundancia. Recorre las funciones de `beggs_brill.py` en el
orden en que están definidas en el módulo (flowmap → holdup → detailed →
`beggs_brill_gradient`).

- `test_flowmap_vectorized`: `beggs_brill_flowmap` con arrays de 5 puntos,
  regímenes verificados uno por uno contra los boundaries L1–L4.
- `test_flowmap_rejects_out_of_domain`: NaN debe levantar `ValueError`
  (falsea todas las máscaras); `Cl=0` **no** sirve como probe porque cae en
  la máscara distributed.
- `test_holdup_vectorized_within_regime` (parametrizado en `i` 0–3):
  `_holdup` sí vectoriza sobre `Cl`/`NFr`/`Nlv` para un régimen fijo
  (`angle` se mantiene escalar — un caño, no un ángulo por punto). No
  estaba documentado en versiones previas de este ADR.
- `test_detailed_rejects_array_rates` / `test_gradient_rejects_array_rates`:
  documentan la causa concreta del contrato escalar — el chequeo de
  co/counter-flow (`if liquid_mass_rate > 0 and ...`) es ambiguo para
  arrays de más de un elemento y levanta `ValueError`.
- `test_detailed_vectorized_over_rates` / `test_gradient_vectorized_over_rates`:
  **`xfail(strict=True)`** — spec ejecutable del comportamiento deseado en
  v0.5 (rates vectorizados → gradientes vectorizados), a nivel interno y
  público. Fallan hoy a propósito; el día que se implemente el
  broadcasting, estos tests **pasan** y el `strict=True` pone la suite en
  rojo como recordatorio de sacar el marker y pinnear valores reales.
- `test_detailed_scalar_contract_today` / `test_gradient_scalar_contract_today`:
  fijan el contrato actual (escalar entra, floats salen) para no dejar que
  un array 0-d se filtre al `GradientResult` en un estado
  semi-vectorizado.

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
- **Unidad de `sigma` — cerrada e implementada.** SI estricto (N/m), no
  dyn/cm, en `_beggs_brill_detailed` y `beggs_brill_gradient` (defaults
  `30e-3`). Conversión `sigma * 1e-3` retirada de los golden tests
  (`test_beggs_brill_vs_book.py`) y de `test_multiphase_vector_1.py`;
  `fluids` espera N/m también, así que no queda compensación pendiente.
- **Vectorización**: no implementada (`_beggs_brill_detailed` es escalar);
  `beggs_brill_flowmap` sí. Ver roadmap v0.5 (broadcasting `pd.Series`).
- **Firma por fase de `compressibility` — pendiente.** `compressibility_gas`
  / `compressibility_liquid` en lugar del valor de mezcla, con la ponderación
  por holdup adentro de la correlación. Decidido 2026-08-09; toca
  `beggs_brill_gradient`, `_beggs_brill_detailed`,
  `test_beggs_brill_vs_fluids.py` y `test_beggs_brill_vs_book.py`. Fija la
  convención de sufijos antes de que entre la segunda correlación en v0.5.

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
