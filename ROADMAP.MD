# fluidnet — Roadmap de trabajo (v0.2 → v2.0)

> Documento de planificación. Design-first: cada etapa cierra decisiones de
> arquitectura **antes** de escribir código. El objetivo del proyecto es
> visibilidad profesional (portfolio + eventual paper JOSS), así que la
> claridad del diseño y la reproducibilidad pesan más que la cantidad de features.

> **Nomenclatura**: se usan números de versión reales (v0.2, v0.5, v1.0, v1.5,
> v2.0). Las etiquetas "MVP / v1 / v2" de versiones anteriores de este
> documento se retiraron: convivían con los números y generaban ambigüedad
> (el texto llegaba a decir "v1 — ≈ v0.5"). *(Unificado 2026-08-07.)*

---

## A dónde vamos

**Qué resuelve la librería.** Simulación steady-state de redes de flujo,
graph-first: el usuario trae su física (`Rate` + `Fluid` + `loss_func`
pluggables), la librería aporta topología, propagación, solvers y calibración.

**Objetivo del proyecto.** Visibilidad profesional y publicación peer-reviewed
(JOSS/SoftwareX). Esto pesa en cada decisión de scope: API limpia y documentada
> cantidad de features. *(Detalle y criterio de decisión: `VISION.md`,
archivo interno no versionado.)*

**Diferenciales y dónde se demuestra cada uno.** Ésta es la tabla que conecta
"qué nos hace publicables" con "qué se construye cuándo". Si una feature
propuesta no sirve a ninguna fila, es scope creep:

| Diferencial | Se demuestra en | Cómo |
|---|---|---|
| **Calibración de campo como diagnóstico** — residuales como señal de condición del activo, no error de ajuste | **v0.5** | solver de fitting + ejemplo de calibración contra observaciones |
| **Forward-propagation como régimen de primera clase** — el caso natural del gathering upstream | v0.2 | solver 1 + notebook wellfield |
| **Loss functions pluggables / arquitectura physics-agnostic** | v0.2 (declarado) → v2.0 (demostrado) | protocolos en v0.2; **necesita un segundo dominio** (térmico o eléctrico AC) para ser creíble como pitch |
| **`Rate` polimórfico que propaga composición** — mezcla de salmueras → índices de saturación | v1.5 | `Fluid` por nodo + `BrineRate` como extensión |
| **Broadcasting vectorizado** (escenarios/timestamps en una pasada) | v0.5 | prerequisito del fitting; diferencial vs. pandapipes |

Lectura importante de esta tabla: **el diferencial más defendible se demuestra
en v0.5**, y los otros dos fuertes son promesas hasta v1.5/v2.0. Es el
argumento estructural para no reordenar v0.5 detrás de nada.

---

## Principios que ordenan el plan

1. **Diseño antes que código.** Cada milestone define firmas públicas
   (docstrings + type hints) y tests de aceptación antes de implementar.
2. **Cada etapa valida algo ante un observador externo.** v0.2 valida que la
   *arquitectura* funciona; v0.5 valida que es *útil* y *diferencial*; v1.0+
   generaliza.
3. **Capas con dependencias en un solo sentido.** `physics/` y `Fluid` son
   capa cero (no importan nada del paquete); `Rate`, `Fluid` y `loss_func` no
   conocen la red; `Network` no conoce la física; los solvers orquestan;
   `Result` lo produce solo el solver.
4. **Core SI estricto.** Unidades de presentación solo en la capa de I/O.
5. **Commits chicos contra `main` limpia**, en el orden del mapa de rescate
   desde mineplanner.

---

## Dominio objetivo y por qué la secuencia es esta

El caso de uso primario es **gathering upstream**: campo de bombeo de
salmuera, campo de petróleo y gas, redes de minería. En ese dominio la
topología real es un DAG — sale del pozo, converge en headers, llega a
planta — y el régimen natural es **forward propagation**, no balance general.
Es exactamente el caso que pandapipes y WNTR tratan solo como subproducto de
un solver de balance.

Esto no es una afirmación sobre las redes de flujo en general: las redes de
distribución de agua potable y los circuitos hidráulicos de edificios son
fuertemente mallados y perfectamente pasivos, y ahí el ciclo es la norma (es
el terreno de EPANET/WNTR y de Hardy Cross). **La restricción DAG de las
primeras etapas es del solver, no de la física** — `forward_propagation`
necesita orden topológico para propagar. Un loop pasivo con losses monótonas
sigue siendo un problema convexo con solución única; simplemente requiere
otro método.

Por eso el orden: primero el régimen del dominio objetivo y el diferencial
(fitting), después las generalizaciones, y al final los casos que obligan a
abandonar la propagación topológica.

---

## Capa 0 — `physics/` · COMPLETADA

**Qué prueba:** que la física está validada de forma independiente de la
arquitectura de red. Es una sub-librería con tests propios y cero dependencias
del resto del paquete.

**Hecho**

- `dimensionless.py`, `friction.py`, `single_phase.py`, `multiphase.py` —
  funciones puras SI→SI, sin estado.
- `GradientResult` (NamedTuple) como contrato de retorno único mono/multifásico.
- Beggs & Brill (1973) cross-validado contra `fluids` v1.3.1 (ChEDL) como
  oráculo. Dos bugs encontrados y corregidos (interpolación de régimen
  transition; guard `EL(0) >= Cl`).
- Suite de tests en tres capas: golden de literatura (Kermit Brown ej. 4.7),
  pinneada vs. `fluids`, live vs. `fluids` (`importorskip`).
- `xfail(strict=True)` como spec ejecutable de la vectorización de v0.5.

**Contrato de la capa (importante para lo que sigue).** `physics` es un
evaluador **puntual de gradiente**: entra un estado local en escalares SI, sale
`GradientResult` en **Pa/m**. No conoce `Rate`, ni `Fluid`, ni longitud de edge,
ni dirección de propagación, ni si su resultado se va a integrar o a
multiplicar por `L`. **Tampoco ve `T` ni composición**: recibe propiedades ya
evaluadas (`ρ`, `μ`, `β`, `σ`). Todo lo que sea integración sobre la cañería,
adaptación de firma o diagnósticos pertenece a la capa `loss_func`, no acá.

**Corolario de que la capa se valide sola.** `physics/` puede adelantarse a los
solvers: es legítimo que una capacidad exista acá antes de que ningún solver
pueda usarla. Es el caso de la vectorización de B&B (v0.5, ver ahí) —
multifásico no es invocable hasta que exista `IntegralLoss` en v1.0. No es un
olvido de secuencia: es el contrato de capa cero funcionando.

**Freeze — con una distinción.** Se congelan las **correlaciones empíricas**
(B&B, modelos de holdup): no se agregan hasta v0.5. El diferencial del proyecto
es red + propagación + fitting; en correlaciones no hay diferencial (`fluids` ya
cubre ese terreno) y no aportan visibilidad. Las **ecuaciones puras**
(`dimensionless`, `friction`, `single_phase`) sí pueden extenderse, pero solo
cuando un solver o una `loss_func` lo requiera — nunca por completitud física.

---

## Capa 0 bis — `Fluid` · DISEÑO CERRADO, SIN IMPLEMENTAR

Hermana de `physics/`, mismo contrato de capa cero: función pura, SI, sin
conocer nada de arriba. **Fábrica stateless de `FluidState`.**

```
Fluid.get_state(*, pressure, temperature=None, composition=...) -> FluidState
FluidState = NamedTuple(density, viscosity, compressibility, sigma)
```

- Lo que vive en la red/nodo no es *un fluido con densidad 1000*, es el
  **modelo** que mapea `(composición, P, T) → propiedades`. Nada se congela.
- Recibe composición como **dato crudo**, nunca un objeto `Rate` (si recibiera
  el `Rate` dejaría de ser capa cero).
- Es quien **declara el régimen** `AlgebraicLoss`/`IntegralLoss`: es el dueño
  del EOS, el único que sabe si `∂ρ/∂P = 0`.

**Scope v0.2**: `IncompressibleFluid` (agua/salmuera simple, ignora `T`) es
suficiente. `IsothermalGas` y fluidos composicionales llegan en v1.0/v1.5.

---

## v0.2 · "La arquitectura funciona end-to-end"

**Qué prueba:** que un ingeniero senior lea el repo en 20 min y confirme que las
capas (`physics` / `Fluid` / `Rate` / `loss_func` / `Network` / solver →
`Result`) encajan. No es un producto todavía; es la prueba estructural.

**Topología**: DAG, single sink. Los nodos intermedios son **nodos de paso**:
sin condición de borde propia, solo confluencia. BC = rates en sources +
presión en el sink.

**Fluido**: **uno solo para toda la red** (`network.fluid`). Caso homogéneo —
todo agua, o pozos de composición equivalente.

**Scope IN**

- `Rate` (ABC/Protocol) + `ScalarRate` con álgebra: `__add__`, `__radd__`
  (para que `sum()` no rompa el polimorfismo), `__mul__`. Contenido:
  magnitud extensiva + composición intensiva, **sin propiedades de fluido**.
  → *forma exacta de la composición trivial pendiente; ver "Abiertas".*
- `Fluid` + `FluidState` con `IncompressibleFluid` como única implementación.
  Propiedades derivadas del `Fluid`, nunca almacenadas en el `Rate` ni como
  atributo suelto de red.
- Protocolo `AlgebraicLoss` implementado. `IntegralLoss` **declarado como
  interfaz documentada** pero sin implementar (llega en v1.0).
- `Network`: wrapper `nx.DiGraph`, validación DAG, `propagate_rates` /
  `propagate_heads`, `to_frames()`, atributo `fluid` de red. Sin clases
  Node/Edge.
- Solver 1: `forward_propagation`.
- `Result` (dataclass frozen): terna garantizada `(rate, p_up, p_down)` por
  edge + `(rate, p)` por nodo + metadata del solve; `to_frames()`.
- Built-ins de loss: `constant_friction` (baseline para tests/docs) +
  `darcy_weisbach`, **construido sobre `physics/single_phase.py`** (no
  reimplementa física). La capa de loss aporta lo que `physics` no hace:
  1. pedirle el `FluidState` al `Fluid` y adaptarlo a escalares SI,
  2. **integrar el gradiente sobre la longitud del edge** (`Pa/m` → `Pa`;
     trivial para incompresible: `grad.total * L`),
  3. aplicar `@diagnostic` (f, Re, v al canal lateral).
  `darcy_weisbach` cubre fluidos incompresibles — `compressibility=0` es la
  *consecuencia* de que el `Fluid` sea incompresible, no una decisión de
  configuración que `darcy_weisbach` tome.
- Tests del core (los de physics, ~20, ya son el piso) + un notebook demo:
  wellfield sintético.
- README con statement of need explícito vs. pandapipes / WNTR, incluyendo la
  aclaración de que DAG es restricción del solver y no de la física.

**Scope OUT**: todo lo de las etapas siguientes.

**Criterio de salida:** notebook corre punta a punta · CI verde · README
posicionado · tag `v0.2`.

---

## v0.5 · "Es útil para un caso real" — **JOSS**

**Qué prueba:** que alguien con un problema real de wellfield/piping lo pueda
usar. Es el punto de submit a JOSS y **el diferencial del proyecto**.

> **Esta etapa no se mueve ni se reordena.** El fitting es lo que ninguna
> librería open source cubre y lo que hace el paper defendible. Es ortogonal a
> las generalizaciones de topología y fluido que vienen después: corre sobre
> `forward_propagation` con un solo fluido y BC simples. El riesgo a vigilar es
> postergarlo detrás de features de física que son más entretenidas de diseñar
> y menos defendibles ante un reviewer.

**Scope IN**

- Solver 3: **fitting** — `EdgeParameter` declarativo (qué parámetro, en qué
  edges, bounds) + optimizer que reutiliza el solver 1 vectorizado como forward.
  Residuales tratados como señal de condición de activo, no solo error de ajuste.
- **Broadcasting `pd.Series`**: múltiples escenarios/timestamps en una pasada.
  Prerequisito del fitting y diferencial vs. pandapipes.
- Vectorización de `_beggs_brill_detailed` (hoy escalar). El
  `xfail(strict=True)` existente es la spec: cuando pase, sacar el marker y
  pinnear valores reales. **Nota de secuencia**: B&B no es invocable por ningún
  solver hasta `IntegralLoss` (v1.0); se vectoriza acá porque impacta el
  optimizer y porque `physics/` se valida sola (ver corolario en Capa 0).
- Decorador `@diagnostic` resuelto → `Result` con `f`, `Re`, `v` por edge
  cuando el solver corre con `full_output=True`.
- **Fin del freeze de correlaciones**: modelos de holdup no-slip (`Hl = Cl`) y
  constant-slip, como baseline y puente pedagógico hacia B&B. *Diferidos desde
  `physics-single-multiphase.md` §5.*
- Docs formales: docstrings numpy · Sphinx/mkdocs en GitHub Pages ·
  2-3 ejemplos reproducibles (uno de calibración contra observaciones sintéticas).
- Empaquetado PyPI real + CHANGELOG + versionado semántico.

**Criterio de salida:** `pip install fluidnet` funciona · docs online · caso de
fitting demostrado · submit a JOSS.

---

## v1.0 · "Condiciones de borde generales"

**Qué prueba:** que la arquitectura soporta redes especificadas de forma
arbitraria, no solo el patrón source→sink.

**Cambio de topología**: los nodos intermedios dejan de ser solo nodos de paso —
pueden llevar condición de borde propia (extracción, inyección, presión
impuesta). El grafo sigue siendo un DAG.

**Scope IN**

- BC generalizadas **2-de-3** `(Q, P_up, P_down)` por nodo/edge, con validación
  explícita de resolubilidad (falla claro, nunca adivina).
- Solver 2: **mass_balance** — BC arbitrarias mezcladas (head/rate) vía
  `scipy.optimize`. Requiere definir la magnitud de balance escalar de `Rate`.
- `IntegralLoss` implementado: integración de `dp/dL` edge a edge sobre el mismo
  esqueleto del solver 1 (pasando `p_boundary`). Cubre monofásico compresible
  (`compressibility != 0`) y multifásico (B&B) con el mismo mecanismo.
- `IsothermalGas` y fluidos compresibles como implementaciones de `Fluid`.
- Trayectoria 3D como atributo de edge → perfil `P(x)` en `Result`
  (la integración por tramos ya la habilita `IntegralLoss`).

**Criterio de salida:** 3 solvers operativos · un caso multifásico documentado ·
tag `v1.0`.

---

## v1.5 · "El fluido depende de la mezcla"

**Qué prueba:** el diferencial de `Rate` polimórfico llevado a su consecuencia
completa — la composición propagada determina las propiedades locales.

**Scope IN**

- **`Fluid` por nodo**: la composición propagada del `Rate` determina el `Fluid`
  local, precalculado en la misma pasada topológica que los rates. Resolución
  `G.nodes[n].get('fluid', network.fluid)` — v0.2–v1.0 quedan como el caso
  degenerado, no se reemplaza mecanismo. Un edge toma el fluido de su nodo
  `upstream`; la mezcla ocurre en el nodo receptor, nunca en el caño.
- Lazo externo de Picard para el acoplamiento composición↔propiedades en
  `mass_balance` (ver "Abiertas" — dirección propuesta, no cerrada).
- **Temperatura como input**: perfiles de `T` prescritos como atributo de
  nodo/edge, consumidos por las leyes constitutivas de edge. Aditivo: el solver
  pasa `T` donde antes pasaba `None`, sin cambio de protocolo. Solving térmico
  acoplado queda fuera.
- Extensiones de ejemplo, **fuera del core**: `BrineRate` (composición iónica →
  saturation indices, mezcla de salmueras), `MultiphaseRate` (fracciones).

---

## v2.0 · "Redes cerradas y elementos activos"

**Qué prueba:** que la arquitectura no estaba atada al régimen de propagación
topológica. Sale del dominio objetivo primario (gathering upstream) hacia
circuitos cerrados: hidráulica de edificios, procesos con recirculación.

**Scope IN**

- **Ciclos/loops pasivos**: levanta la restricción DAG. Un loop pasivo con
  losses monótonas sigue siendo convexo — el método cambia, la garantía no.
  Terreno de EPANET/WNTR/Hardy Cross.
- **Elementos activos** (bombas, compresores): nodos/edges que *aumentan* el
  potencial. Habilita recirculación y redes genuinamente cerradas. Es un
  problema distinto y más duro que el loop pasivo: rompe la monotonía global.
- Unidades de presentación en `Result` (pre/post-proceso; core sigue SI estricto).
- **Segundo dominio demo** (térmico o eléctrico AC) para respaldar el pitch de
  arquitectura physics-agnostic — topológicamente idéntico, `ComplexRate` entra
  sin tocar el álgebra.

---

## Secuencia inmediata (dentro de v0.2, en orden)

1. **Cerrar la forma concreta de `Rate`/`ScalarRate` y `Fluid`/`FluidState`** —
   el split extensivo/intensivo ya está cerrado (2026-08-07); falta la forma
   exacta de la composición trivial de `ScalarRate` y el adaptador
   `as_physics_kwargs()`.
2. **Diseño del caso demo** — red sintética de wellfield: inputs, outputs, qué
   muestra el notebook. Define los tests de aceptación y tensiona las firmas de
   `Rate`/`Fluid` contra un uso real. Sin código.
3. **Firmas públicas restantes** — los dos `Protocol` de loss y `Result`, a
   nivel de interfaz (docstrings + type hints), no implementación.
4. **Mecanismo de `@diagnostic`** (contextvar vs. colector) — bloquea
   `darcy_weisbach`.
5. **Implementar** por piezas chicas siguiendo el mapa de rescate del ADR §3,
   commiteando de a poco contra `main` limpia.

---

## Decisiones

### Cerradas

- **`physics/` es capa cero.** Funciones puras SI→SI, sin estado ni conocimiento
  de red. Contrato de salida: `GradientResult` en Pa/m.
- **`Fluid` es capa cero también** — fábrica stateless de `FluidState`,
  hermana de `physics/`. Recibe composición cruda, nunca un `Rate`. *(2026-08-07)*
- **`Rate` = extensivo + composición intensiva, sin propiedades de fluido.**
  Reemplaza la decisión previa. Razón: el álgebra `__mul__`/`__add__` queda bien
  definida sin reglas ad-hoc, y `Rate` se mantiene homogéneo numéricamente
  (`ComplexRate` para red AC entra sin tocar nada). El diferencial de mezcla de
  salmueras sobrevive: lo que se propaga es composición, las propiedades se
  derivan. *(2026-08-07)*
- **`FluidState` con todos los campos requeridos** — `(ρ, μ, β, σ)`, nada de
  `float | None`. *(2026-08-07)*
- **`temperature` en la firma desde v0.2, `None` = "no suministrado"**, jamás un
  default físico. *(2026-08-07)*
- **Discriminador `AlgebraicLoss`/`IntegralLoss`: lo declara el `Fluid`.**
  Es el dueño del EOS, el único que sabe si `∂ρ/∂P = 0`. La `loss_func` hereda
  el régimen del fluido que recibe. *(2026-08-07)*
- **Dueño del `Fluid`: red hasta v1.0, nodo en v1.5, mismo mecanismo**
  (`G.nodes[n].get('fluid', network.fluid)`). Un edge toma el fluido de su nodo
  `upstream`. *(2026-08-07)*
- **Secuencia de cinco etapas por generalización progresiva** *(2026-08-07)*:
  v0.2 DAG+nodos de paso+un fluido → v0.5 fitting (JOSS, no se mueve) → v1.0 BC
  en nodos intermedios + mass_balance → v1.5 fluido dependiente de composición →
  v2.0 loops y elementos activos. Criterio: primero el régimen del dominio
  objetivo y el diferencial, después las generalizaciones, al final lo que
  obliga a abandonar la propagación topológica. **BC intermedias van antes que
  composición variable** a propósito: son lo que habilita `mass_balance`, que es
  donde la dependencia de datos se vuelve circular; con composición primero,
  llegarían dos fuentes de dificultad mezcladas y no se sabría cuál rompe la
  convergencia.
- **La restricción DAG es del solver, no de la física.** `forward_propagation`
  necesita orden topológico; un loop pasivo con losses monótonas sigue siendo
  convexo y con solución única. Debe decirse explícito en el README: un reviewer
  que venga de EPANET va a leer "DAG required" como desconocimiento del caso
  mallado. *(2026-08-07)*
- **Freeze de correlaciones empíricas hasta v0.5.**
- **La integración vive en `loss_func`, no en `physics`.**
- **`@diagnostic` sobre `full_output` en la firma.**
- **Dos protocolos de loss, no uno.**
- **La loss recibe el `Rate` entero**, no atributos sueltos.
- **SI estricto en el core.**
- **MVP de 26 tests: declarado perdido y superado.** La spec de reconstrucción
  es `docs/design/architecture-v0.2.md` + este roadmap.
- **Firma de `physics/`: keyword-only completo** (ver `CLAUDE.md` #15).
- **Caudal negativo / dirección de flujo: no es feature de `physics`.** La
  resuelve el integrador invirtiendo el sentido de integración (`solve_ivp` con
  `t_span=(L, 0)`).
- **Módulo `fluidnet/fluid.py` en singular**, no `fluids.py` — colisión visual
  con `fluids` (ChEDL). *(2026-08-07)*

### Abiertas

- **¿Adelantar el segundo dominio demo de v2.0 a v0.5?** La fila
  "physics-agnostic" de la tabla de diferenciales es la más débil: está
  *declarada* en v0.2 pero *demostrada* recién en v2.0, o sea que durante todo
  el camino al submit de JOSS es una afirmación sin evidencia. Un segundo
  dominio (red térmica o eléctrica AC) es topológicamente idéntico — el costo
  es el notebook y su validación, no el core. Adelantarlo daría dos
  diferenciales demostrados en el paper en vez de uno.
  **Contra**: agranda v0.5, que es justamente la etapa que no se debe dilatar.
  A evaluar al diseñar el caso demo (Secuencia inmediata #2), aunque sea para
  descartarlo por tiempo.

- **Forma de la composición trivial en `ScalarRate`** — ¿vacía, o un singleton
  tipo `{"fluid": 1.0}` para que el álgebra de mezcla sea uniforme entre
  `ScalarRate` y `BrineRate`? Bloquea la firma final de `__add__`.
- **`as_physics_kwargs()`** — dónde vive el adaptador que traduce
  `(Rate, FluidState, edge_attrs)` a los kwargs exactos de cada función de
  gradiente. Candidato natural: la `loss_func`, única capa que ve las tres cosas
  a la vez.
- **Acoplamiento composición↔propiedades en `mass_balance` (v1.5).** Con
  composición fija el problema hidráulico es convexo (content de Millar):
  solución única bajo losses monótonas. Cuando las propiedades dependen de la
  composición se pierde la función potencial y con ella la garantía de
  existencia/unicidad. **Aparece en DAG, no requiere loops**: en
  `forward_propagation` los caudales salen de propagación directa, pero en
  `mass_balance` son incógnita y la dependencia de datos se vuelve circular
  aunque el grafo sea acíclico. **Dirección propuesta**: lazo externo de Picard
  — campo de composición congelado (por nodo, no global) → solver convexo →
  re-propagar composición → repetir. Converge bajo baja sensibilidad de
  propiedades a la mezcla, que es el régimen del caso de uso objetivo
  (salmueras de densidad similar, pozos de composición equivalente). La no
  convergencia se reporta como resultado con significado físico (red mal
  condicionada), nunca como excepción silenciosa. El `Fluid` stateless es lo que
  hace el lazo barato: no hay caché que invalidar entre iteraciones.
  **Abierto**: criterio de corte; y si en v2.0 la inversión de sentido de flujo
  en loops requiere regularización o basta con detectarla y abortar.
- Mecanismo del canal de `@diagnostic` (contextvar vs. objeto colector) —
  bloquea `darcy_weisbach`.
- Ubicación de `GradientResult`: hoy vive en `single_phase.py` pero es el
  contrato compartido con `multiphase.py`. Candidato a `physics/types.py`.
  Misma pregunta para dónde vive `FluidState`.
- Port fino del solver 3: revisar `header_network_optimizer.py` y
  `network_observations.py` de mineplanner.
