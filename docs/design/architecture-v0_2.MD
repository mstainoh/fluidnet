# fluidnet — Draft 1: Arquitectura v0.2

> Sucede al Draft 0. Incorpora: (a) verificación del landscape con pandapipes incluido, (b) el código real de mineplanner (`header_network.py`, `pressure_functions.py`, `run_fit_network.py`) como referencia de diseño probado en producción, (c) las decisiones de diseño acordadas: result container vía decorador, sin clases Node/Edge, tipado estricto, separación `Rate` / `Fluid`.

---

## 1. Posicionamiento

**Qué es**: librería Python graph-first para simulación steady-state de redes de fluido. El usuario trae su propia física (`Rate` polimórfico + `Fluid` pluggable + loss functions pluggables); la librería aporta topología, propagación, solvers y calibración.

**Objetivo del proyecto**: visibilidad profesional (portfolio GitHub) → publicación en **JOSS** (Journal of Open Source Software) o SoftwareX como antecedente académico. Esto condiciona el diseño: API limpia y documentada > features; tests y ejemplos reproducibles son parte del producto, no accesorios.

**Landscape verificado (07/2026)** — qué existe y por qué fluidnet no es redundante:

| Tool | Dominio | Limitación relevante |
|---|---|---|
| **pandapipes** (Fraunhofer) | gas / district heating, multi-energy grids | monofásico, un fluido por red, física cerrada, sin fitting de parámetros orientado a calibración de campo |
| **WNTR** (EPA) | agua potable, resiliencia | atado a EPANET, agua únicamente |
| **TESPy** | ciclos térmicos | componentes, no redes de tubería genéricas |
| **fluids** | correlaciones puntuales | sin capa de red |
| PIPESIM / PIPEPHASE / OLGA | gathering multifásico O&G | comerciales, cerrados |

**Diferencial de fluidnet** (statement of need para JOSS): ninguna librería open source combina (1) rate polimórfico que propaga **composición** por la red (mezcla de salmueras → saturation indices, multifásico), (2) loss functions pluggables con protocolo mínimo, (3) régimen forward-propagation como caso de primera clase (estándar en gathering O&G/minería, que pandapipes/WNTR tratan solo vía balance general), (4) fitting de parámetros vectorizado contra observaciones de campo.

Sobre (1), la distinción precisa frente a pandapipes: no es "un fluido por red" vs. "muchos fluidos por red". Es que fluidnet declara **un modelo de fluido**, no un fluido con propiedades congeladas — la composición local sale de la propagación y las propiedades se derivan de ella (ver §2.1bis). En v0.2 el modelo es único para la red (caso homogéneo: todo agua, o pozos de la misma composición); en v1.5 la composición propagada determina el fluido nodo a nodo. Es un cambio de proveedor del `Fluid`, no un rediseño.

Sobre (3), el argumento de dominio: el caso de uso primario de fluidnet es **gathering upstream** — campo de bombeo de salmuera, campos de petróleo y gas, redes de minería. En ese dominio la topología real es un DAG (sale del pozo, converge en headers, llega a planta) y el régimen natural es forward propagation, no balance general. pandapipes y WNTR resuelven ese caso solo como subproducto de un solver de balance, que es la herramienta correcta para *su* dominio (distribución de agua, grids multi-energía) y sobredimensionada para éste. La diferencia no es de capacidad sino de régimen de primera clase.

Importante para no sobrevender: esto **no** implica que las redes de flujo sean acíclicas en general (ver §2.3). Los ciclos son la norma en distribución de agua y en circuitos de edificios, y llegan en v2.0. El posicionamiento correcto es "primero el régimen del dominio objetivo", no "las redes son DAG".

El README debe posicionarse explícitamente contra pandapipes/WNTR: es la primera pregunta de cualquier reviewer.

---

## 2. Arquitectura de capas

```
┌─────────────────────────────────────────────────┐
│  Solvers (3 regímenes explícitos)               │
│  forward_propagation │ mass_balance │ fitting   │
├─────────────────────────────────────────────────┤
│  Network  (wrapper de nx.DiGraph, sin física)   │
├──────────────────────┬──────────────────────────┤
│  Rate (polimórfico)  │  loss_func (protocolo)   │
├──────────────────────┼──────────────────────────┤
│  Fluid → FluidState  │  physics/ (gradientes)   │  ← capa cero
├──────────────────────┴──────────────────────────┤
│  Result (contrato de salida)                    │
└─────────────────────────────────────────────────┘
```

Regla de dependencias: `physics/` y `Fluid` son capa cero y no importan nada del paquete. `Rate`, `Fluid` y `loss_func` no conocen la red. `Network` no conoce la física (solo plumbing de grafo). Los solvers orquestan. `Result` es producido exclusivamente por solvers.

**Cadena de evaluación canónica**:

```
Rate ──► composición (intensiva, se propaga por la red)
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

### 2.1 Rate — la abstracción central

Dataclass (frozen) con álgebra:

- `__add__` → mezcla en nodos de convergencia. Para `ScalarRate` es suma; para rates composicionales, suma de extensivos + mezcla de composición ponderada por caudal.
- `__mul__` (escalar) → scaling, necesario para el optimizer y para escenarios vectorizados.

**Contenido: magnitud extensiva + composición intensiva. Sin propiedades de fluido.** Esta es una corrección explícita del Draft 0 y de las primeras versiones de este documento, que ponían densidad y viscosidad adentro del `Rate`. El rationale del cambio (sesión de diseño 2026-08-07):

- **El álgebra queda bien definida.** Si `density` es un campo del `Rate`, `rate * 2` es ambiguo: el caudal escala, la densidad no. Habría que escribir una regla ad-hoc "escalá los extensivos, dejá los intensivos" *dentro* de la operación algebraica — exactamente el tipo de cosa que se rompe cuando aparece el tercer tipo de rate.
- **`Rate` se mantiene homogéneo numéricamente.** Un `Rate` es representable como float, array o complejo, y en todos los casos es optimizable. Con un `density: float` congelado adentro, el tipo deja de ser homogéneo. Un `ComplexRate` (fasores, redes AC — el segundo dominio de demo) entra sin tocar el álgebra.
- **Rate es un input, no un cálculo.** El `Rate` es el argumento de la correlación (o el output, si estamos haciendo root solving). El fluido es lo que entra en la fórmula de gradiente. Son roles distintos y conviene que sean tipos distintos.
- **El diferencial de mezcla de salmueras no se pierde.** Lo que se propaga por la red no son *propiedades*, es **composición** — que sí es lineal bajo mezcla ponderada por caudal e invariante bajo scaling. Las propiedades nunca se almacenan: se derivan del `Fluid` en el punto donde se necesitan, con la presión y temperatura locales. Esto también satisface el requisito de que el estado del fluido sea función de `P` y no una constante congelada.

Jerarquía mínima v0.2: `Rate` (protocolo/ABC) → `ScalarRate` (monofásico, cubre agua/salmuera simple; composición trivial). El usuario extiende sin tocar el core: `BrineRate` (composición iónica → saturation indices), `MultiphaseRate` (fracciones), quedan como ejemplos/extensiones, no core.

**Decisión heredada de mineplanner a corregir**: `propagate_rates` usa `sum(propagated[p] for p in predecessors)` — `sum()` arranca de `0` (int). `Rate` debe soportar `0 + rate` (`__radd__`) o la propagación usa `functools.reduce`. Detalle chico, pero es exactamente el bug que rompe el polimorfismo.

### 2.1bis Fluid — fábrica de estado, no estado

`Fluid` es **stateless**: no es un objeto con densidad 1000, es el **modelo** que sabe mapear `(composición, P, T) → propiedades`.

```
Fluid.get_state(*, pressure, temperature=None, composition=...) -> FluidState
FluidState = NamedTuple(density, viscosity, compressibility, sigma)
```

Esto reconcilia dos cosas que parecen chocar: el fluido es un atributo *declarativo* de la red (lo pone el usuario) y a la vez el rate lo *afecta* (la composición sale de la propagación). Se resuelve porque lo que se declara es el modelo, no el valor.

**Contrato de capa cero.** `Fluid` recibe la composición como **dato crudo** (mapping / array), nunca un objeto `Rate`. Si recibiera el `Rate` dejaría de ser capa cero y aparecería una dependencia circular conceptual entre la abstracción de caudal y la de fluido.

**`FluidState` tiene todos los campos requeridos.** Nada de `float | None`, aunque `sigma` y `compressibility` no se usen en la mayoría de los casos monofásicos. Razones: (a) `β = 0` es el valor físico exacto de un líquido incompresible, no una ausencia — ya cerrado en `physics-single-multiphase.md` §1; (b) un `sigma=None` filtrándose a B&B es el mismo problema que un default físico invisible, y además rompe el filtrado de kwargs por `signature` (un campo faltante no es lo mismo que un campo presente que la firma no pide). Un fluido monofásico puede levantar `NotImplementedError` en `sigma` si se quiere ser explícito, pero no devolver `None`.

**`temperature` está en la firma desde v0.2, con default `None` que significa "no suministrado", nunca un valor implícito.** Todo fluido es en general función de `(P, T)`; se deja opcional para los casos en que `T` es irrelevante (incompresible) o ya fue fijada. Reglas: el fluido incompresible la ignora; el gas isotérmico la fija en construcción (`IsothermalGas(T=323.15)`), no en el call site; un fluido que la necesita y recibe `None` levanta `ValueError`. Con eso `None` nunca encubre una hipótesis de modelado.

La consecuencia de tenerla desde el día uno es que v2 (perfiles de temperatura prescritos) es **aditivo**: el solver pasa `T` desde un atributo de nodo/edge en vez de `None`, sin cambio de protocolo. Agregar el parámetro después habría roto la firma.

**Dueño del `Fluid`**: atributo de red hasta v1.0, atributo de nodo desde v1.5, con resolución `G.nodes[n].get('fluid', network.fluid)`. El fluido único de red es el caso degenerado del fluido por nodo, no un mecanismo distinto que después se reemplaza. En v1.5 el fluido de cada nodo se precalcula en la misma pasada topológica que ya hace `propagate_rates` — no hay maquinaria nueva. **Un edge toma el fluido de su nodo `upstream`**: la mezcla ocurre *en* el nodo receptor, nunca dentro del caño. En v1 es una identidad, pero la semántica queda escrita desde ahora.

### 2.2 loss_func — dos protocolos, no uno

El código de mineplanner ya reveló que hay dos firmas distintas según el régimen físico:

**Protocolo algebraico** (monofásico, incompresible — el dp no depende de la presión absoluta):

```
loss_func(rate: Rate, fluid: Fluid, **edge_attrs, **network_attrs) -> float
```

**Protocolo de integración** (multifásico / compresible — el dp depende de p local, se integra dp/dL desde la frontera conocida):

```
loss_func(rate: Rate, fluid: Fluid, p_boundary: float, **edge_attrs, **network_attrs) -> float
```

Ambos declarados como `typing.Protocol` explícitos (`AlgebraicLoss`, `IntegralLoss`). Convención de signos heredada y confirmada: `dp = p_downstream − p_upstream` (pérdida → negativo). Kwargs extra absorbidos con `**kwargs` — el patrón de mineplanner de "declarar solo lo que usás" funciona bien y se mantiene.

Consecuencia directa de que `physics/` sea enteramente keyword-only (`CLAUDE.md` decisión #15): `loss_func` puede despachar hacia la función de gradiente correcta filtrando sus propios kwargs por `inspect.signature(gradient_fn).parameters` en vez de necesitar un adaptador por modelo. `physics` tampoco conoce la orientación del edge — es `loss_func` quien adapta signo/dirección (caudal, inclinación) antes de invocar la función de gradiente; la dirección de integración en sí la resuelve el solver (`t_span` de `solve_ivp`), no `physics` ni `loss_func`.

**El discriminador `AlgebraicLoss`/`IntegralLoss` lo declara el `Fluid`.** *(Cerrado 2026-08-07; este documento lo listaba como abierto.)* El problema era real: qué protocolo aplica es un valor de *runtime* (`compressibility == 0`), no una propiedad estática de la firma de `loss_func` ni de `gradient_fn`, así que no se podía leer de `signature`. Y hay una tensión adicional: `AlgebraicLoss` no tiene presión en la firma — es literalmente su definición — así que si el `Fluid` siempre necesitara `P` para devolver `ρ`, una loss algebraica no podría llamarlo.

La salida no es que lo declare el usuario al registrar la loss ni que lo chequee el solver: **es una propiedad del `Fluid`**. El `Fluid` es dueño del EOS, así que es el único que sabe si `∂ρ/∂P = 0`. Un fluido incompresible expone propiedades sin necesitar `P` y por lo tanto es compatible con `AlgebraicLoss`; un fluido compresible exige `P` y solo es compatible con `IntegralLoss`. La `loss_func` hereda el régimen del fluido que recibe. Es coherente con "ningún solver adivina" (§2.4) y con que `compressibility` sea estado y no flag.

**Diagnósticos vía decorador** — la solución al conflicto "contrato escalar estricto vs. resultado rico":

- El protocolo exige `-> float`. Punto. Tipado estricto, usuario casual escribe 5 líneas.
- Un decorador `@diagnostic` permite a la función registrar intermedios (f, Re, v, régimen de flujo, perfil P(x)) en un canal lateral que **el solver** recolecta cuando se pide `full_output=True`. La loss function nunca devuelve un dict al solver.
- El `full_output: bool` de `darcy_dp` actual desaparece de la firma pública (queda como mecanismo interno del decorador).

Built-ins v0.2: `constant_friction` (baseline trivial, para tests y docs) y `darcy_weisbach` (portado de `pressure_functions.py`, que ya está en buen estado: Chen approx, régimen laminar/transición/turbulento, minor losses K, vectorizado numpy). `darcy_weisbach` obtiene las propiedades pidiéndole el `FluidState` al `Fluid`, no leyéndolas del `Rate` ni de la red.

### 2.3 Network — plumbing sin física

Wrapper de `nx.DiGraph` con validación (DAG requerido en v0.2).

**La restricción DAG es del solver, no de la física — y conviene decirlo explícito.** `forward_propagation` necesita orden topológico para propagar rates aguas abajo y presiones aguas arriba; de ahí sale la restricción. No es una afirmación sobre qué redes de flujo existen: las redes de distribución de agua potable y los circuitos hidráulicos de edificios son fuertemente mallados y completamente pasivos, y un simple caño en paralelo entre dos nodos ya es un ciclo. Matemáticamente el loop pasivo no rompe nada: con losses monótonas el problema sigue siendo convexo (content de Millar) y la solución sigue siendo única — solo requiere otro método de solución (Hardy Cross, o el solver de balance de EPANET/WNTR). Los ciclos entran en v2.0 junto con los elementos activos.

Esto tiene que quedar claro en el README: un reviewer que venga de EPANET va a leer "DAG required" como desconocimiento del caso mallado si no se aclara. El argumento correcto es de dominio, no de generalidad — ver §1.

**Sin clases Node/Edge**: los atributos viven en los dicts nativos de networkx (`G.nodes[n]`, `G.edges[u,v]`); las vistas tabulares las dan `to_frames()` (portado de `graph_edges_to_frame`/`graph_nodes_to_frame`).

El `Fluid` de red es un atributo de `Network`, pero es el único residuo de física que se admite ahí — y es un *modelo* declarado por el usuario, no un cálculo. En v2 se mueve a atributo de nodo con fallback a la red.

Se rescata directamente de mineplanner (código ya probado en producción):

- `GenericNetwork` ABC: validación DAG, `new_with_settings`, `__add__` vía `nx.compose`, `copy`, subgraph helpers, `topological_sort_edges`.
- Separación `propagate_rates` / `propagate_heads` (topological sort forward para rates, reversed para heads).
- Broadcasting vectorizado: rates como `pd.Series` (múltiples escenarios/timestamps en una pasada) — esto es un diferencial real vs. pandapipes y clave para el fitting.

Se corrige:

- Condiciones de borde: hoy son `sink_head` (escalar, single sink) + rates en sources. v0.2 generaliza a **especificación 2-de-3 por nodo/edge** `(Q, P_up, P_down)` — pero el solver forward valida que la especificación dada sea resoluble por propagación y falla explícito si no (nunca adivina).
- `clip_min_head` sale del path de propagación (esconde errores físicos); si se mantiene, es opt-in con warning.
- `density` como atributo de red desaparece: es propiedad derivada del `Fluid`, no un número guardado en el grafo.

### 2.4 Solvers — tres regímenes explícitos

El usuario elige el régimen; ningún solver "adivina":

1. **`forward_propagation`** — red convergente (DAG, single sink en v0.2), BC = rates en sources + presión en sink. Caudales por suma directa, presiones por back-propagation. **Punto de partida y caso demo.** Con `IntegralLoss` el mismo esqueleto cubre multifásico (pasa `p_boundary` edge a edge).
2. **`mass_balance`** — combinaciones arbitrarias de BC (head/rate mezclados), resuelto con `scipy.optimize` (fsolve/root). Requiere que el residual sea escalar → define qué debe exponer `Rate` (una magnitud de balance, e.g. caudal másico total).
3. **`fitting`** — calibración de parámetros de edge contra observaciones. Se porta el diseño declarativo de mineplanner: `EdgeParameter` (qué parámetro, en qué edges, bounds) + optimizer que reutiliza el solver 1 vectorizado como forward model. El runner (`run_fit_network.py`) demuestra el patrón end-to-end: observaciones long-format → pivots → fit paralelo por header → residuales base vs. fitted.

### 2.5 Result — el contrato de salida

Producido por el solver, nunca por loss functions. Dataclass frozen:

- **Garantizado siempre**: la terna completa `(rate, p_up, p_down)` por edge; `(rate, p)` por nodo; metadata del solve (régimen, convergencia, iteraciones).
- **Opcional / lazy**: diagnósticos del decorador (f, Re, v por edge), perfil `P(x)` si el edge tiene planimetría (trayectoria 3D como atributo de edge), propiedades derivadas del `Fluid` en cada nodo (composición de mezcla, `FluidState` local, saturation indices vía extensión).
- **Vistas**: `.to_frames() -> (df_edges, df_nodes)`, `.to_graph() -> nx.DiGraph` anotado (patrón de `get_propagated_network` actual, útil para plotting).

`Result` es también la interfaz de consumo externo (mineplanner u otro pipeline): un objeto serializable con vistas pandas, sin exigir que el consumidor conozca networkx.

---

## 3. Mapa de rescate desde mineplanner

| Pieza | Origen | Destino en fluidnet | Cambio |
|---|---|---|---|
| `GenericNetwork` ABC | `header_network.py` | `Network` | quitar física residual (`as_pressure`, unidades) hacia Rate/Fluid/Result |
| `propagate_rates`/`propagate_heads` | ídem | solver `forward_propagation` | `sum()` → reduce compatible con Rate; sacar clip |
| Firma de `MultiphaseHeaderNetwork` | ídem (stub) | `IntegralLoss` protocol | de override escondido a protocolo público |
| `darcy_dp` + friction | `pressure_functions.py` | `losses/darcy.py` | `full_output` → decorador; propiedades desde `Fluid`, no desde la red |
| `density` como atributo de red | `header_network.py` | `fluid.py` (`Fluid`/`FluidState`) | de escalar guardado a propiedad derivada de `(composición, P, T)` |
| `EdgeParameter` / optimizer declarativo | `run_fit_network.py` + optimizer | solver `fitting` | desacoplar de pandas-pipeline de mineplanner |
| `graph_*_to_frame`, `to_frame` | `header_network.py` | `Result.to_frames` / `Network.to_frames` | directo |
| Broadcasting pd.Series | `propagate_heads` | núcleo de vectorización | formalizar: escalar y Series como casos del mismo tipo |

---

## 4. Alcance v0.2 (para no derrapar)

**In**: DAG single-sink, `ScalarRate`, `Fluid`/`FluidState` con `IncompressibleFluid`, protocolos de loss declarados, solver 1 completo con Result, `constant_friction` + `darcy_weisbach`, tests, un notebook demo (red de wellfield sintética).

**Out (documentado como roadmap)**: ciclos/loops, mass_balance (solver 2), fitting (solver 3), multifásico concreto, BrineRate/saturation indices, fluido por nodo, temperatura como input del solver, unidades con pint.

Justificación: el solver 1 con Result completo es el mínimo que valida las capas a la vez y da un demo publicable. Solvers 2 y 3 son incrementos sobre la misma arquitectura, no rediseños — y lo mismo vale para el pasaje de fluido-de-red a fluido-de-nodo, que es un cambio de proveedor y no de protocolo.

---

## 5. Decisiones

**Cerradas (05/07/2026):**

1. **Unidades — SI estricto en el core.** Todo cálculo interno en SI, sin pint. Las unidades de presentación pueden vivir como atributo de red/Result para pre-proceso (conversión de inputs) y post-proceso (vistas, plots), pero nunca dentro de loss functions ni solvers. Los multiplicadores manuales de mineplanner (`rate_units`, `pressure_units`) desaparecen de las firmas de física.
2. **La loss function recibe el `Rate` entero**, no atributos sueltos. Es coherente con que Rate sea la abstracción central (sumable, multiplicable) y evita firmas que crecen con cada propiedad nueva. Para el caso trivial se proveen helpers/adaptadores que extraen los escalares.

**Cerradas (07/08/2026 — sesión de diseño `Rate`/`Fluid`):**

3. **`Rate` = extensivo + composición intensiva, sin propiedades de fluido** (§2.1). Reemplaza la formulación previa de este mismo documento.
4. **`Fluid` = fábrica stateless de `FluidState`, capa cero** (§2.1bis).
5. **`FluidState` con campos requeridos**; `temperature` en la firma con `None` = "no suministrado" (§2.1bis).
6. **El régimen `AlgebraicLoss`/`IntegralLoss` lo declara el `Fluid`** (§2.2) — cierra la decisión que este documento listaba como abierta.
7. **Fluido de red hasta v1.0 → de nodo en v1.5, mismo mecanismo**; el edge toma el fluido del nodo `upstream` (§2.1bis).

**Abiertas (no bloquean la arquitectura):**

8. **Mecanismo del canal de diagnósticos** (contextvar vs. objeto colector pasado por el solver) — a resolver en diseño detallado del decorador `@diagnostic`.
9. **MVP de 26 tests**: declarado perdido y superado (ver `ROADMAP.md`). Este draft es la spec de reconstrucción.
10. **Diseño del solver 3**: falta revisar `header_network_optimizer.py` y `network_observations.py` de mineplanner para afinar el port de `EdgeParameter`/observaciones (fuera de scope v0.2, no urge).

> Nota de higiene documental: `README_CLAUDE.md` establece que los ADR no llevan sección "Abiertas" (el backlog de diseño vive en `ROADMAP.md`). Esta sección se mantiene por continuidad histórica, pero las entradas 8–10 están duplicadas en `ROADMAP.md § Decisiones abiertas`, que es la fuente de verdad. Candidata a limpiarse en una pasada futura.

---

## 6. Próximos pasos

1. Cerrar la forma concreta de la composición trivial de `ScalarRate` y del adaptador `as_physics_kwargs()` — es lo único que queda entre el diseño cerrado y la implementación de la capa `Rate`/`Fluid`.
2. Diseño detallado del caso demo: red sintética de wellfield, inputs, qué muestra el notebook — esto define los tests de aceptación y tensiona las firmas contra un uso real. **Sin código todavía.**
3. Firmas públicas restantes: los dos `Protocol` de loss y `Result` — a nivel de interfaz (docstrings + type hints), no de implementación.
4. Recién ahí: implementar, portando en el orden del mapa de rescate (§3), commiteando de a piezas chicas contra `main` limpia.
