# fluidnet — Draft 1: Arquitectura v0.2
 
> Sucede al Draft 0. Incorpora: (a) verificación del landscape con pandapipes incluido, (b) el código real de mineplanner (`header_network.py`, `pressure_functions.py`, `run_fit_network.py`) como referencia de diseño probado en producción, (c) las decisiones de diseño acordadas: result container vía decorador, sin clases Node/Edge, tipado estricto.
 
---
 
## 1. Posicionamiento
 
**Qué es**: librería Python graph-first para simulación steady-state de redes de fluido. El usuario trae su propia física (Rate polimórfico + loss functions pluggables); la librería aporta topología, propagación, solvers y calibración.
 
**Objetivo del proyecto**: visibilidad profesional (portfolio GitHub) → publicación en **JOSS** (Journal of Open Source Software) o SoftwareX como antecedente académico. Esto condiciona el diseño: API limpia y documentada > features; tests y ejemplos reproducibles son parte del producto, no accesorios.
 
**Landscape verificado (07/2026)** — qué existe y por qué fluidnet no es redundante:
 
| Tool | Dominio | Limitación relevante |
|---|---|---|
| **pandapipes** (Fraunhofer) | gas / district heating, multi-energy grids | monofásico, un fluido por red, física cerrada, sin fitting de parámetros orientado a calibración de campo |
| **WNTR** (EPA) | agua potable, resiliencia | atado a EPANET, agua únicamente |
| **TESPy** | ciclos térmicos | componentes, no redes de tubería genéricas |
| **fluids** | correlaciones puntuales | sin capa de red |
| PIPESIM / PIPEPHASE / OLGA | gathering multifásico O&G | comerciales, cerrados |
 
**Diferencial de fluidnet** (statement of need para JOSS): ninguna librería open source combina (1) rate polimórfico que propaga composición por la red (mezcla de salmueras → saturation indices, multifásico), (2) loss functions pluggables con protocolo mínimo, (3) régimen forward-propagation como caso de primera clase (estándar en gathering O&G/minería, que pandapipes/WNTR tratan solo vía balance general), (4) fitting de parámetros vectorizado contra observaciones de campo. El README debe posicionarse explícitamente contra pandapipes/WNTR: es la primera pregunta de cualquier reviewer.
 
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
├──────────────────────┴──────────────────────────┤
│  Result (contrato de salida)                    │
└─────────────────────────────────────────────────┘
```
 
Regla de dependencias: `Rate` y `loss_func` no conocen la red. `Network` no conoce la física (solo plumbing de grafo). Los solvers orquestan. `Result` es producido exclusivamente por solvers.
 
### 2.1 Rate — la abstracción central
 
Dataclass (frozen) con álgebra:
 
- `__add__` → mezcla en nodos de convergencia. Para `ScalarRate` es suma; para rates composicionales es mezcla ponderada por caudal (composición, salinidad, fracciones de fase).
- `__mul__` (escalar) → scaling, necesario para el optimizer y para escenarios vectorizados.
- Propiedades del fluido (densidad, viscosidad) **viven en el Rate, no en la red** — corrige la deuda de mineplanner donde `density` es atributo de red. Esto es lo que habilita mezcla de salmueras: dos rates con densidades distintas se suman y el resultado tiene la densidad de mezcla.
Jerarquía mínima v0.2: `Rate` (protocolo/ABC) → `ScalarRate` (monofásico, cubre agua/salmuera simple). El usuario extiende sin tocar el core: `BrineRate` (composición iónica → saturation indices), `MultiphaseRate` (fracciones), quedan como ejemplos/extensiones, no core.
 
**Decisión heredada de mineplanner a corregir**: `propagate_rates` usa `sum(propagated[p] for p in predecessors)` — `sum()` arranca de `0` (int). `Rate` debe soportar `0 + rate` (`__radd__`) o la propagación usa `functools.reduce`. Detalle chico, pero es exactamente el bug que rompe el polimorfismo.
 
### 2.2 loss_func — dos protocolos, no uno
 
El código de mineplanner ya reveló que hay dos firmas distintas según el régimen físico:
 
**Protocolo algebraico** (monofásico, incompresible — el dp no depende de la presión absoluta):
 
```
loss_func(rate: Rate, **edge_attrs, **network_attrs) -> float
```
 
**Protocolo de integración** (multifásico / compresible — el dp depende de p local, se integra dp/dL desde la frontera conocida):
 
```
loss_func(rate: Rate, p_boundary: float, **edge_attrs, **network_attrs) -> float
```
 
Ambos declarados como `typing.Protocol` explícitos (`AlgebraicLoss`, `IntegralLoss`). Convención de signos heredada y confirmada: `dp = p_downstream − p_upstream` (pérdida → negativo). Kwargs extra absorbidos con `**kwargs` — el patrón de mineplanner de "declarar solo lo que usás" funciona bien y se mantiene.

Consecuencia directa de que `physics/` sea enteramente keyword-only (`CLAUDE.md` decisión #10): `loss_func` puede despachar hacia la función de gradiente correcta filtrando sus propios kwargs por `inspect.signature(gradient_fn).parameters` en vez de necesitar un adaptador por modelo. `physics` tampoco conoce la orientación del edge — es `loss_func` quien adapta signo/dirección (caudal, inclinación) antes de invocar la función de gradiente; la dirección de integración en sí la resuelve el solver (`t_span` de `solve_ivp`), no `physics` ni `loss_func`.

**El discriminador `AlgebraicLoss`/`IntegralLoss` no es inspeccionable desde la firma.** A diferencia del despacho de kwargs de arriba (que sí lee de `signature`), qué protocolo aplica es un valor de *runtime* — `compressibility == 0` (o `mix_compressibility == 0` en B&B) — no una propiedad estática de la firma de `loss_func` ni de `gradient_fn`. **Abierto**: si el régimen lo declara el usuario explícitamente al registrar la loss (coherente con "ningún solver adivina", ver §2.4) o si se chequea en runtime a partir del valor de compresibilidad.
 
**Diagnósticos vía decorador** — la solución al conflicto "contrato escalar estricto vs. resultado rico":
 
- El protocolo exige `-> float`. Punto. Tipado estricto, usuario casual escribe 5 líneas.
- Un decorador `@diagnostic` permite a la función registrar intermedios (f, Re, v, régimen de flujo, perfil P(x)) en un canal lateral que **el solver** recolecta cuando se pide `full_output=True`. La loss function nunca devuelve un dict al solver.
- El `full_output: bool` de `darcy_dp` actual desaparece de la firma pública (queda como mecanismo interno del decorador).
Built-ins v0.2: `constant_friction` (baseline trivial, para tests y docs) y `darcy_weisbach` (portado de `pressure_functions.py`, que ya está en buen estado: Chen approx, régimen laminar/transición/turbulento, minor losses K, vectorizado numpy).
 
### 2.3 Network — plumbing sin física
 
Wrapper de `nx.DiGraph` con validación (DAG requerido en v0.2). **Sin clases Node/Edge**: los atributos viven en los dicts nativos de networkx (`G.nodes[n]`, `G.edges[u,v]`); las vistas tabulares las dan `to_frames()` (portado de `graph_edges_to_frame`/`graph_nodes_to_frame`).
 
Se rescata directamente de mineplanner (código ya probado en producción):
 
- `GenericNetwork` ABC: validación DAG, `new_with_settings`, `__add__` vía `nx.compose`, `copy`, subgraph helpers, `topological_sort_edges`.
- Separación `propagate_rates` / `propagate_heads` (topological sort forward para rates, reversed para heads).
- Broadcasting vectorizado: rates como `pd.Series` (múltiples escenarios/timestamps en una pasada) — esto es un diferencial real vs. pandapipes y clave para el fitting.
Se corrige:
 
- Condiciones de borde: hoy son `sink_head` (escalar, single sink) + rates en sources. v0.2 generaliza a **especificación 2-de-3 por nodo/edge** `(Q, P_up, P_down)` — pero el solver forward valida que la especificación dada sea resoluble por propagación y falla explícito si no (nunca adivina).
- `clip_min_head` sale del path de propagación (esconde errores físicos); si se mantiene, es opt-in con warning.
### 2.4 Solvers — tres regímenes explícitos
 
El usuario elige el régimen; ningún solver "adivina":
 
1. **`forward_propagation`** — red convergente (DAG, single sink en v0.2), BC = rates en sources + presión en sink. Caudales por suma directa, presiones por back-propagation. **Punto de partida y caso demo.** Con `IntegralLoss` el mismo esqueleto cubre multifásico (pasa `p_boundary` edge a edge).
2. **`mass_balance`** — combinaciones arbitrarias de BC (head/rate mezclados), resuelto con `scipy.optimize` (fsolve/root). Requiere que el residual sea escalar → define qué debe exponer `Rate` (una magnitud de balance, e.g. caudal másico total).
3. **`fitting`** — calibración de parámetros de edge contra observaciones. Se porta el diseño declarativo de mineplanner: `EdgeParameter` (qué parámetro, en qué edges, bounds) + optimizer que reutiliza el solver 1 vectorizado como forward model. El runner (`run_fit_network.py`) demuestra el patrón end-to-end: observaciones long-format → pivots → fit paralelo por header → residuales base vs. fitted.
### 2.5 Result — el contrato de salida
 
Producido por el solver, nunca por loss functions. Dataclass frozen:
 
- **Garantizado siempre**: la terna completa `(rate, p_up, p_down)` por edge; `(rate, p)` por nodo; metadata del solve (régimen, convergencia, iteraciones).
- **Opcional / lazy**: diagnósticos del decorador (f, Re, v por edge), perfil `P(x)` si el edge tiene planimetría (trayectoria 3D como atributo de edge), propiedades derivadas del Rate en cada nodo (composición de mezcla, saturation indices vía extensión).
- **Vistas**: `.to_frames() -> (df_edges, df_nodes)`, `.to_graph() -> nx.DiGraph` anotado (patrón de `get_propagated_network` actual, útil para plotting).
`Result` es también la interfaz de consumo externo (mineplanner u otro pipeline): un objeto serializable con vistas pandas, sin exigir que el consumidor conozca networkx.
 
---
 
## 3. Mapa de rescate desde mineplanner
 
| Pieza | Origen | Destino en fluidnet | Cambio |
|---|---|---|---|
| `GenericNetwork` ABC | `header_network.py` | `Network` | quitar física residual (`as_pressure`, unidades) hacia Rate/Result |
| `propagate_rates`/`propagate_heads` | ídem | solver `forward_propagation` | `sum()` → reduce compatible con Rate; sacar clip |
| Firma de `MultiphaseHeaderNetwork` | ídem (stub) | `IntegralLoss` protocol | de override escondido a protocolo público |
| `darcy_dp` + friction | `pressure_functions.py` | `losses/darcy.py` | `full_output` → decorador; density desde Rate |
| `EdgeParameter` / optimizer declarativo | `run_fit_network.py` + optimizer | solver `fitting` | desacoplar de pandas-pipeline de mineplanner |
| `graph_*_to_frame`, `to_frame` | `header_network.py` | `Result.to_frames` / `Network.to_frames` | directo |
| Broadcasting pd.Series | `propagate_heads` | núcleo de vectorización | formalizar: escalar y Series como casos del mismo tipo |
 
---
 
## 4. Alcance v0.2 (para no derrapar)
 
**In**: DAG single-sink, `ScalarRate`, protocolos de loss declarados, solver 1 completo con Result, `constant_friction` + `darcy_weisbach`, tests, un notebook demo (red de wellfield sintética).
**Out (documentado como roadmap)**: ciclos/loops, mass_balance (solver 2), fitting (solver 3), multifásico concreto, BrineRate/saturation indices, unidades con pint.
 
Justificación: el solver 1 con Result completo es el mínimo que valida las 4 capas a la vez y da un demo publicable. Solvers 2 y 3 son incrementos sobre la misma arquitectura, no rediseños.
 
---
 
## 5. Decisiones
 
**Cerradas (05/07/2026):**
 
1. **Unidades — SI estricto en el core.** Todo cálculo interno en SI, sin pint. Las unidades de presentación pueden vivir como atributo de red/Result para pre-proceso (conversión de inputs) y post-proceso (vistas, plots), pero nunca dentro de loss functions ni solvers. Los multiplicadores manuales de mineplanner (`rate_units`, `pressure_units`) desaparecen de las firmas de física.
2. **La loss function recibe el `Rate` entero**, no atributos sueltos. Es coherente con que Rate sea la abstracción central (sumable, multiplicable) y evita firmas que crecen con cada propiedad nueva. Para el caso trivial se proveen helpers/adaptadores que extraen los escalares.
**Abiertas (no bloquean la arquitectura):**
 
3. **Mecanismo del canal de diagnósticos** (contextvar vs. objeto colector pasado por el solver) — a resolver en diseño detallado del decorador `@diagnostic`.
4. **MVP de 26 tests**: sigue sin aparecer en el remoto. Si no está local, la reconstrucción es barata con este draft como spec.
5. **Diseño del solver 3**: falta revisar `header_network_optimizer.py` y `network_observations.py` de mineplanner para afinar el port de `EdgeParameter`/observaciones (fuera de scope v0.2, no urge).
---
 
## 6. Próximos pasos
 
1. Diseño detallado del caso demo (siguiente sesión): red sintética de wellfield, inputs, qué muestra el notebook — esto define los tests de aceptación. **Sin código todavía.**
2. Diseño detallado de firmas públicas: `Rate`/`ScalarRate`, los dos `Protocol` de loss, `Result` — a nivel de interfaz (docstrings + type hints), no de implementación.
3. Confirmar destino del MVP de 26 tests (decisión abierta 4).
4. Recién ahí: implementar, portando en el orden del mapa de rescate (§3), commiteando de a piezas chicas contra `main` limpia.
 
