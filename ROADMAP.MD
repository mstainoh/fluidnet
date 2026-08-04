# fluidnet — Roadmap de trabajo (v0.2 → v1.0)

> Documento de planificación. Design-first: cada etapa cierra decisiones de
> arquitectura **antes** de escribir código. El objetivo del proyecto es
> visibilidad profesional (portfolio + eventual paper JOSS), así que la
> claridad del diseño y la reproducibilidad pesan más que la cantidad de features.

---

## Principios que ordenan el plan

1. **Diseño antes que código.** Cada milestone define firmas públicas
   (docstrings + type hints) y tests de aceptación antes de implementar.
2. **Cada etapa valida algo ante un observador externo.** MVP valida que la
   *arquitectura* funciona; v1 valida que es *útil*; v2 valida que es
   *diferencial*.
3. **Cinco capas, dependencias en un solo sentido.** `physics/` no importa nada
   del paquete; `Rate` y `loss_func` no conocen la red; `Network` no conoce la
   física; los solvers orquestan; `Result` lo produce solo el solver.
4. **Core SI estricto.** Unidades de presentación solo en la capa de I/O.
5. **Commits chicos contra `main` limpia**, en el orden del mapa de rescate
   desde mineplanner.

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
`GradientResult` en **Pa/m**. No conoce `Rate`, ni longitud de edge, ni
dirección de propagación, ni si su resultado se va a integrar o a
multiplicar por `L`. El régimen (`AlgebraicLoss` vs. `IntegralLoss`) es
decisión de `loss_func`, no algo que `physics` declare o infiera. Todo lo
que sea integración sobre la cañería, adaptación de firma o diagnósticos
pertenece a la capa `loss_func`, no acá.

**Freeze — con una distinción.** Se congelan las **correlaciones empíricas**
(B&B, modelos de holdup): no se agregan hasta v0.5. El diferencial del proyecto
es red + propagación + fitting; en correlaciones no hay diferencial (`fluids` ya
cubre ese terreno) y no aportan visibilidad. Las **ecuaciones puras**
(`dimensionless`, `friction`, `single_phase`) sí pueden extenderse, pero solo
cuando un solver o una `loss_func` lo requiera — nunca por completitud física.

---

## MVP — v0.2 · "La arquitectura funciona end-to-end"

**Qué prueba:** que un ingeniero senior lea el repo en 20 min y confirme que las
capas (`physics` / `Rate` / `loss_func` / `Network` / solver → `Result`) encajan.
No es un producto todavía; es la prueba estructural.

**Scope IN**

- `Rate` (ABC/Protocol) + `ScalarRate` con álgebra: `__add__`, `__radd__`
  (para que `sum()` no rompa el polimorfismo), `__mul__`.
  → *firma pendiente de cerrar; ver "Decisiones abiertas".*
- Propiedades de fluido (densidad, viscosidad) accesibles **desde el `Rate`**,
  no como atributo de red.
- Protocolo `AlgebraicLoss` implementado. `IntegralLoss` **declarado como
  interfaz documentada** pero sin implementar (se completa en v2).
- `Network`: wrapper `nx.DiGraph`, validación DAG, `propagate_rates` /
  `propagate_heads`, `to_frames()`. Sin clases Node/Edge.
- Solver 1: `forward_propagation` — single sink, BC = rates en sources + P en sink.
- `Result` (dataclass frozen): terna garantizada `(rate, p_up, p_down)` por
  edge + `(rate, p)` por nodo + metadata del solve; `to_frames()`.
- Built-ins de loss: `constant_friction` (baseline para tests/docs) +
  `darcy_weisbach`, **construido sobre `physics/single_phase.py`** (no
  reimplementa física). La capa de loss aporta lo que `physics` no hace:
  1. adaptar `Rate` → escalares SI,
  2. **integrar el gradiente sobre la longitud del edge** (`Pa/m` → `Pa`;
     trivial para incompresible: `grad.total * L`),
  3. aplicar `@diagnostic` (f, Re, v al canal lateral).
  `darcy_weisbach` cubre fluidos incompresibles en v0.2 — `compressibility=0`
  es la *consecuencia* de ese alcance, no una decisión de configuración que
  `darcy_weisbach` tome. Con compresibilidad != 0 el término de momento
  depende de P y la física deja de ser algebraica (pasa a `IntegralLoss`,
  v2). El mismo `physics` sirve a los dos protocolos sin cambios.
- Tests del core (los de physics, ~20, ya son el piso) + un notebook demo:
  wellfield sintético.
- README con statement of need explícito vs. pandapipes / WNTR.

**Scope OUT** (documentado como roadmap): fitting, mass_balance, multifásico
acoplado, broadcasting `pd.Series`, ciclos, `BrineRate`.

**Criterio de salida:** notebook corre punta a punta · CI verde · README
posicionado · tag `v0.2`.

---

## v1 — ≈ v0.5 · "Es útil para un caso real"

**Qué prueba:** que alguien con un problema real de wellfield/piping lo pueda
usar. Es el punto de submit a JOSS.

> **Nota de orden:** adelanto el solver de **fitting** (3) por delante del de
> **mass_balance** (2), invirtiendo la numeración. Razón: el fitting es el
> diferencial que ninguna librería open source cubre y lo que hace el paper
> defendible; mass_balance es más genérico y pandapipes ya lo resuelve para su
> dominio.

**Scope IN**

- Solver 3: **fitting** — `EdgeParameter` declarativo (qué parámetro, en qué
  edges, bounds) + optimizer que reutiliza el solver 1 vectorizado como forward.
- **Broadcasting `pd.Series`**: múltiples escenarios/timestamps en una pasada.
  Prerequisito del fitting y diferencial vs. pandapipes.
- Vectorización de `_beggs_brill_detailed` (hoy escalar). El
  `xfail(strict=True)` existente es la spec: cuando pase, sacar el marker y
  pinnear valores reales.
- Decorador `@diagnostic` resuelto → `Result` con `f`, `Re`, `v` por edge
  cuando el solver corre con `full_output=True`.
- BC generalizadas **2-de-3** `(Q, P_up, P_down)` por nodo/edge, con validación
  explícita de resolubilidad (falla claro, nunca adivina).
- **Fin del freeze de correlaciones**: modelos de holdup no-slip (`Hl = Cl`) y
  constant-slip, como baseline y puente pedagógico hacia B&B. *Diferidos desde
  `physics-single-multiphase.md` §5 — no eran MVP.*
- Docs formales: docstrings numpy · Sphinx/mkdocs en GitHub Pages ·
  2-3 ejemplos reproducibles (uno de calibración contra observaciones sintéticas).
- Empaquetado PyPI real + CHANGELOG + versionado semántico.

**Criterio de salida:** `pip install fluidnet` funciona · docs online · caso de
fitting demostrado · submit a JOSS.

---

## v2 — ≈ v1.0 · "Nadie más hace esto"

**Qué prueba:** consolida el diferencial completo del landscape, idealmente ya
con feedback de usuarios (issues del paper).

**Scope IN**

- Solver 2: **mass_balance** — BC arbitrarias mezcladas (head/rate) vía
  `scipy.optimize`. Requiere definir la magnitud de balance escalar de `Rate`.
- `IntegralLoss` implementado: integración de `dp/dL` edge a edge sobre el mismo
  esqueleto del solver 1 (pasando `p_boundary`). Cubre dos casos con el mismo
  mecanismo: monofásico compresible (`compressibility != 0`, física ya
  disponible) y multifásico (B&B, ya disponible).
- Extensiones de ejemplo, **fuera del core**: `BrineRate` (composición iónica →
  saturation indices, mezcla de salmueras), `MultiphaseRate` (fracciones).
- Trayectoria 3D como atributo de edge → perfil `P(x)` en `Result`
  (la integración por tramos ya la habilita `IntegralLoss`).
- Ciclos/loops en la red (levanta la restricción DAG para el solver 2).
- Unidades de presentación en `Result` (pre/post-proceso; core sigue SI estricto).

**Criterio de salida:** 3 solvers operativos · un caso multifásico documentado ·
tag `v1.0`.

---

## Secuencia inmediata (dentro del MVP, en orden)

1. **Cerrar la firma de `Rate`/`ScalarRate`** — es el bloqueante de todo: sin
   ella no hay `loss_func`, ni `Network`, ni `Result`. Incluye resolver la
   unidad de `sigma` (dyn/cm → N/m) y quién es dueño del modelo de propiedades.
2. **Diseño del caso demo** — red sintética de wellfield: inputs, outputs, qué
   muestra el notebook. Define los tests de aceptación y tensiona la firma de
   `Rate` contra un uso real. Sin código.
3. **Firmas públicas restantes** — los dos `Protocol` de loss y `Result`, a
   nivel de interfaz (docstrings + type hints), no implementación.
4. **Mecanismo de `@diagnostic`** (contextvar vs. colector) — bloquea
   `darcy_weisbach`, que es built-in de MVP.
5. **Implementar** por piezas chicas siguiendo el mapa de rescate del ADR §3,
   commiteando de a poco contra `main` limpia.

---

## Decisiones

### Cerradas

- **`physics/` es capa cero.** Funciones puras SI→SI, sin estado ni conocimiento
  de red. No importa nada del paquete; `loss_func` consume `physics`, nunca al
  revés. Contrato de salida: `GradientResult` en Pa/m (gradiente puntual, no dp).
- **Freeze de correlaciones empíricas hasta v0.5.** Las ecuaciones puras se
  extienden solo por necesidad de un solver.
- **La integración vive en `loss_func`, no en `physics`.** `physics` da `Pa/m`;
  convertir a `Pa` sobre la longitud del edge es responsabilidad de la loss. Es
  lo que separa `AlgebraicLoss` (integral trivial) de `IntegralLoss` (integral
  real con `p_boundary`).
- **`@diagnostic` sobre `full_output` en la firma.** Mantiene el contrato público
  estricto `loss_func(...) -> float`: el usuario casual escribe una loss en 5
  líneas sin pensar en diagnósticos, y la inspección queda como canal lateral
  opt-in que **recolecta el solver**. La info de eje se accede vía `Result` con
  `full_output=True`.
- **Dos protocolos de loss, no uno.** `AlgebraicLoss` (dp independiente de la P
  absoluta) e `IntegralLoss` (dp = integral de dp/dL, necesita `p_boundary`). El
  solver forward soporta ambas con el mismo esqueleto.
- **La loss recibe el `Rate` entero**, no atributos sueltos. Helpers extraen
  escalares para el caso trivial.
- **SI estricto en el core.** Unidades de presentación solo en capa de I/O.
- **MVP de 26 tests: declarado perdido y superado.** No se busca más. La spec de
  reconstrucción es `docs/design/architecture-v0.2.md` + este roadmap; los ~20
  tests de physics son el piso real.
- **Firma de `physics/`: keyword-only completo** (ver `CLAUDE.md` decisión
  cerrada #10). Toda función de gradiente es enteramente kw-only, sin
  excepción por cantidad de rates — necesario para que `loss_func` pueda
  despachar genéricamente (`gradient_fn(**kwargs)` filtrado por
  `signature`) sin ramificar por modelo.
- **Caudal negativo / dirección de flujo: no es feature de `physics`.**
  `beggs_brill_gradient`/`single_phase_gradient` no interpretan signo de
  caudal como sentido de flujo; la dirección la resuelve el integrador
  aguas arriba invirtiendo el sentido de integración (`solve_ivp` con
  `t_span=(L, 0)` en vez de `(0, L)`), no un caso especial dentro de
  `physics` ni de `loss_func`. Cierra la ambigüedad que dejaba abierta la
  rama de "flujo reverso" de `_beggs_brill_detailed` (ver
  `physics-single-multiphase.md` §2).

### Abiertas

- **Discriminador `AlgebraicLoss`/`IntegralLoss` no inspeccionable desde la
  firma.** Es un valor de runtime (`compressibility == 0`), no una
  propiedad estática de la firma de `loss_func` — contradice la lectura
  inicial de "despacho por `signature`" (ver `architecture-v0.2.md` §2.2,
  que sí aplica para el filtrado de kwargs, pero no para esto). Abierto: si
  el régimen lo declara el usuario al registrar la loss (coherente con
  "ningún solver adivina") o si se chequea en runtime.
- **Firma de `Rate`/`ScalarRate`** — bloqueante del MVP (ver secuencia §1).
  Sub-decisiones: alcance del estado (cinemática sola vs. cinemática +
  propiedades vs. cinemática + composición con propiedades derivadas); quién es
  dueño del modelo de propiedades (`Rate` o red); unidad de `sigma`.
- Mecanismo del canal de `@diagnostic` (contextvar vs. objeto colector) —
  bloquea `darcy_weisbach`.
- Ubicación de `GradientResult`: hoy vive en `single_phase.py` pero es el
  contrato compartido con `multiphase.py`. Candidato a `physics/types.py` para
  que la dependencia entre módulos de physics no sea lateral.
- Port fino del solver 3: revisar `header_network_optimizer.py` y
  `network_observations.py` de mineplanner (fuera de scope v0.2).