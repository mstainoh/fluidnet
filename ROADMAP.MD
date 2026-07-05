ROADMAP.md

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
3. **Core SI estricto, sin física en la topología.** `Rate` y `loss_func` no
   conocen la red; `Network` no conoce la física; los solvers orquestan;
   `Result` lo produce solo el solver.
4. **Commits chicos contra `main` limpia**, en el orden del mapa de rescate
   desde mineplanner.
---
 
## MVP — v0.2 · "La arquitectura funciona end-to-end"
 
**Qué prueba:** que un ingeniero senior lea el repo en 20 min y confirme que
las 4 capas (`Rate` / `loss_func` / `Network` / solver → `Result`) encajan.
No es un producto todavía; es la prueba estructural.
 
**Scope IN**
- `Rate` (ABC/Protocol) + `ScalarRate` con álgebra: `__add__`, `__radd__`
  (para que `sum()` no rompa el polimorfismo), `__mul__`.
- Propiedades de fluido (densidad, viscosidad) **en el Rate**, no en la red.
- Protocolo `AlgebraicLoss` implementado. `IntegralLoss` **declarado como
  interfaz documentada** pero sin implementar (se completa en v2).
- `Network`: wrapper `nx.DiGraph`, validación DAG, `propagate_rates` /
  `propagate_heads`, `to_frames()`. Sin clases Node/Edge.
- Solver 1: `forward_propagation` — single sink, BC = rates en sources + P en sink.
- `Result` (dataclass frozen): terna garantizada `(rate, p_up, p_down)` por
  edge + `(rate, p)` por nodo + metadata del solve; `to_frames()`.
- Built-ins de loss: `constant_friction` (baseline para tests/docs) +
  `darcy_weisbach` (portado de `pressure_functions.py`).
- Tests (recuperar o reconstruir los 26) + un notebook demo: wellfield sintético.
- README con statement of need explícito vs. pandapipes / WNTR.
**Scope OUT** (documentado como roadmap): fitting, mass_balance, multifásico,
broadcasting pd.Series, ciclos, `BrineRate`.
 
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
> dominio. Si preferimos roadmap con numeración literal, se invierte — pero se
> pierde el punch del diferencial temprano.
 
**Scope IN**
- Solver 3: **fitting** — `EdgeParameter` declarativo (qué parámetro, en qué
  edges, bounds) + optimizer que reutiliza el solver 1 vectorizado como forward.
- **Broadcasting pd.Series**: múltiples escenarios/timestamps en una pasada.
  Prerequisito del fitting y diferencial vs. pandapipes.
- Decorador `@diagnostic` resuelto (ver decisión cerrada abajo) → `Result`
  con `f`, `Re`, `v` por edge cuando el solver corre con `full_output=True`.
- BC generalizadas **2-de-3** `(Q, P_up, P_down)` por nodo/edge, con
  validación explícita de resolubilidad (falla claro, nunca adivina).
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
- `IntegralLoss` implementado: multifásico/compresible, integración dp/dL edge
  a edge sobre el mismo esqueleto del solver 1 (pasando `p_boundary`).
- Extensiones de ejemplo, **fuera del core**: `BrineRate` (composición iónica →
  saturation indices, mezcla de salmueras), `MultiphaseRate` (fracciones).
- Trayectoria 3D como atributo de edge → perfil `P(x)` en `Result`.
- Ciclos/loops en la red (levanta la restricción DAG para el solver 2).
- Unidades de presentación en `Result` (pre/post-proceso; core sigue SI estricto).
**Criterio de salida:** 3 solvers operativos · un caso multifásico documentado ·
tag `v1.0`.
 
---
 
## Secuencia inmediata (dentro del MVP, en orden)
 
1. **Diseño del caso demo** — red sintética de wellfield: inputs, outputs, qué
   muestra el notebook. Esto **define los tests de aceptación**. Sin código.
2. **Firmas públicas** — `Rate` / `ScalarRate`, los dos `Protocol` de loss,
   `Result`: a nivel de interfaz (docstrings + type hints), no implementación.
3. **Confirmar destino del MVP de 26 tests** (decisión abierta 4). Si no está
   local, este roadmap + Draft 1 son la spec para reconstruirlo (barato).
4. **Implementar** por piezas chicas siguiendo el mapa de rescate del Draft 1 §3,
   commiteando de a poco contra `main` limpia.
---
 
## Decisiones
 
### Cerradas
- **`@diagnostic` sobre `full_output` en la firma.** El atractivo de `full_output`
  era exponer atributos de eje (f, Re, v, régimen) directamente. Se prefiere el
  decorador porque mantiene el contrato público estricto `loss_func(...) -> float`:
  el usuario casual escribe una loss en 5 líneas sin pensar en diagnósticos, y la
  inspección queda como canal lateral opt-in que **recolecta el solver**, no la
  loss function. No se pierde la info de eje — se accede vía `Result` con
  `full_output=True`.
- **Dos protocolos de loss, no uno.** `AlgebraicLoss` (dp independiente de la P
  absoluta) e `IntegralLoss` (dp = integral de dp/dL, necesita `p_boundary`).
  Firmas explícitas dejan claro qué física necesita cada tipo; el solver forward
  soporta ambas con el mismo esqueleto.
- **La loss recibe el `Rate` entero**, no atributos sueltos. Coherente con Rate
  como abstracción central; helpers extraen escalares para el caso trivial.
- **SI estricto en el core.** Unidades de presentación solo en capa de I/O.
### Abiertas (no bloquean la arquitectura)
- Mecanismo del canal de `@diagnostic` (contextvar vs. objeto colector) — a
  resolver en diseño detallado del decorador, antes de tocar código.
- Ubicación del MVP de 26 tests (¿local? ¿reconstruir?).
- Port fino del solver 3: revisar `header_network_optimizer.py` y
  `network_observations.py` de mineplanner (fuera de scope v0.2).
 
