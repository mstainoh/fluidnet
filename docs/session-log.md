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
