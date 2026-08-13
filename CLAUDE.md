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
5. **`FluidState`: `NamedTuple` de campos requeridos, nunca `float | None`,
   construido por llamada.** `β = 0` es el valor físico exacto del agua, no
   una ausencia. Un `sigma=None` filtrándose a B&B es el mismo problema que
   un default físico invisible, y rompe el filtrado de kwargs por
   `signature`.
   **Forma cerrada (2026-08-10)**: subclases `SinglePhaseState` (nombres
   pelados) y `MultiPhaseState` (sufijos de fase, #19), con
   `as_physics_kwargs()` **del lado del estado**.
   Rationale de por qué el parseo es del estado y no de `loss_func`: el
   número de fases es propiedad de la **implementación del `StateModel`**,
   no del valor — un flash puede cruzar el punto de burbuja a lo largo de
   `x`, pero un modelo multifásico igual emite ambas fases (una con fracción
   cero). La clase del estado es fija por modelo, así que el parseo es
   estático.
   Se construye una vez por evaluación del gradiente: tiene que ser barato.
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
   **Alcance acotado (2026-08-09)**: esto vale para `Fluid`, **no** para el
   `Protocol` neutro `StateModel` (#18). `T` no asciende al protocolo: lo
   contamina — en el demo eléctrico AC no significa nada — y puede complicar
   la existencia de la solución.
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
18. **`StateModel`: `Protocol` neutro de dos métodos; `Fluid` lo implementa.**
    Transforma la variable *across* del nodo (más el estado propagado) en los
    argumentos del `gradient_fn`. Fluidos: `(comp, T, P) → (ρ, μ, β, σ)`.
    Eléctrico AC: `(V) → impedancia`.
    **Nombre cerrado (2026-08-10)**: `StateModel`. Descartados `Medium` (sólo
    tiene sentido en continuos) y `ConstitutiveModel` (colisiona con
    `LossFunc`, que *es* la constitutiva).

    ```
    StateModel.bind(**fields: object) -> BoundState             # 1× por eje
    BoundState.__call__(*, x, across) -> State                  # 1× por paso
    ```

    - **`composition` sale del `Protocol` (corrección 2026-08-13).** No todo
      `StateModel` tiene composición — un modelo de agua dulce se parametriza
      solo por `T`. Forzarla en la firma neutra es el mismo error que forzar
      `temperature` (ver más abajo): la composición pasa a ser un campo más,
      declarado por la implementación concreta (`Fluid.bind(*, composition,
      temperature=None)`), que es donde hay información para tipar.
    - **La distinción propagado/prescrito se muda al solver (2026-08-13).**
      El `Protocol` ya no la codifica; la codifica quien arma los kwargs de
      `bind`: `model.bind(**propagated_fields(rate),
      **prescribed_fields(G, u, v))`. Sigue siendo la distinción que sostiene
      el límite declarado en #27 (los prescritos no se balancean en nodos; la
      composición viaja por la topología) — documentada en el ADR, no tipada.
    - **La cadena anidada `comp → T → P` es binding time**, no nesting
      sintáctico: el orden es por frecuencia de cambio. Fijadas composición y
      campos prescritos, la parcial que queda es sólo función de `across` —
      que es lo que hace prolija la ODE (una única variable independiente en
      el integrando).
    - **`bind` es aplicación parcial, no construcción.** Motivo de ownership,
      no de performance: la composición sale de `propagate_rates` (runtime), y
      si entrara por `__init__` el `StateModel` dejaría de poder ser atributo
      declarativo de red/nodo (#7).
    - **`x` está en la firma; los campos físicos no.** `x` es domain-neutral y
      ya vive en el vocabulario del integrador. `temperature` en `__call__`
      reabriría esta misma decisión; `**fields` en `__call__` muere con `mypy
      strict` y obliga a `loss_func` a traducir en vez de componer.
    - **`temperature` NO asciende al `Protocol`**: se liga en `bind`, cuya
      firma la declara la implementación concreta (`Fluid.bind(composition,
      temperature=None)`). Consistente con #6.
    - Costo aceptado: `x` es un argumento inerte en la firma pública hasta v2.
      Se documenta en el docstring.
19. **Convención de sufijos de fase. El `StateModel` entrega propiedades por
    fase; toda propiedad de mezcla es cómputo de `physics`.**
    `density_gas` / `density_liquid` / `compressibility_gas` /
    `compressibility_liquid`; monofásico usa el nombre pelado
    (`density`, `compressibility`).
    Razón: la compresibilidad de mezcla se pondera por holdup, y el holdup lo
    calcula la correlación — el `StateModel` no puede entregarla porque no
    conoce el holdup.
    **Esto revierte la unificación `mix_compressibility → compressibility`**
    del commit `7142191` para el caso multifásico. Aquella decisión resolvía
    un problema real (emitir una sola clave sin ramificar por modelo) pero
    elige el nombre equivocado bajo esta regla. El filtrado por `signature`
    sigue funcionando sin tabla: cada gradiente declara qué necesita.
20. **Contrato de `LossFunc`: dos métodos.**
    `solve_dp(rate, state, **edge_attrs) -> dp` abstracto;
    `solve_rate(dp, state, **edge_attrs) -> rate` en el protocolo integral con
    `NotImplementedError` por default (habilita resolver la red sobre campo
    de potenciales — no es el régimen de fluidos, sí el de otros dominios).
    Las direcciones `Q→P1→P2` y `Q→P2→P1` **no son modos distintos**: es el
    signo del `t_span` (decisión 2026-08-04). Ejes ortogonales reales:
    régimen algebraico/integral × cuál es la incógnita.
    `AlgebraicLoss` es el **caso degenerado** de `IntegralLoss` (sin `P` en el
    integrando, la integral colapsa a `grad × L`); se mantienen los dos
    protocolos porque no tiene sentido formular una ODE cuando la solución
    analítica existe y es simple de codificar.
21. **Vocabulario canónico, no tabla de renombres.** Los nombres de `physics`
    son contrato. `loss_func` arma `{**rate_kwargs, **state_kwargs,
    **edge_attrs}` y filtra por `signature(gradient_fn)`. Dict de override
    **sólo** ante una razón genuina de desviarse: un dict identidad es
    ritual, y una tabla que crece con cada modelo se desincroniza en silencio
    (el kwarg no pasa, entra un default, revienta lejos del origen). Contrato
    de entrega: nombre canónico + valor en SI. **`loss_func` compone, no
    traduce**: la aridad multifásica es cómputo (`Rate` da el extensivo,
    `StateModel` la fracción, el producto da los rates por fase).
22. **`Rate` se define por contrato, no por contenido**: *puedo sumarme con
    otros y dar balance cero, y puedo meterme en una loss func.*
    - `as_physics_kwargs()` **sí existe** en `Rate` — lo extensivo va a la
      loss function siempre. Lo que vive en `loss_func` es la composición con
      la fracción de fase, no el acceso al extensivo.
    - **La composición es atributo de subclase, no de clase.** `ScalarRate`
      no tiene el campo; no lo tiene vacío. Uniformidad en el método, no en
      el dato: un singleton `{"fluid": 1.0}` obligaría a iterar dict +
      división ponderada por nodo en el loop interno del optimizer de v0.5, y
      no vectoriza.
    - **`as_physics_kwargs` se arma una vez por eje, fuera del `solve_ivp`.**
      Lo licencia que en steady-state el mass rate sea constante a lo largo
      del eje; el RHS sólo actualiza `pressure`.
    - **No existe `MultiphaseRate` con fracciones.** El split líquido/gas
      depende de un flash `(P, T, comp)` y cambia a lo largo del caño: es
      trabajo de EOS. Un `StateModel` de fracción impuesta es válido como
      hipótesis explícita de modelado, pero es un fluido, no un rate.
23. **Conversión de unidades fuera del contrato de `LossFunc`** — atributo de
    `Network`, capa de I/O. Si `loss_func` convierte, la decisión #1 (SI
    estricto) se vuelve nominal, y es de las que un reviewer chequea leyendo
    una firma.
24. **El canal `@diagnostic` es post-proceso, no interceptor.** El
    diagnóstico se evalúa sobre la solución **ya convergida** (el `P(x)` que
    devolvió el integrador), no durante el solve. Consecuencias:
    - Los pasos rechazados de `solve_ivp` quedan fuera por construcción,
      no por filtrado.
    - La grilla de evaluación es **declarada** (extremos, `sol.t`, o k
      puntos), nunca la que eligió el integrador. No hay que reducir N
      evaluaciones arbitrarias a un valor de eje: el problema se disuelve.
    - Overhead exacto cero durante el fitting: los iterados intermedios del
      optimizer nunca piden diagnóstico.
    - El caso algebraico es el degenerado (grilla de un punto), mismo
      mecanismo, sin rama especial — igual que `AlgebraicLoss` respecto de
      `IntegralLoss`.
    Vive en `loss_func`. **`physics/` no se toca**: sigue siendo evaluador
    puntual y directionally blind.
25. **Dos niveles de diagnóstico. `GradientResult` es el nivel 0.**
    La descomposición `(gravity, friction, momentum)` es información
    diagnóstica real, existe siempre y es gratis — toda función de gradiente
    la provee. El nivel 1 (intermedios de correlación: holdup, régimen, `f`,
    `Re`) es **opt-in** vía una función `detailed` hermana.
    - `GradientResult` **se queda como `NamedTuple`**. Sin campo
      `extra: dict[str, Any]` (violaría #11 una capa más abajo y rompe
      `--strict`), sin `__array__` (colapsaría la descomposición en silencio;
      la coerción a escalar se escribe `.total`).
    - **Sin polimorfismo por flag.** Nada de `-> GradientResult | dict`.
      Una implementación `_detailed`, un wrapper público que extrae. Overhead
      medido del dict: ~19% en el peor caso (escalar, correlación barata), y
      se divide por N al vectorizar en v0.5 — un dict por llamada de array,
      no por elemento.
    - **El par se declara explícito**, no por convención de nombre: la loss
      recibe `detailed_fn` (default `None`), no introspecciona `_*_detailed`.
      Una correlación de terceros no está obligada a exponer intermedios;
      sin `detailed_fn` se entrega el nivel 0.
    - `diagnose` es el tercer método del protocolo `LossFunc`, con el mismo
      tratamiento que `solve_rate`: `NotImplementedError` por default.
    **Corolario de secuencia: `@diagnostic` ya no bloquea `darcy_weisbach`.**
    Se escribe hoy sin ninguna conciencia de diagnóstico y el mecanismo se
    le suma en v0.5 sin tocarle el cuerpo.
26. **Dos clases hermanas de `BoundState`, discriminadas en `bind` por
    `callable(field)`.** El corte no es "campo fijo vs. variable" sino **si el
    objeto ligado necesita `x`**: `None` y un escalar son el mismo caso (`x` se
    descarta); un callable es el otro. La rama se decide una vez, fuera del
    loop. Evita que `None` viaje disfrazado de función por la capa más caliente.
27. **Los campos prescritos son datos, no cantidades conservadas.** No se impone
    balance sobre ellos en los nodos. La consistencia de un perfil de `T` entre
    ramas que mezclan es hipótesis explícita del usuario. Va declarado en README.
28. **Regla de ligado**: *todo lo que no depende de `x` se liga antes del
    integrador; lo que depende de `x` entra por el `bound`.* Unifica #18, el
    hoisting de `rate.as_physics_kwargs()` (#21) y #26.
29. **Vocabulario: `gradient_fn` ≠ `loss_func`.** `gradient_fn` es la función
    pura de `physics/` (kwargs planos en SI → derivada, directionally blind).
    `loss_func` es la capa (`AlgebraicLoss`/`IntegralLoss`) que compone
    `rate` + `bound` + `edge_attrs`, arma el `rhs` e integra. No usar
    `loss_func` para referirse al gradiente: vuelve ilegible el rationale de #18.
    Corolario: `as_physics_kwargs()` es **un contrato con varios dueños**
    (`Rate`, `State`, y en v1.5 los atributos de eje calculados), no un método
    repetido por casualidad.
30. **`across` en el `Protocol`, `pressure` de la implementación concreta para
    abajo.** El `Protocol` es neutro por diseño (#18): `pressure` ahí es la
    misma evidencia en contra del diferencial physics-agnostic que motivó el
    nombre neutro. Hacia abajo, `pressure` ya está en `physics/`, es el
    vocabulario del dominio, y renombrarlo tocaría capa cero testeada.
    Traducción en el `__call__` del `BoundState`, una línea.
    - **`across: ArrayLike`, no `float` (corrección 2026-08-13).** `solve_ivp`
      pasa `y` como `ndarray` siempre, incluso para una ODE escalar (`shape
      (1,)`): tipar `float` documenta un desempaquetado, no un contrato.
      Ensanchar una entrada es gratis — un `float` sigue siendo un
      `ArrayLike` válido y ningún caller se rompe; distinto de ensanchar una
      *salida*, que es la decisión que sigue abierta (ver excepción de
      `FluidState`/`GradientResult` en "Tipado"). `x` queda `float`: es la
      coordenada del integrador, siempre escalar. Costo declarado en
      docstring: las implementaciones deben ser array-safe; en v0.2 se
      documenta, no se testea.
    - **`State.as_physics_kwargs() -> dict[str, ArrayLike]` (corrección
      2026-08-13).** Su consumidor es `gradient_fn`, que ya acepta
      `ArrayLike` (ver `friction_factor(re: ArrayLike, ...)`). Tiparlo
      `float` angosta contra un consumidor que no lo pide — la firma de
      salida de la frontera debe coincidir con la de entrada de `physics/`.
      Esto **no** reabre la decisión abierta sobre los campos de
      `FluidState`/`GradientResult` (siguen `float` en v0.2): un `float` es
      un `ArrayLike`, así que la firma del método puede ser ancha sin tocar
      los campos almacenados. Son dos decisiones distintas.
    Firmas: `StateModel.bind(**fields: object) -> BoundState`;
    `BoundState.__call__(*, x: float, across: ArrayLike) -> State`;
    `State.as_physics_kwargs() -> dict[str, ArrayLike]`;
    `Fluid.bind(*, composition, temperature=None)`;
    `Fluid.get_state(*, composition, temperature, pressure)`.

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
  abierta. **No confundir con la firma de `State.as_physics_kwargs() ->
  dict[str, ArrayLike]`, que es otra decisión y ya está cerrada** (#30,
  2026-08-13): un campo `float` almacenado sigue siendo un `ArrayLike`
  válido en el dict de salida, así que la firma del método no obliga a
  tocar los campos.
- `GradientResult`: no lo conviertas a `dataclass`, no le agregues campos, no
  le agregues `__array__`. Es contrato de capa cero (decisión #25). Si un
  error de mypy parece pedir eso, pará y preguntá.
  
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
