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
   **Campos `ArrayLike`, no `float` (cerrado 2026-08-13).** Reemplaza la
   nota de "Tipado" que los daba como abiertos. `SinglePhaseFluidState`
   (`density`/`viscosity`/`compressibility`) pasa a `ArrayLike`, igual que
   `GradientResult` — que en realidad **nunca fue `float`**: es `ArrayLike`
   desde que se creó (`physics/types.py`, commit `9780d43`). No hubo que
   "mover" nada ahí; la única inconsistencia era la docs (ver ROADMAP
   Cerradas y el ADR `physics-single-multiphase.md` §1). `IncompressibleFluid`
   sigue construyéndose con `float` sin cambios — un `float` es un
   `ArrayLike` válido, no se ensancha nada por escalar. Nota técnica sin
   resolver: `self._asdict()` devuelve `dict[str, Any]`, no
   `dict[str, ArrayLike]`; pasa inadvertido bajo compatibilidad con `Any`,
   pero nadie lo verifica. Construir el dict a mano con los tres nombres lo
   arreglaría al costo de repetirlos — no vale la pena todavía.
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
    StateModel.bind(**fields: object) -> BoundStateModel        # 1× por eje
    BoundStateModel.__call__(*, x, across) -> State              # 1× por paso
    ```

    - **`BoundState` renombrado a `BoundStateModel` (2026-08-13)**: más
      entendible (queda claro que es el objeto ligado del `StateModel`, no
      un estado en sí). De paso, `StateModel`/`BoundStateModel` pasan a ser
      genéricos sobre el `State` concreto (`TypeVar` covariante): sin esto,
      todo `bind()` concreto devolvía el `BoundStateModel` desnudo, cuyo
      `__call__` tipa `-> State` (el `Protocol` neutro), y el tipo concreto
      se perdía — `fluid.bind()(x=..., across=...).density` no tipaba pese a
      que en runtime siempre fue el objeto concreto. Una implementación
      anota `bind(...) -> BoundStateModel[SinglePhaseFluidState]` y el campo
      sobrevive toda la cadena. Ver `docs/design/architecture-v0.2.md` §2.1bis
      para el protocolo completo con el `TypeVar`.

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

    **El contrato del `Protocol` `Rate` es `as_physics_kwargs()`, no
    `physics_key`.** Lo único que el `Protocol` promete — y lo único que
    `loss_func` consume — es el método; el vocabulario canónico vive en las
    *claves del dict* que devuelve. `physics_key: str` / `physics_keys:
    tuple[str, ...]` (#36) son el mecanismo con el que las clases base de
    conveniencia `ScalarRateBase`/`VectorRateBase` (#35) *implementan* ese
    método — un `ClassVar` interno de esas clases, no parte del `Protocol`.
    Esto no es una excepción a "vocabulario canónico, no tabla de
    renombres": es la misma regla enunciada un nivel más arriba. Una
    implementación de `Rate` que armara `as_physics_kwargs()` a mano, sin
    `physics_key` en absoluto, seguiría cumpliendo el contrato igual.
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
    - **Firma genérica con `Self` (corrección 2026-08-14).** El protocolo
      declara `__add__(self, other: Self) -> Self`. Los parámetros son
      contravariantes: un protocolo que promete `other: Rate` obliga a toda
      implementación a aceptar cualquier `Rate`, y
      `BrineRate.__add__(other: BrineRate)` deja de satisfacerlo bajo
      `--strict`. Con `Self`, mypy rechaza `MassRate + BrineRate`
      estáticamente y el retorno preserva el tipo concreto.
    - **Sin `__radd__`, sin `mix()`, sin `__sub__`.** El balance de nodo es
      `reduce(add, rates)`, que sólo requiere `__add__`. `sum()` arranca en
      `0` y ensuciaría la firma con `Literal[0]`; un `mix()` como método
      contradice #34. `__sub__` no tiene cliente e invita a
      `rate_in - rate_out`, el patrón que cancela el denominador de
      `_combine`: quitarlo hace el invariante estructural en vez de
      documental. `__neg__` se queda (orientación de arista).
    - **Sin `@runtime_checkable`.** Verifica presencia de métodos, no
      firmas, y no verifica `physics_key` (un `ClassVar` sin valor no existe
      en runtime). Un `isinstance` que pasa para clases rotas es peor que no
      tenerlo.
    - **El nombre canónico es `ClassVar`.** Un `name` por instancia es la
      tabla de renombres de #21 distribuida en vez de centralizada, con el
      mismo modo de falla. Corolario: el solver no declara ningún parámetro
      de nombre.
    - **`__array_ufunc__ = None` en `BaseRate`.** Sin eso, `ndarray * rate`
      gana precedencia y construye un array de objetos en vez de delegar a
      `__rmul__`.
    - **No heredar de `float` ni de `ndarray`.** `float.__add__` devuelve
      `float` plano: el balance en nodo rompe la abstracción. Heredar
      declara sustituibilidad por un número en todo contexto, lo contrario
      de la frontera `Rate` vs. propiedad de fluido de #3. Consistente con
      #34.
    - **`BaseRate` separa `_rebuild` de `_combine`.** Es la codificación
      literal de las dos reglas de #3: `_rebuild(value)` = *mismo intensivo,
      nuevo extensivo* (`__mul__` y `__neg__` salen correctos por
      construcción, la composición es invariante bajo escalado);
      `_combine(other)` = la regla de mezcla. Una subclase composicional
      overridea esas dos y nada más. `__slots__` vive acá, no en el
      `Protocol`; cada subclase declara el suyo o recupera el `__dict__`.
    - **El solver opera sobre `value`; el álgebra de `Rate` es de la capa de
      propagación.** El balance numérico suma extensivos crudos y adentro
      del `solve_ivp` circulan arrays, no objetos `Rate` (#22, hoisting).
      `__add__` es la operación de `propagate_rates`, pasada topológica
      sobre la red ya resuelta.
    - **Composición como trazador pasivo (v0.2).** `BrineRate` propaga y
      mezcla composición **sin efecto sobre las propiedades físicas**. Es
      hipótesis de modelado explícita, no versión incompleta: es lo que
      elimina la dependencia circular temida en `mass_balance`. Cuando haya
      realimentación (v2+) sigue siendo iteración externa desacoplada
      (Picard), no un solve simultáneo.
    - **La composición se representa como `dict[str, ArrayLike]`.** La
      objeción original (iterar dict en el loop interno del optimizer, sin
      vectorizar) asumía que la mezcla corría dentro del solve. No corre.
      Sin esa restricción el dict gana sobre el array denso: no necesita
      registro de especies a nivel red ni convención de ejes, y la clave
      ausente ya es el cero de la mezcla (`.get(k, 0.0)`) — por eso la
      composición vacía es `{}` y nunca `None`. Se mantiene que `MassRate`
      no lleva el campo: uniformidad en el método, no en el dato.
    - **`BrineRate` valida positividad en `__init__`; `BaseRate` no.** Un
      caudal másico negativo con composición positiva no tiene significado
      físico, y la mezcla ponderada lo convierte en composición negativa sin
      activar ningún guard (`+5` con `−4` da denominador 1, resultado finito
      y absurdo). El chequeo en `__init__` lo agarra en el nacimiento, y
      cubre los negativos que llegan vía `__neg__`/`_rebuild` porque pasan
      por el constructor. No asciende a `BaseRate`: escalares, vectores y
      fasores admiten negativos legítimamente (#3, `ComplexRate`). Es
      invariante de caudal, no de through-variable.
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
26. **Dos clases hermanas de `BoundStateModel`, discriminadas en `bind` por
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
    Traducción en el `__call__` del `BoundStateModel`, una línea.
    - **`across: ArrayLike`, no `float` (corrección 2026-08-13).** `solve_ivp`
      pasa `y` como `ndarray` siempre, incluso para una ODE escalar (`shape
      (1,)`): tipar `float` documenta un desempaquetado, no un contrato.
      Ensanchar una entrada es gratis — un `float` sigue siendo un
      `ArrayLike` válido y ningún caller se rompe; el ensanchado de *salida*
      corría la misma lógica y **ya se cerró** para `FluidState` (ver #5,
      2026-08-13; `GradientResult` nunca fue `float`, ver ahí). `x` queda
      `float`: es la coordenada del integrador, siempre escalar. Costo declarado en
      docstring: las implementaciones deben ser array-safe; en v0.2 se
      documenta, no se testea.
    - **`State.as_physics_kwargs() -> dict[str, ArrayLike]` (corrección
      2026-08-13).** Su consumidor es `gradient_fn`, que ya acepta
      `ArrayLike` (ver `friction_factor(re: ArrayLike, ...)`). Tiparlo
      `float` angosta contra un consumidor que no lo pide — la firma de
      salida de la frontera debe coincidir con la de entrada de `physics/`.
      En su momento esto no reabría la decisión sobre los campos de
      `FluidState`/`GradientResult` — eran dos decisiones distintas, una de
      firma de método y otra de tipo de campo almacenado. La segunda ya se
      cerró (#5, 2026-08-13: `FluidState` pasa a `ArrayLike`;
      `GradientResult` siempre lo fue).
    Firmas: `StateModel[S].bind(**fields: object) -> BoundStateModel[S]`;
    `BoundStateModel[S].__call__(*, x: float, across: ArrayLike) -> S`;
    `State.as_physics_kwargs() -> dict[str, ArrayLike]`;
    `Fluid.bind(*, composition, temperature=None)`;
    `Fluid.get_state(*, composition, temperature, pressure)`.
31. **Inyección de propiedades en correlaciones de viscosidad (2026-08-14).**
    El callable de viscosidad recibe `(pressure, temperature,
    **injectables)`. La clase `Fluid` inyecta lo que ya calculó o conoce:
    `density`, `molecular_weight`, y `pressure_reduced`/
    `temperature_reduced` si `uses_reduced_properties`. Lo que es privado del
    modelo de viscosidad (`mu_ref`, `S`, …) se fija con `functools.partial`.
    Razón: evita doble fuente de verdad para `molecular_weight` y elimina el
    fallback a densidad ideal que había en `RealGas`. Corolario de #5/#10:
    las correlaciones no pueden defaultear parámetros físicos (`density`,
    `molecular_weight`, reducidas) — kw-only sin default, para que un typo en
    el nombre inyectado falle con `TypeError` en vez de usar un default
    silencioso. `**kwargs` de catch-all sí es válido, para ignorar
    injectables no usados.
    **Mecanismo único en `CompressibleFluid`, no por subclase (misma
    sesión, corrección).** El alcance inicial del cambio fue solo
    `RealGas`; la asimetría con `IdealGas` no era una decisión, era que el
    pedido acotó el alcance. `_viscosity_injectables`/`viscosity()` subieron
    a `CompressibleFluid` (`state/fluids/single_phase_fluids.py`) — un EOS
    ideal no implica una correlación de viscosidad `T`-only: `mu = f(T)`
    (Sutherland) es propiedad de la correlación elegida, no de la clase;
    `IdealGas` + LGE (density-dependent) es una combinación válida y usa la
    densidad ideal correctamente. `uses_reduced_properties` (default
    `False`) y el hook `_reduced_injectables` (default `{}`) son lo único
    que una subclase overridea — `RealGas` los override, `IdealGas` no.
    **Rename `molar_weight` → `molecular_weight`** en `IdealGas`/`RealGas`
    (constructor y atributo): eran el mismo dato con dos nombres — el
    injectable ya se llamaba `molecular_weight` (también el nombre usado
    en `physics/gas_correlations/viscosity.py`), y mantener `molar_weight`
    en el constructor era la tabla de renombres que #21 (vocabulario
    canónico) prohíbe. **Bug de paso, `RealGas.compressibility`**: `dz_fn`
    se llamaba con `(pressure, temperature)` absolutos aun cuando `z_fn` se
    evaluaba en reducidas (`Pc`/`Tc` dados) — inconsistencia entre el par
    `z_fn`/`dz_fn` de la misma correlación. Corregido con un helper
    `_reduced_pt` compartido por `z()` y `compressibility()`: mismos inputs
    para ambos. La convención de salida de `dz_fn` no cambia (`dZ/dP`
    respecto de presión *absoluta*, `1/Pa`) — si la correlación es
    analítica en reducidas, el factor de cadena `1/Pc` es responsabilidad
    de quien escribe `dz_fn`, no de `RealGas`.
32. **Organización de módulos e imports.** Los alias de tipo transversales viven
    en `fluidnet/_types.py` (capa −1, no importa nada del paquete).
    `physics/types.py` conserva solo lo específico de física. `physics/` y
    `state/` son hermanos: ninguno importa del otro; si aparece la tentación de
    un import lateral, el símbolo va a `_types.py` o el diseño está mal. Los
    `__init__.py` re-exportan solo si el subpaquete es fachada pública; el
    código interno y los tests importan del módulo que define.
    `fluidnet/__init__.py` expone solo `__version__` hasta v1.0. Verificado por
    `import-linter` (`lint-imports`).
    Los contratos de `import-linter` corren automáticamente vía `pre-commit`; el
    comando canónico sigue siendo `lint-imports` sin argumentos.
33. **Abstracto vs. concreto es un eje de profundidad, no de hermandad.** El
    protocolo vive en `<paquete>/protocol.py`; las implementaciones concretas en
    subpaquetes por dominio (`state/fluids/`, y a futuro `state/electrical/`). No
    existe un paquete `protocols/` transversal.
34. **`StateModel` es el contrato; las ABC son conveniencia.** La librería anota
    siempre contra el Protocol, nunca contra una clase base concreta.
    `CompressibleFluidBase` es pública como ayuda para implementadores (regala
    `viscosity` y `bind`), no como requisito. Herencia opcional, protocolo
    obligatorio.
35. **`BaseRate` pasa a ABC; único método abstracto `as_physics_kwargs`.**
    Hermanos concretos `ScalarRateBase` y `VectorRateBase`, sin herencia
    entre ellos: `physics_key: str` (escalar) y `physics_keys: tuple[str,
    ...]` (vector, #36) son tipos incompatibles — forzar un ancestro común
    entre los dos obligaría a mypy a mentir sobre uno de los dos campos.
    Coherente con #34 (el `Protocol` es el contrato, las ABC son
    conveniencia): un implementador que no quiera ninguna de las dos
    convenciones de `physics_key(s)` puede ignorar `BaseRate` por completo e
    implementar `as_physics_kwargs()` directo (#21).
36. **Convención de eje para rates multi-cantidad: eje de cantidad primero.**
    `VectorRateBase.value` tiene shape `(n_cantidades, *shape_escenario)`,
    no `(*shape_escenario, n_cantidades)`. Razón: `__mul__` contra una
    fracción de split de shape `(n_cantidades,)` tiene que broadcastear
    directo contra `(n_cantidades, *shape_escenario)` — numpy alinea por la
    derecha, así que con el eje de cantidad al final la fracción
    broadcastearía contra el eje de escenario en vez del de cantidad.
    **Complementaria, no contradictoria, con "`StateModel` array-safe
    elementwise sobre el último eje"** (ADR `architecture-v0.2.md` §2.1bis,
    cierre 2026-08-13; ROADMAP §Abiertas "Vectorización por escenarios").
    Son dos arrays distintos con roles de eje distintos: el eje "último" de
    esa decisión es el eje de **escenario/evaluación** de `across`/`x` en el
    integrador; el eje "primero" de ésta es el eje de **cantidad** dentro de
    un único `Rate` multifásico. Leídas sueltas parecen chocar (¿primero o
    último?); resuelto, cada una fija el extremo opuesto del mismo array
    `(n_cantidades, *shape_escenario)`: cantidad adelante, escenario atrás,
    y la ODE opera elementwise sobre ese último eje sin ver el primero.
37. **Constructor canónico de `VectorRateBase`: un único array empaquetado.**
    `__init__(value: ArrayLike)` con `value.shape[0] == len(physics_keys)`
    (#36) es el contrato; `from_phases(**kwargs)` (classmethod) y
    properties nombradas de solo lectura (p. ej. `.gas`, `.liquid`) son
    ergonomía de borde, no el contrato — construyen o leen sobre el array
    empaquetado, nunca lo reemplazan. Consecuencia: `_rebuild` queda uniforme
    en toda la jerarquía (`ScalarRateBase` y `VectorRateBase` reciben y
    devuelven el mismo tipo de `value`, un `ArrayLike`), sin rama especial
    para el caso multi-cantidad.
38. **`LossFunc` es una instancia, y es stateless respecto del eje.** El
    constructor lleva **únicamente política numérica**: `gradient_fn`,
    `detailed_fn`, método de root-find, tolerancias, parámetros de
    `solve_ivp`, grilla de diagnóstico. Los **datos de eje** (`D`, `L`,
    rugosidad, inclinación, `K`) llegan siempre como argumentos de llamada,
    nunca como estado de constructor. Consecuencia directa: una sola
    instancia es reutilizable en todos los ejes y puede vivir como atributo
    de red, con override por eje vía el mismo mecanismo ya usado para el
    fluido (`G.edges[u, v].get('loss', network.loss)`) — sin mecanismo
    nuevo. Una instancia por eje sigue siendo legítima, pero como diferencia
    de **política** (otra tolerancia, otra correlación), jamás de datos.
    Razón fuerte: el fitting de v0.5 varía parámetros de eje (rugosidad,
    diámetro efectivo); si esos datos vivieran en la instancia, el optimizer
    tendría que clonar `LossFunc` por evaluación y por eje, o mutar un
    objeto compartido entre ejes dentro del loop del optimizer.
    *(2026-08-17)*
39. **Los dos protocolos de loss difieren en el momento de binding del
    estado.** No es una diferencia de convención sino de lifecycle:
    `AlgebraicLoss` recibe un `State` **ya evaluado** (el solver bindea y
    evalúa una vez, fuera de la loss) y no tiene presión en la firma — que
    es literalmente su definición. `IntegralLoss` recibe el
    `BoundStateModel` **sin evaluar** más `p_boundary`, y lo evalúa dentro
    del `rhs` en cada paso. Cae solo del discriminante ya cerrado (#7, el
    `StateModel` declara el régimen): el régimen *es* cuándo se puede
    evaluar el estado — un fluido incompresible se evalúa antes de conocer
    `P`, uno compresible no. Si ambos protocolos recibieran el
    `BoundStateModel`, la loss algebraica tendría que invocarlo con un
    `across` de mentira, que es API que le miente al usuario.
    *(2026-08-17)*
40. **`solve_rate` tiene default funcional, no `NotImplementedError`.**
    Revierte el default de la decisión del 2026-08-09. Razón: en un DAG con
    BC en nodos intermedios (v1.0) la formulación es necesariamente nodal —
    la formulación por caudales (mesh/Hardy Cross) requiere ciclos
    independientes como base, y en un DAG no hay ninguno. O sea que
    `Q = f(P_up, P_down)` no es un extra de otros dominios: **es el corazón
    del solver 2**, y un `NotImplementedError` heredado rompería
    `mass_balance` para toda loss escrita por un usuario.
    Semántica del default: *"si querés, dame la inversa explícita; si no, te
    la armo yo"*. Se expone vía `solve_rate_is_defaulted` — **property de
    solo lectura**, sin setter — más un `log.info` al construir. Aparece en
    el `repr` y `Result` puede reportarlo como metadata del solve. Si el
    usuario provee una inversa que no cierra, es problema del usuario: la
    librería no valida física ajena (coherente con "ningún solver adivina",
    ADR §2.4). *(2026-08-17)*
41. **Root-find de `solve_rate`: elegible, default bracketed (`brentq`).**
    Newton disponible como opción. Razones del default:
    (a) un extremo del bracket es **gratis por física** — `Q = 0` da
    `dp = 0` y, con constitutiva monótona, el signo del residual ahí es
    conocido siempre; solo hay que expandir el otro extremo hasta cambio de
    signo, sin tuning ni input del usuario;
    (b) monotonía **no** garantiza convergencia global de Newton (hace falta
    además condición de convexidad), y hay un modo de falla concreto: en
    régimen turbulento `dp ~ Q²`, así que `d(dp)/dQ → 0` cuando `Q → 0` —
    Newton arrancando de caudal chico divide por derivada casi nula, y ése
    es el arranque natural de un solver sin estimación previa;
    (c) en régimen integral la derivada numérica cuesta **una integración de
    ODE completa extra por iteración**; Newton hace menos iteraciones pero
    cada una cuesta el doble, y el balance se mide en v0.5, no se afirma
    ahora.
    Un default que puede explotar en el caso de arranque más común es mal
    default aunque sea el más rápido cuando anda. Nota para el paper: la
    unicidad que sostiene el solver de red (content de Millar) es la misma
    que hace la inversión por eje incondicionalmente convergente.
    Corolario habilitado por #38: una loss concreta con inversa analítica
    (Darcy: `Q ∝ √dp`) sobrescribe `solve_rate` y no usa root-find alguno.
    *(2026-08-17)*
42. **`EdgeResult` y `Result` son contratos distintos.** `EdgeResult` es la
    salida de `solve_dp`/`solve_rate`: envuelve el `sol` de `solve_ivp` más
    properties chicas (`p_in`, `p_out`, `dp`, `rate`). `Result` es la salida
    del **solver de red**. En el caso algebraico `EdgeResult.sol is None` —
    no hay ODE y no hay nada que guardar. El objeto es "el estado resuelto
    de un eje", no "el resultado de la ODE": con esa definición el
    algebraico es el degenerado natural en vez de un caso raro, mismo
    criterio que ya fija los dos protocolos. *(2026-08-17)*
43. **`Result` es una red gemela de `nx.DiGraph`** (o wrapper sobre ella),
    con los resultados como atributos de nodo y eje — no el dataclass frozen
    que describía el ADR. Reutiliza el acceso diccionario-like nativo de
    networkx, `to_frames()` ya está portado de mineplanner, y la
    vectorización de v0.5 entra sin tocar nada (los dicts de networkx
    guardan arrays igual que floats). **Forma cerrada, implementación
    diferida**: no bloquea `LossFunc`/`EdgeResult`. Consecuencia sobre el
    ADR §2.5: la promesa "consumible sin exigir que el consumidor conozca
    networkx" se restringe a las vistas — `to_frames()` es la interfaz
    pública, el grafo es detalle de implementación. *(2026-08-17)*
44. **`diagnose()`: índice `(edge, x)`, esquema abierto.** Cierra el ítem
    abierto desde 2026-08-10. El output es el de salida de la `detailed_fn`
    tal cual — no se inventa estructura. Un `dict` por punto de evaluación,
    con índice `(edge, x)` y `x = NaN` en el caso algebraico (no `None`:
    `NaN` significa lo mismo — no hubo integración, no hay coordenada — y no
    rompe `.loc` sobre el nivel del `MultiIndex`). Trivialmente convertible
    a `DataFrame`. **Los campos los elige el autor de la `detailed_fn`, no
    el usuario final** — es esquema abierto, no configurable; el usuario
    elige la loss, y el filtrado de columnas lo hace pandas
    (`df[['f', 'Re']]`), no un parámetro de la API. Una lista de nombres
    habría obligado a `detailed_fn` a declarar y validar su vocabulario por
    adelantado, infraestructura para resolver algo que pandas ya resuelve.
    **Columnas ragged son comportamiento declarado**: si dos ejes usan
    losses distintas (que #38 permite explícitamente), sus `detailed_fn`
    devuelven campos distintos y el `DataFrame` concatenado da unión de
    columnas con `NaN`. Es correcto, y hay que documentarlo o el primer
    usuario que mezcle correlaciones lo va a leer como bug. *(2026-08-17)*
45. **Grilla de diagnóstico: `t_eval` relativo, `dense_output` descartado.**
    `dense_output=True` es interpolación cara para una grilla que se puede
    declarar de antemano; `t_eval` **es** la grilla declarada de la decisión
    del 2026-08-10. Va como parámetro opcional de la instancia
    (`EdgeResult.sol` queda poblado sólo si se pidió). **Tiene que ser
    relativo** — un `int` (n puntos equiespaciados) o un array normalizado
    en `[0, 1]` — y la loss arma `t_eval = grid * L` en cada llamada: `L`
    cambia por eje, así que una grilla absoluta en metros metería estado de
    eje en el constructor y rompería #38. *(2026-08-17)*

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
- Golden tests: preferir literatura (Kermit Brown ej. 4.7; Ahmed,
  *Reservoir Engineering Handbook*, ej. 2-14 para viscosidad LGE) sobre
  valores pinneados de `fluids` cuando ambos estén disponibles.
  `tests/physics/test_gas_viscosity_vs_book.py` sigue el mismo patrón que
  `test_beggs_brill_vs_book.py`: expone un `_lee_gonzalez_eakin_detailed`
  (intermedios `K`/`X`/`Y`) hermano de `lee_gonzalez_eakin_viscosity`,
  mismo mecanismo que `_beggs_brill_detailed` (#25) pero a nivel de
  correlación de viscosidad, no de gradiente. El caso del libro fue el que
  detectó un bug real: faltaba el prefactor `1e-4` de la Ec. 2-63
  (`mu[cP] = 1e-4 * K * exp(X * rho^Y)`) — sin él, el resultado daba 4
  órdenes de magnitud de más.
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
3. Commits atómicos, chicos, contra `dev` limpia.
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
- **Cerrado (2026-08-13, ver #5): campos de `FluidState` son `ArrayLike`.**
  `GradientResult` ya lo era desde su creación (`physics/types.py`, commit
  `9780d43`) — nunca fue `float`; la afirmación previa de esta sección
  ("decisión de diseño abierta", "siguen `float`") describía un estado que
  el código no tenía. Ya no es excepción: tratar como housekeeping normal
  de tipado.
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

## Atribución en commits

```json
{
  "attribution": {
    "commit": "Assisted-by: Claude Code <noreply@anthropic.com>",
    "pr": ""
  }
}
```
