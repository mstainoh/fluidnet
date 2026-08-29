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

**Scope formal.** El dominio de problemas es: potencial en nodos (*across*),
flujo en ejes (*through*), relación constitutiva monótona expresable como
`Δ = ∫ f(P, x, ...) dx`. Formalmente, un grafo lineal / bond graph con
elementos disipativos. La garantía de existencia y unicidad tiene dos
niveles que no hay que confundir: Lipschitz para la ODE del eje, content de
Millar para el problema de red. Quedan fuera, como límite matemático
declarado: histéresis, relaciones implícitas no reducibles a ODE, y
elementos activos (rompen la monotonía).

Este es el enunciado correcto del alcance del proyecto — más preciso y más
defendible que "redes de fluidos" o "redes acíclicas", y es el que debe ir al
README.

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

**Definición de `Rate` por contrato, no por contenido**: *puede sumarse con
otros rates y dar balance cero, y puede meterse en una loss function.* Eso es
todo lo que el core exige. Es la definición que hace que `ComplexRate`
(fasores, red AC) entre sin tocar el álgebra.

**La composición es atributo de subclase, no de clase.** `ScalarRate` no
tiene el campo — no lo tiene vacío ni con un singleton `{"fluid": 1.0}`. La
uniformidad va en el método (la regla de mezcla), no en el dato. Un singleton
obligaría a iterar un dict y hacer una división ponderada por nodo en el caso
monofásico, que es justamente el loop interno del optimizer de v0.5, y no
vectoriza.

`as_physics_kwargs()` **sí es un método de `Rate`**: la magnitud extensiva va
a la loss function siempre. Lo que no es responsabilidad del `Rate` es
componerla con la fracción de fase — eso es `loss_func`. Y se arma una vez
por eje, **fuera del `solve_ivp`**: en steady-state sin intercambio de masa
el mass rate es constante a lo largo del eje, así que el RHS sólo actualiza
`pressure`. Sin ese hoisting, con 10²–10³ evaluaciones del RHS por eje, el
costo del wrapper deja de ser despreciable frente a un float pelado.

**No existe un `MultiphaseRate` con fracciones.** El split líquido/gas depende
de un flash `(P, T, comp)` y cambia a lo largo del caño: es trabajo de EOS,
o sea del `StateModel`. Un `StateModel` de fracción impuesta ("asumo 30 % de
vapor") es válido como hipótesis explícita de modelado, pero es un fluido, no
un rate.

Jerarquía mínima v0.2: `Rate` (protocolo/ABC) → `ScalarRate`. `BrineRate`
(composición iónica → saturation indices) queda como ejemplo/extensión, no
core.

**Decisión heredada de mineplanner a corregir**: `propagate_rates` usa `sum(propagated[p] for p in predecessors)` — `sum()` arranca de `0` (int). `Rate` debe soportar `0 + rate` (`__radd__`) o la propagación usa `functools.reduce`. Detalle chico, pero es exactamente el bug que rompe el polimorfismo.

**Alternativas descartadas (`BaseRate`, `CLAUDE.md` #35–#37).**

- *Clase única con `physics_keys: tuple[str, ...]` siempre, escalar como
  1-tupla.* Habría evitado la partición en `ScalarBaseRate`/`VectorBaseRate`.
  Descartada por legibilidad del camino caliente — `as_physics_kwargs()`
  devolviendo `{key: value}` para el caso escalar, sin desempaquetar una
  1-tupla en cada `__init__`/`__repr__`/callsite — **no por performance**:
  `as_physics_kwargs()` está hoisteado fuera de `solve_ivp` (`CLAUDE.md`
  #22), así que ninguna de las dos formas paga costo en el loop caliente.
- *Asimetría de validación deliberada.* `VectorBaseRate.__init__` hace
  `asarray` + chequeo de `shape[0] == len(physics_keys)`; `ScalarBaseRate`
  no valida nada. No es una inconsistencia a limpiar: un escalar no tiene
  eje de cantidad que pueda desalinearse, así que no hay invariante que
  proteger. El costo de validar ahí sería ritual, igual que un dict
  identidad en #21.

### 2.1bis `StateModel` / `Fluid` — fábrica de estado, no estado

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

**El protocolo es neutro; `Fluid` es una implementación.** Lo que la capa
necesita genéricamente es un transformador de la variable *across* del nodo
(más el estado propagado) en los argumentos del `gradient_fn`. En fluidos eso
es `(comp, T, P) → (ρ, μ, β, σ)`; en una red eléctrica AC es
`(V) → impedancia`. La forma es la misma.

**El protocolo tiene dos métodos, en dos tiempos.** La cadena `comp → T → P`
no es nesting sintáctico: es **binding time**, ordenado por frecuencia de
cambio. La composición es constante a lo largo del eje (steady-state sin
intercambio de masa); los campos prescritos son función conocida de `x`; el
potencial es la incógnita, que cambia en cada evaluación del gradiente.

```
StateModel.bind(**fields: object) -> BoundStateModel         # 1× por eje
BoundStateModel.__call__(*, x, across) -> State               # 1× por evaluación
```

**`composition` sale del `Protocol`** *(corrección 2026-08-13)*. No todo
`StateModel` tiene composición — un modelo de agua dulce se parametriza solo
por `T`. Forzarla en la firma neutra es el mismo error que forzar
`temperature`, ya descartado más abajo: la composición pasa a ser un campo
más, declarado por la implementación concreta (`Fluid.bind(*, composition,
temperature=None)`), que es donde hay información para tipar.

**La distinción propagado/prescrito se muda al solver** *(corrección
2026-08-13)*. El `Protocol` ya no la codifica; la codifica quien arma los
kwargs de `bind`:

```
bound = model.bind(
    **propagated_fields(rate),      # de propagate_rates: composición
    **prescribed_fields(G, u, v),   # de atributos declarados: T, etc.
)
```

Sigue siendo la distinción que sostiene el límite declarado más abajo
("campos prescritos son datos, no cantidades conservadas"): los prescritos
no se balancean en nodos, la composición viaja por la topología. Queda
documentada acá, no tipada en el `Protocol`.

**Es aplicación parcial, no construcción.** La distinción es indiferente para
el intérprete y decisiva para la arquitectura: la composición sale de
`propagate_rates`, o sea de runtime. Si entrara por `__init__`, el
`StateModel` dejaría de poder ser atributo declarativo de la red y habría que
instanciar el EOS en tiempo de solve. Con `bind` hay un modelo declarativo por
red/nodo y un objeto ligado liviano por eje, efímero: no se cachea, no se
serializa, no aparece en `Result`.

**`x` asciende al protocolo; los campos físicos no.** Un potencial secundario
prescrito como perfil `T(x)` necesita que el modelo sepa dónde está. Las tres
formas posibles de lograrlo no son equivalentes:

| Forma | Problema |
|---|---|
| `__call__(*, across, temperature)` | `temperature` asciende al protocolo — no significa nada en el demo AC |
| `__call__(*, across, **fields)` | muere con `mypy strict`; obliga a `loss_func` a saber qué campos existen en el dominio, o sea a traducir |
| `__call__(*, x, across)` | `x` no es magnitud física sino coordenada del eje: existe en todo dominio y ya está en el vocabulario del integrador |

El perfil queda capturado en el closure del `BoundStateModel`, y la firma de
`bind` la declara la implementación concreta (`Fluid.bind(composition,
temperature=None)`). Consecuencia buscada: **agregar perfiles de temperatura
en v2 no toca `physics/`, ni `loss_func`, ni el protocolo** — toca el parser
de atributos de eje y el `bind` de `Fluid`. El costo visible hoy es un
argumento inerte en la firma pública durante dos versiones; es más barato que
el cambio de firma alternativo.

**Escalar, ausente y perfil se resuelven una vez, en `bind`.** El
discriminante no es "fijo vs. variable" sino **si el objeto ligado necesita
`x`**: `None` y un escalar son el mismo caso. Dos clases hermanas, rama
decidida por `callable(field)` fuera del loop.

**El objeto ligado no entra al integrador.** Lo que entra a `solve_ivp` es el
`rhs`; el `bound` se llama adentro y devuelve un `State`, no una derivada. La
cadena completa:

```
StateModel → bind → BoundStateModel → (x, across) → State
                                                       → as_physics_kwargs()
rate.as_physics_kwargs() ────────────────────────────→ gradient_fn ← edge_attrs
```

**Regla general que ordena la capa**: *todo lo que no depende de `x` se liga
antes del integrador; lo que depende de `x` entra por el `bound`.* Es el
mismo criterio detrás del hoisting de `rate.as_physics_kwargs()` y de las dos
clases hermanas — tres decisiones que parecían sueltas y son una sola.

**`across` en el `Protocol`, `pressure` de la implementación concreta para
abajo.** *(Cerrado 2026-08-10, `CLAUDE.md` #30.)* El `Protocol` es neutro por
diseño: `pressure` ahí es la misma evidencia en contra del diferencial
physics-agnostic que motivó el nombre neutro (ver arriba, "que el protocolo se
llame `Fluid`"). Hacia abajo, `pressure` ya está en `physics/`, es el
vocabulario del dominio, y renombrarlo tocaría capa cero testeada. La
traducción vive en el `__call__` del `BoundStateModel` concreto, una línea.

**`across: ArrayLike`, no `float`** *(corrección 2026-08-13)*. `solve_ivp`
pasa `y` como `ndarray` siempre, incluso para una ODE escalar (`shape
(1,)`): tipar `float` documenta un desempaquetado, no un contrato. Ensanchar
una entrada es gratis — un `float` sigue siendo un `ArrayLike` válido y
ningún caller se rompe; el ensanchado de *salida* corría la misma lógica y
**ya se cerró** (`CLAUDE.md` #5, 2026-08-13): campos de `FluidState` pasan a
`ArrayLike` (`GradientResult` nunca fue `float`, ver ahí). `x` queda
`float`: es la coordenada del integrador, siempre escalar. Costo declarado
en docstring: las implementaciones deben ser array-safe; en v0.2 se
documenta, no se testea.

**`State.as_physics_kwargs() -> dict[str, ArrayLike]`** *(corrección
2026-08-13)*. Su consumidor es `gradient_fn`, que ya acepta `ArrayLike` (ver
`friction_factor(re: ArrayLike, ...)`); tiparlo `float` angosta contra un
consumidor que no lo pide, y la firma de salida de esta frontera debe
coincidir con la de entrada de `physics/`. En su momento esto no reabría la
decisión sobre los campos almacenados de `FluidState`/`GradientResult` —
eran dos decisiones distintas; la segunda ya se cerró (ver el párrafo
anterior).

**`StateModel`/`BoundStateModel` genéricos sobre el `State` concreto**
*(2026-08-13, renombre `BoundState` → `BoundStateModel` incluido)*. Sin
parámetro de tipo, toda implementación concreta de `bind` tenía que
devolver el `BoundStateModel` desnudo, cuyo `__call__` tipa `-> State` — el
`Protocol` neutro de dos métodos (`as_physics_kwargs()` nomás). El tipo
concreto (`SinglePhaseFluidState.density`, etc.) se perdía en el camino:
`fluid.bind()(x=..., across=...).density` no tipaba, aunque en runtime el
objeto devuelto siempre fue el concreto. Con un `TypeVar` covariante ligado
a `State`, `StateModel[S]`/`BoundStateModel[S]` propagan `S` a través de
toda la cadena `bind → BoundStateModel → State`; una implementación
concreta anota `bind(...) -> BoundStateModel[SinglePhaseFluidState]` y el
tipo de campo sobrevive hasta el call site. `StateModel`/`BoundStateModel`
en sí mismos siguen neutros — el parámetro de tipo no ata nada a fluidos,
solo lo transporta.

Protocolo resultante:

```python
from typing import Protocol, TypeVar

S_co = TypeVar("S_co", bound="State", covariant=True)


class StateModel(Protocol[S_co]):
    def bind(self, **fields: object) -> BoundStateModel[S_co]: ...


class BoundStateModel(Protocol[S_co]):
    def __call__(self, *, x: float, across: ArrayLike) -> S_co: ...


class State(Protocol):
    def as_physics_kwargs(self) -> dict[str, ArrayLike]: ...
```

**Propiedades por fase; la mezcla la calcula `physics`.** *(sin cambios — ver
convención de sufijos.)* La forma concreta del estado queda cerrada:
`SinglePhaseState` / `MultiPhaseState`, `NamedTuple`, con
`as_physics_kwargs()` del lado del estado. El parseo es estático porque **el
número de fases es propiedad de la implementación del `StateModel`, no del
valor**: un flash puede cruzar el punto de burbuja a lo largo de `x`, pero un
modelo multifásico emite igual ambas fases, una con fracción cero.

**Límite declarado: los campos prescritos son datos, no cantidades
conservadas.** No se impone balance sobre ellos en los nodos. Si dos
corrientes a distinta temperatura se mezclan, reconciliar el perfil es
hipótesis del usuario. Esto es lo que separa un campo *prescrito* de uno
*resuelto*: el segundo exige un segundo par (across, through) — la conjugada
de `T` es flujo de calor, no mass rate — y por lo tanto un segundo dominio de
bond graph balanceado en cada nodo. **fluidnet resuelve un dominio; los
dominios adicionales se admiten como campos prescritos.** Esa frase explica
de una sola vez por qué el problema es ODE y no PDE, por qué v2 es aditivo y
qué abre v2.0+.

**Sobre la vectorización — cierre 2026-08-13.** La integración en `x` no es
vectorizable: el paso `n+1` depende del `n`. La array-safety de `get_state`
y de `gradient_fn` es **una sola restricción de diseño que habilita tres
cosas distintas** — y conviene no confundirlas:

| Caso | Qué es | Qué compra |
|---|---|---|
| `vectorized=True` de `solve_ivp` | `(n, k)`: k puntos de prueba del mismo sistema en el mismo `t` | Jacobiano por diferencias finitas en 1 llamada en vez de n — real en BDF/Radau |
| Escenarios apilados | `(N,)`: N sistemas con Jacobiano diagonal | N corridas en una pasada |
| Potencial acoplado | `(2,)`: `[P, T]` o `[Re, Im]` — componentes ligadas | v2 sin tocar el protocolo |

`vectorized=True` **no** sirve para escenarios: sus columnas comparten `t`,
comparten paso y no aparecen en `sol.y`. Es una confusión que el nombre
invita a cometer.

La formulación correcta no es "soportamos escenarios" sino: **las
implementaciones de `StateModel` deben ser array-safe elementwise sobre el
último eje.** De esa única restricción se derivan los tres casos de la
tabla. El caso "escenarios apilados" es la forma del loop interno del
fitting de v0.5 — ver ROADMAP §Abiertas para el alcance real de ese caso
(no es tan directo como "apilar y listo").

**Complementaria, no contradictoria, con "eje de cantidad primero" de
`VectorBaseRate`** (`CLAUDE.md` #36). Son dos arrays distintos con roles de
eje distintos: acá el eje "último" es el de escenario/evaluación de
`across`/`x`; en `VectorBaseRate.value` el eje "primero" es el de cantidad
(las fases de un rate multifásico). Un `Rate` multi-cantidad vectorizado por
escenarios termina con shape `(n_cantidades, *shape_escenario)` — cantidad
adelante para que `__mul__` por fracción de split broadcastee bien (#36),
escenario atrás para que la ODE opere elementwise sobre él sin ver el eje de
cantidad. Ambas decisiones fijan extremos opuestos del mismo array a
propósito, no por casualidad.

### 2.2 loss_func — dos protocolos, no uno

El código de mineplanner ya reveló que hay dos firmas distintas según el régimen físico:

**Protocolo algebraico** (monofásico, incompresible — el dp no depende de la presión absoluta):

```
AlgebraicLoss.solve_dp(*, rate: Rate, state: State, **edge_attrs) -> ArrayLike
```

**Protocolo de integración** (multifásico / compresible — el dp depende de p local, se integra dp/dL desde la frontera conocida):

```
IntegralLoss.solve_dp(*, rate: Rate, state: BoundStateModel,
                      p_boundary: ArrayLike, **edge_attrs) -> EdgeResult
```

Ambos declarados como `typing.Protocol` explícitos (`AlgebraicLoss`, `IntegralLoss`). Convención de signos heredada y confirmada: `dp = p_downstream − p_upstream` (pérdida → negativo). Kwargs extra absorbidos con `**kwargs` — el patrón de mineplanner de "declarar solo lo que usás" funciona bien y se mantiene.

**Tres correcciones sobre la versión previa de esta sección** *(2026-08-17)*.
(a) La firma anota `State`/`BoundStateModel`, **no `Fluid`**: el `StateModel` es
el protocolo neutro y `Fluid` una implementación suya; anotar `Fluid` rompería
la arquitectura physics-agnostic en la firma misma — el demo eléctrico AC de
v2.0 no podría tipar. (b) El retorno es `ArrayLike`, **no `float`**: mismo caso
exacto que la corrección `FluidState: float → ArrayLike` del 2026-08-13, y
requisito de la vectorización por escenarios de v0.5. (c) La diferencia entre
los dos protocolos **es el momento de binding del estado** (`CLAUDE.md` #39):
el algebraico recibe un estado ya evaluado y no tiene presión en la firma; el
integral recibe el modelo ligado sin evaluar y lo evalúa en cada paso del
`rhs`. Si ambos recibieran el `BoundStateModel`, la loss algebraica tendría que
invocarlo con un `across` de mentira.

Consecuencia directa de que `physics/` sea enteramente keyword-only (`CLAUDE.md` decisión #15): `loss_func` puede despachar hacia la función de gradiente correcta filtrando sus propios kwargs por `inspect.signature(gradient_fn).parameters` en vez de necesitar un adaptador por modelo. `physics` tampoco conoce la orientación del edge — es `loss_func` quien adapta signo/dirección (caudal, inclinación) antes de invocar la función de gradiente; la dirección de integración en sí la resuelve el solver (`t_span` de `solve_ivp`), no `physics` ni `loss_func`.

**El discriminador `AlgebraicLoss`/`IntegralLoss` lo declara el `Fluid`.** *(Cerrado 2026-08-07; este documento lo listaba como abierto.)* El problema era real: qué protocolo aplica es un valor de *runtime* (`compressibility == 0`), no una propiedad estática de la firma de `loss_func` ni de `gradient_fn`, así que no se podía leer de `signature`. Y hay una tensión adicional: `AlgebraicLoss` no tiene presión en la firma — es literalmente su definición — así que si el `Fluid` siempre necesitara `P` para devolver `ρ`, una loss algebraica no podría llamarlo.

La salida no es que lo declare el usuario al registrar la loss ni que lo chequee el solver: **es una propiedad del `Fluid`**. El `Fluid` es dueño del EOS, así que es el único que sabe si `∂ρ/∂P = 0`. Un fluido incompresible expone propiedades sin necesitar `P` y por lo tanto es compatible con `AlgebraicLoss`; un fluido compresible exige `P` y solo es compatible con `IntegralLoss`. La `loss_func` hereda el régimen del fluido que recibe. Es coherente con "ningún solver adivina" (§2.4) y con que `compressibility` sea estado y no flag.

**Diagnósticos: no hay decorador.** *(Corrección 2026-08-17; este bloque
describía un mecanismo ya muerto.)* El `@diagnostic` como decorador con canal
lateral quedó obsoleto en dos pasos: el 2026-08-10 el mecanismo pasó a ser
**post-proceso sobre la solución convergida** (ni contextvar ni colector
durante el solve), y el 2026-08-17 `LossFunc` quedó cerrado como **instancia**
(`CLAUDE.md` #38) — no se decora un método abstracto de un `Protocol` y se
espera que el registro lateral funcione igual.

Lo que queda: `detailed_fn` es un **argumento del constructor** de la
instancia de loss, declarado explícito, nunca descubierto por convención de
nombre (`_*_detailed`). `diagnose()` se corre después de converger, replayeando
`detailed_fn` sobre el `P(x)` ya integrado en la grilla declarada por `t_eval`.
El overhead es chico y la alternativa no aporta información: no hace falta el
régimen de flujo en mil puntos por eje — si se quiere esa resolución, se declara
la grilla y se paga explícitamente. Durante el fitting el costo de diagnóstico
es cero, que era el argumento original a favor del post-proceso.

Output: el de la `detailed_fn` tal cual, con índice `(edge, x)` y esquema
abierto (`CLAUDE.md` #44). El `full_output: bool` de `darcy_dp` desaparece de
la firma pública igual, pero porque `EdgeResult` ya lo hace innecesario, no
porque lo absorba un decorador.

Built-ins v0.2: `constant_friction` (baseline trivial, para tests y docs) y `darcy_weisbach` (portado de `pressure_functions.py`, que ya está en buen estado: Chen approx, régimen laminar/transición/turbulento, minor losses K, vectorizado numpy). `darcy_weisbach` obtiene las propiedades pidiéndole el `FluidState` al `Fluid`, no leyéndolas del `Rate` ni de la red.

**Contrato de `LossFunc`: dos métodos.** `solve_dp(rate, state,
**edge_attrs) -> dp` es abstracto; `solve_rate(dp, state, **edge_attrs) ->
rate` vive en el protocolo integral con `NotImplementedError` por default.

Las dos direcciones `Q→P1→P2` y `Q→P2→P1` **no son modos distintos del
contrato**: son el signo del `t_span`, ya resuelto por el integrador. Los
ejes ortogonales reales son el régimen (algebraico/integral, declarado por el
`StateModel`) y cuál es la incógnita (`dp` o `rate`); el segundo es común a
ambos regímenes, así que vive en el contrato base y no duplica la matriz.

**`solve_rate` tiene default funcional** *(corrección 2026-08-17; revierte el
`NotImplementedError` de 2026-08-09)*. La justificación previa —"no es el
régimen natural de los fluidos, pero sí el de otros dominios"— era falsa. Si
`mass_balance` plantea incógnitas nodales de presión, cada eje tiene que
entregar `Q = f(P_up, P_down)`: eso **es** `solve_rate`. Y no hay alternativa
dentro del scope: la formulación por caudales (mesh/loop, Hardy Cross) necesita
ciclos independientes como base, y en un DAG no hay ninguno — los caudales
quedan determinados por las BC solas. Para redes con BC en nodos intermedios
(v1.0) la formulación nodal no es una opción entre varias: es la única. Un
`NotImplementedError` heredado rompería el solver 2 para toda loss escrita por
un usuario.

Semántica: *"si querés, dame la inversa explícita; si no, te la armo yo"*,
expuesto vía `solve_rate_is_defaulted` (property de solo lectura) más un
`log.info`. Default por root-find **bracketed** (`brentq`), con Newton
elegible; razones en `CLAUDE.md` #41 — el extremo `Q = 0` del bracket es gratis
por física, monotonía sola no da convergencia global de Newton, y
`d(dp)/dQ → 0` en turbulento es el modo de falla concreto. Una loss con inversa
analítica sobrescribe `solve_rate` y no usa root-find alguno.

**Consecuencia sobre la anidación de métodos numéricos** (ROADMAP §Abiertas):
`solve_rate` sobre `IntegralLoss` es ODE dentro de root-find dentro de
root-find. Tres niveles, y a partir de esta corrección están en el camino a
v1.0, no fuera de él.

**Por qué se mantienen dos protocolos** si el algebraico es formalmente el
caso degenerado del integral (sin `P` en el integrando, la integral colapsa a
`grad × L`): porque no tiene sentido formular una ODE cuando la solución
analítica existe y es simple de poner en código. Es una razón de formulación,
no de performance — conviene decirlo así, porque "¿por qué dos protocolos?"
es una pregunta previsible de reviewer.

**Vocabulario canónico en vez de traducción.** Los nombres de los parámetros
de `physics` son contrato. `loss_func` arma la unión `{**rate_kwargs,
**state_kwargs, **edge_attrs}` y filtra por `signature(gradient_fn)`; un dict
de override existe sólo donde un modelo tenga razón genuina para desviarse.
Una tabla de renombres general sería deuda: crece con cada modelo nuevo y se
desincroniza en silencio — el kwarg no pasa, entra un default, y el error
aparece lejos del origen.

Con eso, `loss_func` queda reducida a un `solve_ivp` wrappeando una función
de `physics`, que es lo que la hace replicable modelo a modelo. **Compone, no
traduce**: la aridad multifásica es cómputo (`Rate` aporta el extensivo, el
`StateModel` la fracción de fase, el producto da los rates por fase), no
renaming.

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

**Corrección 2026-08-17**: `Result` **no es un dataclass frozen** — es una red
gemela de `nx.DiGraph` (o un wrapper sobre ella) con los resultados como
atributos de nodo y eje (`CLAUDE.md` #43). Reutiliza el acceso
diccionario-like nativo de networkx, `to_frames()` ya está portado de
mineplanner, y la vectorización de v0.5 entra sin tocar nada. Un `frozen=True`
conteniendo un `nx.DiGraph` mutable daba inmutabilidad decorativa de todos
modos. Consecuencia sobre la promesa de más abajo: "consumible sin exigir que
el consumidor conozca networkx" se restringe a las **vistas** — `to_frames()`
es la interfaz pública, el grafo es detalle de implementación.

`Result` es distinto de `EdgeResult` (`CLAUDE.md` #42), que es la salida de
`solve_dp`/`solve_rate`: estado resuelto de **un eje**, con el `sol` de
`solve_ivp` (o `None` en el caso algebraico) más `p_in`/`p_out`/`dp`/`rate`.

Producido por el solver, nunca por loss functions. Contenido:

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

**Cerradas (10/08/2026 — sesión de diseño `@diagnostic`):**

11. **El canal de diagnósticos es post-proceso, no intercepción.** Cierra el
    ítem 8 de "Abiertas". La disyuntiva contextvar vs. colector presuponía que
    el diagnóstico se capturaba *durante* el solve; una vez que se replayea
    sobre la solución convergida, no hay nada que capturar en vuelo y la
    disyuntiva desaparece. Motivo de fondo: el gradiente es puntual y el
    diagnóstico es de eje; la asimetría no la resuelve el contenedor de
    retorno, la resuelve mover la evaluación después de la integración, donde
    la coordenada del eje ya está determinada.
12. **`GradientResult` inmutable en forma y en contrato** — `NamedTuple`, tres
    componentes, `total` como property. Rechazadas explícitamente: campo
    `extra: dict[str, Any]` (introduce `Any` en capa cero y repite el patrón
    que #11 de `CLAUDE.md` prohíbe una capa más arriba) y `__array__` →
    `total` (coerción lossy silenciosa de una descomposición).
13. **Diagnóstico en dos niveles; el nivel 1 se declara, no se descubre.** La
    descomposición es nivel 0 y universal. Los intermedios de correlación son
    nivel 1 y opcionales, pasados como `detailed_fn` explícita al construir la
    loss. Descubrir el hermano por convención de nombre (`_*_detailed`) se
    descartó: no es greppeable desde el consumidor y no funciona para
    correlaciones de terceros, que es justamente el caso que sostiene el
    diferencial physics-agnostic.

**Abiertas (no bloquean la arquitectura):**

9. **MVP de 26 tests**: declarado perdido y superado (ver `ROADMAP.md`). Este draft es la spec de reconstrucción.
10. **Diseño del solver 3**: falta revisar `header_network_optimizer.py` y `network_observations.py` de mineplanner para afinar el port de `EdgeParameter`/observaciones (fuera de scope v0.2, no urge).

> Nota de higiene documental: `README_CLAUDE.md` establece que los ADR no llevan sección "Abiertas" (el backlog de diseño vive en `ROADMAP.md`). Esta sección se mantiene por continuidad histórica, pero las entradas 9–10 están duplicadas en `ROADMAP.md § Decisiones abiertas`, que es la fuente de verdad. Candidata a limpiarse en una pasada futura.

---

## 6. Próximos pasos

1. Cerrar la forma concreta de la composición trivial de `ScalarRate` y del adaptador `as_physics_kwargs()` — es lo único que queda entre el diseño cerrado y la implementación de la capa `Rate`/`Fluid`.
2. Diseño detallado del caso demo: red sintética de wellfield, inputs, qué muestra el notebook — esto define los tests de aceptación y tensiona las firmas contra un uso real. **Sin código todavía.**
3. Firmas públicas restantes: los dos `Protocol` de loss y `Result` — a nivel de interfaz (docstrings + type hints), no de implementación.
4. Recién ahí: implementar, portando en el orden del mapa de rescate (§3), commiteando de a piezas chicas contra `main` limpia.
