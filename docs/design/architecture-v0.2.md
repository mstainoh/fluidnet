# fluidnet — Arquitectura v0.2 (ADR)

> Archivo nuevo — sólo contiene la sección §2.1bis (`StateModel`), cerrada en
> la sesión de diseño 2026-08-10. El resto del documento (mapa de rescate
> desde mineplanner, §1–§2, §3 y siguientes, referenciados desde `CLAUDE.md`
> y `ROADMAP.md`) todavía no existe en el repo y hay que agregarlo aparte.

## §2.1bis — `StateModel`: protocolo neutro y ligado por eje

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
StateModel.bind(*, composition, **fields) -> BoundState    # 1× por eje
BoundState.__call__(*, x, across) -> State                 # 1× por evaluación
```

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

El perfil queda capturado en el closure del `BoundState`, y la firma de
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
StateModel → bind → BoundState → (x, across) → State
                                                  → as_physics_kwargs()
rate.as_physics_kwargs() ───────────────────────→ gradient_fn ← edge_attrs
```

**Regla general que ordena la capa**: *todo lo que no depende de `x` se liga
antes del integrador; lo que depende de `x` entra por el `bound`.* Es el
mismo criterio detrás del hoisting de `rate.as_physics_kwargs()` y de las dos
clases hermanas — tres decisiones que parecían sueltas y son una sola.

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

**Sobre la vectorización.** La integración en `x` no es vectorizable: el paso
`n+1` depende del `n`. El eje vectorizable es el de **escenarios** — `y0`
como array de N condiciones iniciales independientes, N ODEs desacopladas en
una pasada, una evaluación vectorizada del `rhs` por paso en vez de N. Es la
forma del loop interno del fitting de v0.5, y el motivo por el que las
implementaciones de `StateModel` deben ser array-safe.
