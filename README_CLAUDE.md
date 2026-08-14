### Verificación

Dos niveles, deliberadamente asimétricos:

- **Automático, en cada commit** (`pre-commit`): `ruff check --fix`,
  `ruff format`, `lint-imports`. Son reglas: si fallan, el commit no entra.
  Requiere `pre-commit install` una vez por clon.
- **Manual, al cerrar sesión**: `python -m mypy` y `pytest`. Son correctitud, no
  reglas; corren completos y no bloquean commits intermedios.

No hay CI todavía: la verificación depende del operador local. Agendado en
`ROADMAP §v1.0` (bloque de infraestructura de repositorio).
