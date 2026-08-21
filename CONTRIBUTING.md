## Automated verification

This project follows a very simple principle: **every declared claim is
verified, or it is removed.** If `pyproject.toml` declares Python 3.10 support,
a job proves it. If the documentation draws architectural layers, a contract
checks them. *Key idea: the alternative to verification is not "trust" — it is asserting
without evidence*.

Code writing is based on four pillars:

1. **Syntactic coherence** — no dead variables or unused imports, and
   protection against the language's own traps (`flake8-bugbear`): mutable
   defaults, loop closures that capture the variable instead of its value.
   Checked by `ruff`, which parses each file into an AST and applies rules to
   it. `ruff` runs in milliseconds, so this verification also runs as a pre-commit hook.

2. **Functional coherence** — explicit type signatures on function and class
   arguments and returns, no `Any` returns, type consistency across chained
   calls. Checked by `mypy --strict`, which follows types through the whole
   call graph. Note that the value here is not catching a `TypeError`: `fluidnet`'s
   architecture is built on `Protocol`s (`StateModel`, `AlgebraicLoss` /
   `IntegralLoss`). With static checking, `Protocol` is not a comment:
   it is a contract verified at every point of use. `strict` flag is
   explicitly used to forbid the *absence* of typing, not just inconsistency.

3. **Architectural coherence** — sibling layers do not import from each other,
   lower layers do not import from higher ones (not even indirectly), and
   layer zero stays free of network and I/O dependencies (`networkx`,
   `pandas`). Checked by `import-linter`, which ignores code and types
   entirely and inspects the package's import topology. The contracts in
   `pyproject.toml` are the executable form of the layer diagram.

4. **Expected behaviour** — numerical correctness against literature golden
   values, structural invariants, and cross-validation against independent
   implementations. Checked by `pytest`.

Style is not a separate pillar: PEP 8 conformance (`E`), import ordering (`I`)
and modern-syntax migration (`UP`) are enforced in the same `ruff` pass.

All rules live in `pyproject.toml`. Every push or merge to `dev` or `main`
runs these checks on a clean VM via GitHub Actions, across the full
supported Python matrix. **A failing check is treated as a bug and fixed —
even when the fix is a single comment line.**

### Current scope

The layer contracts verify the modules that exist today (`rate`, `state`,
`physics`, `_types`); violations in any direction are caught. The upper layers
(`losses`, `network`, `solvers`) are declared as optional in the contract and
will be enforced once those packages exist. For that part of the hierarchy the
contract is preventive, not a fulfilled verification.

Standards get harder to hold as a project grows. This infrastructure was put
in place early, before it was strictly needed, and is carried through the
whole development — not retrofitted before a release.

## Reporting a problem

Bugs, unexpected numerical results and documentation errors all go to
[Issues](https://github.com/mstainoh/fluidnet/issues). A useful report
includes the `fluidnet` version, the Python version, and a minimal snippet
that reproduces the behaviour. If the problem is numerical rather than a
crash, state what you expected and where the reference value comes from —
a textbook example, field data, or another implementation.

## Getting help

Questions about how to model a particular network, or whether a use case
fits the library's scope, are welcome as Issues too. There is no separate
forum: an Issue is the right place, and the answer stays searchable for
whoever asks next.

## Contributing code

```bash
git clone https://github.com/mstainoh/fluidnet.git
cd fluidnet
pip install -e ".[dev]"
pre-commit install
```

Branch from `dev` — `main` only ever takes tagged releases. Before opening
a pull request, run `pre-commit run --all-files`, `python -m mypy` and
`pytest` locally; CI runs the same checks on a clean VM, so anything
failing locally will fail there. Pull requests target `dev`.

`fluids` (ChEDL) is a development dependency used as a cross-validation
oracle in tests. Locally those tests may skip if it is not installed; in CI
a dedicated guard step forbids it — a skip there is a configuration failure,
not a pass.

This project is developed design-first: architectural decisions are closed
and documented before code is written. Small fixes, tests and documentation
can go straight to a pull request. Anything that changes a protocol, a layer
boundary or a public signature is worth raising as an Issue first — not as a
barrier, but because the decision record matters as much as the diff.

### AI-assisted development

Claude (Anthropic) is used as an implementation tool in this project.
Architectural decisions are made and recorded by the maintainer; the
assistant executes against closed specifications. This is precisely why the
verification described above is not optional: assisted code generation makes
automated checking a requirement, not a nicety.

Commits where the assistant generated a substantial portion of the diff carry
an `Assisted-by:` trailer naming the tool. This records assistance, not
authorship: responsibility for every line rests with the human who signs the
commit. Contributors using similar tooling are encouraged to do the same —
the commit history of this project is meant to be readable as a record of how
the code was produced, and a history that hides that is not much of a record. Commits before 2026-08-21 use a `Co-authored-by:` trailer
instead; the convention changed, the meaning did not.

The design documents in this repository — `CLAUDE.md` for closed decisions,
`ROADMAP.md` for the open design backlog, and `docs/design/` for the
architecture record — are written to be read by both humans and assistants.
Anyone cloning the repo, with or without AI tooling, should be able to
reconstruct why a signature looks the way it does without asking the
maintainer — whether to add a feature or to fork.

## What to contribute

This library lifts pointwise differential equations onto a graph structure.
By design, it is **not** a correlation repository. New differential equations
(loss functions) are very welcome — and you can equally write your own
against the protocol without upstreaming them. Example applications on
different problems are also welcome.

The interface is deliberately physics-agnostic: the protocol is meant to
accommodate parameters and variables that differ between models — holdup
adjustments, interfacial tension models, correlation tuning. If your equation
or problem does not fit the protocol, or fitting it requires disproportionate
work on your side, that is worth an Issue: it usually means the protocol is
too narrow, which is useful information.