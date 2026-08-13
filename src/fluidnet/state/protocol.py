"""``StateModel`` / ``BoundState`` — the neutral state protocol.

Domain-neutral by design (``CLAUDE.md`` #18): transforms the *across*
variable of a node (plus any prescribed fields) into whatever a
``gradient_fn`` needs. Fluids: ``(composition, T, P) -> (rho, mu, beta,
sigma)``. Electrical AC demo: ``(V) -> impedance``. Same shape either way.

Two methods, two binding times (#28, #30):

    StateModel.bind(*, composition, **fields) -> BoundState   # 1x per edge
    BoundState.__call__(*, x, across) -> State                 # 1x per step

``temperature`` and every other physical field live on the concrete
implementation's ``bind`` signature, never here (#18). ``x`` is the only
axis-neutral argument allowed in ``__call__`` (#30).
"""
