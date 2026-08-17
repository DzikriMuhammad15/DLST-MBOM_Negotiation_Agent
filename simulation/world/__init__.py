"""World entities, ground-truth state, state machine, idle logic, order generation, routing, and fare.

This package implements the simulated ride-hailing/EV-charging world
described in ``.claude/project_instruction/01-dunia-simulasi/``. It has no
knowledge of the negotiation mechanism; only ``simulation.bridge`` is allowed
to bridge between this package and ``negotiation``.
"""
