"""Layer 1: negotiation mechanism (NegMAS SAO + Reinforcement Learning).

This package contains the negotiation mechanism ported as-is from the reference
paper implementation, plus the only internal adaptation allowed by the project
specification: two input representation modes (PADDED / ENCODER) and the
bundle/action_library abstraction. This package never imports from
``simulation.world`` (layer separation invariant, see
``.claude/project_instruction/00-overview/00-05``).
"""
