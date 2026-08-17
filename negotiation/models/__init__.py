"""Neural network models used by negotiator agents (policy, value, DDPG, IOP).

Network architectures and forward-pass formulas are ported as-is from the
reference implementation. The ``encoders`` sub-package adds the only new
components: mode-specific front-ends (PADDED / ENCODER) for dynamic-size
state and action spaces.
"""
