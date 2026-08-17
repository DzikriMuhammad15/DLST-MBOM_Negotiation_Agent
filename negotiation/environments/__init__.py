"""NegMAS SAO mechanism wiring (CustomStartSAOMechanism, NegMASBilateralEnv).

This package wraps the NegMAS ``SAOMechanism`` so that negotiation outcomes
are bundle indices into an externally supplied ``action_library``, and so
that the fleet agent is always added first (first proposer).
"""
