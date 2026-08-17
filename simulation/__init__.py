"""Layers 2 & 3: simulated world, bridge, engine, training, registry, metrics, and dashboard glue.

This package owns everything that is "new" relative to the reference paper
implementation: the taxi/SPKLU/order world, the engine that advances time and
triggers negotiations, the bridge that is the sole module allowed to know
both the world and the negotiation mechanism, and supporting infrastructure
(metrics, registry, training orchestration, configuration).
"""
