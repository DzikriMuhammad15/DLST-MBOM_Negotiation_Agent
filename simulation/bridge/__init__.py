"""The bridge: the sole "bilingual" module that knows both the world and the negotiation mechanism.

Responsible for snapshotting the world, building per-side agent views,
constructing the action library / utility functions / domain spec, running
negotiations, and applying deal/no-deal/partial/emergency outcomes back to
the world. See ``.claude/project_instruction/05-bridge-dan-negosiasi/``.
"""
