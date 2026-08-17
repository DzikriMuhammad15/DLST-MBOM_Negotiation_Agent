"""Mode components (PADDED / ENCODER) for dynamic-dimension state and action spaces.

These sub-modules (set encoder / DeepSets pooling, padded front-end, and the
per-candidate score head) are attached as sub-modules of each agent's
actor/critic networks, so they are saved and loaded together with the rest of
the network's ``state_dict``. See
``.claude/project_instruction/03-adaptasi-baru/03-01-dua-mode.md`` and
``03-02-encoder-padding-detail.md``.
"""
