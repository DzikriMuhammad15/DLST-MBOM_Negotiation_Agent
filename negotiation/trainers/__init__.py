"""Trainer implementations (PolicyGradient, PPO, DDPG, IOP) with exact paper loss formulas.

Trainers are shared across both PADDED and ENCODER modes; only the batching
plumbing (mask vs. candidate-feature padding) differs per mode, not the loss
or update logic itself. See
``.claude/project_instruction/03-adaptasi-baru/03-05-trainer-plumbing.md``.
"""
