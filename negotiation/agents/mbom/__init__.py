"""MBOM opponent-modeling components (IOP model, recursive imagination rollout, Bayesian mixer).

These modules implement the Model-Based Opponent Modeling pipeline as-is:
predicting the opponent's action distribution (IOP), running recursive
imagination rollouts, and mixing multiple IOP predictions via a Bayesian
mixer (level-k reasoning).
"""
