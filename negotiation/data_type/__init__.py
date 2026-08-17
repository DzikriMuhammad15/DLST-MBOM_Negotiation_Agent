"""Negotiation data types (Step, Episode) shared by agents and trainers.

These types carry per-step and per-episode information used to reconstruct
action distributions consistently between sampling time and training-update
time (important for PPO's recomputed log-probabilities).
"""
