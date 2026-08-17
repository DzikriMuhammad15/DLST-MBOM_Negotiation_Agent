# -*- coding: utf-8 -*-
"""Registrasi ADITIF tipe agen "GreedyMH" ke factory lama (tanpa ubah kode lama).

``agent_factory.build_agent`` (dan jalur rebuild per-episode
``rebuild_agent_shell_with_trainers``) me-resolve ``construct_agent_shell`` &
``_build_trainers`` lewat global modul ``agent_factory`` — sehingga wrap pada
kedua simbol modul itu cukup untuk memperkenalkan tipe baru ke SEMUA jalur
pembangunan agen (run_pipeline.build_side_agent, rebuild per-episode) tanpa
menyentuh file lama.
"""
from __future__ import annotations

from negotiation.agents import agent_factory

from greedy_agent import AGENT_GREEDY, GreedyNegotiator

_orig_shell = agent_factory.construct_agent_shell
_orig_trainers = agent_factory._build_trainers


def install_greedy_agent_type() -> None:
    """Pasang tipe "GreedyMH" (idempoten)."""

    def shell(cfg):
        if cfg.agent_type == AGENT_GREEDY:
            return GreedyNegotiator(name=cfg.name, ufun=cfg.ufun,
                                    domain_spec=cfg.domain_spec)
        return _orig_shell(cfg)

    def trainers(cfg):
        if cfg.agent_type == AGENT_GREEDY:
            return {}  # non-learning — tanpa trainer (pola Rule/RandomIOP)
        return _orig_trainers(cfg)

    agent_factory.construct_agent_shell = shell
    agent_factory._build_trainers = trainers
    agent_factory.ALL_AGENT_TYPES = set(agent_factory.ALL_AGENT_TYPES) | {AGENT_GREEDY}
    print(f"[metaheuristic] tipe agen '{AGENT_GREEDY}' terdaftar (aditif)")
