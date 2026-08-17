"""MBOM recursive-imagination rollout engines (AS-IS).

Two engines (one per phase) that run *imagined* negotiations to estimate the best
opponent response, used by MBOM recursive imagination. They continue the real
negotiation **from its mid-point** via ``CustomStartSAOMechanism(start_step=...)``
(the rollout resumes from the current ``relative_time`` rather than step 0). The
opponent is modeled by an estimated utility callable wrapped into a NegMAS
:class:`~negotiation.environments.negmas_env.BundleUFun` so the
:class:`~negotiation.agents.iop_agent.RandomIOPAgent` used as the imagined
opponent negotiates on the *opponent's* estimated utility, not the user's.

All rollout / discounting formulas are unchanged from the paper code; only the
imports / outcome representation are adapted to the new bundle-index layer
(outcomes are ``(idx,)`` tuples; ``decode_action`` is index-direct).
"""

import random
from typing import Any, Callable, List, Optional, Sequence

from negotiation.environments.negmas_env import (
    BundleUFun,
    DomainSpec,
    NegMASBilateralEnv,
    build_bundle_issues,
)


def refresh_rollout_engines(agent: Any) -> None:
    """Suntikkan konteks dunia per-negosiasi (C3) ke kedua rollout engine agen.

    FIX E-1: engine dibangun saat konstruksi agen ketika ``domain_spec`` masih
    ``None`` dan ``action_library`` masih kosong; ``configure_negotiation`` hanya
    memperbarui atribut agen sehingga engine selamanya melihat library kosong
    (propose-recursion no-op, respond-recursion ``randint(0, -1)`` crash).
    Fungsi ini dipanggil dari ``configure_negotiation`` agen MBOM/DLST-MBOM agar
    engine memakai library nyata + ``DomainSpec`` minimal yang konsisten.
    """
    lib = list(getattr(agent, "action_library", []) or [])
    if not lib:
        return
    spec = getattr(agent, "domain_spec", None)
    issues = getattr(spec, "issues", None) if spec is not None else None
    n_issue_vals = None
    if issues:
        try:
            n_issue_vals = issues[0].cardinality
        except Exception:
            n_issue_vals = None
    if spec is None or not issues or n_issue_vals != len(lib):
        ufun = getattr(agent, "_world_ufun", None) or (lambda b: 0.0)
        spec = DomainSpec(
            issues=build_bundle_issues(len(lib)),
            action_library=lib,
            ufun_fleet=ufun,
            ufun_spklu=ufun,
        )
    for engine in (
        getattr(agent, "rollout_engine_propose", None),
        getattr(agent, "rollout_engine_respond", None),
    ):
        if engine is not None:
            engine.action_library = lib
            engine.domain_spec = spec


def _opponent_ufun(opponent_util_fn: Callable[[Any], float], action_library: Sequence[Any]) -> BundleUFun:
    """Wrap an estimated opponent utility callable into a NegMAS ``BundleUFun``.

    ``opponent_util_fn`` is called with a **bundle** (``action_library[idx]``); the
    ``BundleUFun`` decodes ``(idx,)`` and delegates, so the imagined opponent uses
    the opponent's estimated utility (not the user's).
    """
    return BundleUFun(action_library, opponent_util_fn)


def _start_step(relative_time: float, max_steps: int) -> int:
    """Resume step for a rollout starting at ``relative_time`` (AS-IS clamp)."""
    start_step = int(round(float(relative_time) * (max_steps + 1))) - 1
    return max(0, min(start_step, max_steps))


class NegotiationRolloutEnginePropose:
    """Imagined-negotiation engine for the IOP propose pipeline (AS-IS).

    Example:
        >>> NegotiationRolloutEnginePropose.__name__
        'NegotiationRolloutEnginePropose'
    """

    def __init__(
        self,
        domain_spec: Any,
        action_library: List[Any],
        issue_names: List[str],
        gamma: float = 0.99,
        rollout_k: int = 2,
        max_steps: int = 10,
    ) -> None:
        self.action_library = action_library
        self.issue_names = issue_names
        self.gamma = gamma
        self.rollout_k = rollout_k
        self.domain_spec = domain_spec
        self.max_steps = max_steps

    def build_opponent_state(self, user_offer: Any, relative_time: float = 0.0) -> dict:
        """Synthesize the opponent's state dict from the user's offer (AS-IS)."""
        return {"current_offer": user_offer, "relative_time": float(relative_time)}

    def compute_best_response(
        self,
        initial_user_state_basis: Any,
        opponent_util_fn: Callable[[Any], float],
        opponent_value_fn: Callable[[Any], float],
        user_agent: Any,
        relative_time: float,
        n_uniform_sample: int = 5,
    ) -> int:
        """Return the opponent's best propose response index over the library (AS-IS).

        FIX E-2: tawaran awal user di dalam imajinasi diambil lewat
        ``user_agent.rollout_initial_offer(...)`` yang polimorfik — agen diskret
        men-sample indeks dari kepala propose-nya, agen hybrid DLST menjalankan
        alur propose DLST lengkap (aktor DDPG -> strategy template -> indeks).
        Unpacking 4-tuple lama selalu ``ValueError`` untuk hybrid (aktor DLST
        mengembalikan vektor kontinu 14-dim).
        """
        best_action_idx = 0
        best_value = float("-inf")

        user_offer = user_agent.rollout_initial_offer(initial_user_state_basis, relative_time)
        opponent_state = self.build_opponent_state(user_offer, relative_time=relative_time)

        for candidate_idx in range(len(self.action_library)):
            for _ in range(n_uniform_sample):
                agent_clone = user_agent.clone_for_rollout()
                rollout_value = self._simulate_rollout(
                    user_agent_clone=agent_clone,
                    opponent_state=opponent_state,
                    opponent_first_action=candidate_idx,
                    opponent_util_fn=opponent_util_fn,
                    opponent_value_fn=opponent_value_fn,
                    relative_time=relative_time,
                )
                if rollout_value > best_value:
                    best_value = rollout_value
                    best_action_idx = candidate_idx
        return best_action_idx

    def _simulate_rollout(
        self,
        user_agent_clone: Any,
        opponent_state: dict,
        opponent_first_action: int,
        opponent_util_fn: Callable[[Any], float],
        opponent_value_fn: Optional[Callable[[Any], float]] = None,
        relative_time: float = 0.0,
    ) -> float:
        """Run one imagined rollout and return its discounted value (AS-IS)."""
        # Local import avoids a circular import (iop_agent imports nothing here).
        from negotiation.agents.iop_agent import RandomIOPAgent

        if self.rollout_k == 0:
            return 0.0

        cumulative = 0.0
        last_state_for_value = opponent_state

        opponent_initial_offer = user_agent_clone.make_offer(opponent_first_action)
        first_reward = float(opponent_util_fn(self.action_library[opponent_first_action]))
        cumulative += first_reward

        ufun_for_opponent = _opponent_ufun(opponent_util_fn, self.action_library)
        opponent_agent = RandomIOPAgent(
            name="opponent_agent_rollout",
            ufun=ufun_for_opponent,
            domain_spec=self.domain_spec,
        )
        opponent_agent.configure_negotiation(
            {}, None, None, self.action_library, opponent_util_fn,
            user_agent_clone.utility_min, user_agent_clone.utility_max,
        )

        env = NegMASBilateralEnv(domain_spec=self.domain_spec, max_steps=self.max_steps)
        start_step = _start_step(relative_time, self.max_steps)

        _, _ = env.run(
            user_agent_clone,
            opponent_agent,
            first_actor="a",
            start_with="respond",
            initial_offer={"bundle_index": opponent_first_action},
            skip_agent_reset=False,
            start_step=start_step,
        )

        episode_payload = opponent_agent.build_episode_payload_propose(None)

        t = 1
        k = 1
        for step in episode_payload.steps:
            if k == self.rollout_k:
                break
            reward = float(step.utility or 0.0) * (self.gamma ** t)
            cumulative += reward
            last_state_for_value = step.next_state if step.next_state is not None else step.state
            t += 1
            k += 1

        if opponent_value_fn is not None:
            value_state_basis = user_agent_clone._state_basis_for_iop(last_state_for_value)
            value_estimate = float(opponent_value_fn(value_state_basis))
            cumulative += value_estimate * (self.gamma ** t)

        return cumulative


class NegotiationRolloutEngineRespond:
    """Imagined-negotiation engine for the IOP respond pipeline (AS-IS).

    Example:
        >>> NegotiationRolloutEngineRespond.__name__
        'NegotiationRolloutEngineRespond'
    """

    def __init__(
        self,
        domain_spec: Any,
        action_library: List[Any],
        issue_names: List[str],
        gamma: float = 0.99,
        rollout_k: int = 2,
        max_steps: int = 10,
    ) -> None:
        self.action_library = action_library
        self.issue_names = issue_names
        self.gamma = gamma
        self.rollout_k = rollout_k
        self.domain_spec = domain_spec
        self.max_steps = max_steps

    def build_opponent_state(self, user_offer: Any, relative_time: float = 0.0) -> dict:
        """Synthesize the opponent's state dict from the user's offer (AS-IS)."""
        return {"current_offer": user_offer, "relative_time": float(relative_time)}

    def compute_best_response(
        self,
        initial_user_state_basis: Any,
        opponent_util_fn: Callable[[Any], float],
        opponent_value_fn: Callable[[Any], float],
        user_agent: Any,
        relative_time: float,
        n_uniform_sample: int = 5,
    ) -> int:
        """Return the opponent's best respond action (0 accept / 1 reject) (AS-IS).

        FIX E-2: lihat catatan pada
        :meth:`NegotiationRolloutEnginePropose.compute_best_response`.
        """
        best_action_idx = 0
        best_value = float("-inf")

        user_offer = user_agent.rollout_initial_offer(initial_user_state_basis, relative_time)
        opponent_state = self.build_opponent_state(user_offer, relative_time=relative_time)

        for candidate_idx in range(2):  # 0 accept, 1 reject
            for _ in range(n_uniform_sample):
                agent_clone = user_agent.clone_for_rollout()
                rollout_value = self._simulate_rollout(
                    user_agent_clone=agent_clone,
                    opponent_state=opponent_state,
                    opponent_first_action=candidate_idx,
                    opponent_util_fn=opponent_util_fn,
                    opponent_value_fn=opponent_value_fn,
                    relative_time=relative_time,
                )
                if rollout_value > best_value:
                    best_value = rollout_value
                    best_action_idx = candidate_idx
        return best_action_idx

    def _simulate_rollout(
        self,
        user_agent_clone: Any,
        opponent_state: dict,
        opponent_first_action: int,
        opponent_util_fn: Callable[[Any], float],
        opponent_value_fn: Optional[Callable[[Any], float]] = None,
        relative_time: float = 0.0,
    ) -> float:
        """Run one imagined respond rollout and return its discounted value (AS-IS)."""
        from negotiation.agents.iop_agent import RandomIOPAgent

        if self.rollout_k == 0:
            return 0.0

        cumulative = 0.0
        current_offer = opponent_state.get("current_offer")
        last_state_for_value = opponent_state

        if opponent_first_action == 0:
            # Opponent immediately accepts the user's offer.
            idx = user_agent_clone._offer_to_action_idx(current_offer)
            utility = float(opponent_util_fn(self.action_library[idx])) if 0 <= idx < len(self.action_library) else 0.0
            return cumulative + utility

        rand_idx = random.randint(0, len(self.action_library) - 1)
        ufun_for_opponent = _opponent_ufun(opponent_util_fn, self.action_library)
        opponent_agent = RandomIOPAgent(
            name="opponent_agent_rollout",
            ufun=ufun_for_opponent,
            domain_spec=self.domain_spec,
        )
        opponent_agent.configure_negotiation(
            {}, None, None, self.action_library, opponent_util_fn,
            user_agent_clone.utility_min, user_agent_clone.utility_max,
        )

        env = NegMASBilateralEnv(domain_spec=self.domain_spec, max_steps=self.max_steps)
        start_step = _start_step(relative_time, self.max_steps)

        _, _ = env.run(
            user_agent_clone,
            opponent_agent,
            first_actor="a",
            start_with="respond",
            initial_offer={"bundle_index": rand_idx},
            skip_agent_reset=False,
            start_step=start_step,
        )

        episode_payload = opponent_agent.build_episode_payload_respond(None)

        t = 1
        k = 1
        for step in episode_payload.steps:
            if k == self.rollout_k:
                break
            reward = float(step.utility or 0.0) * (self.gamma ** t)
            cumulative += reward
            last_state_for_value = step.next_state if step.next_state is not None else step.state
            t += 1
            k += 1

        if opponent_value_fn is not None:
            value_state_basis = user_agent_clone._state_basis_for_iop(last_state_for_value)
            value_estimate = float(opponent_value_fn(value_state_basis))
            cumulative += value_estimate * (self.gamma ** t)

        return cumulative
