"""Save / load trained agents + their signature under the registry (``06-02``).

:class:`AgentRegistry` persists one negotiator as a **versioned artifact
directory** under ``training.registry_path`` containing:

* one ``*.pt`` checkpoint **per attached trainer** (``propose`` / ``respond`` /
  DLST ``threshold`` + any MBOM IOP trainers), written by the trainer's own
  ``save_checkpoint`` — this captures every model ``state_dict`` (the **encoder
  rides along as a sub-module**, Gotcha G5) **and** the optimizer / target-network
  state (``02-08``);
* ``signature.json`` — the versioned :class:`~simulation.registry.signature.Signature`
  (Contract C7);
* ``agent_meta.json`` — the agent + trainer + mode configuration so the artifact
  is self-describing (``06-02``).

Loading is **into an already-built agent** of the same type (re-built by the
caller via :func:`negotiation.agents.agent_factory.build_agent` for the current
config). The registry first compares the stored signature against the live agent
signature and raises :class:`SignatureMismatchError` (naming the field) on any
structural mismatch (``06-02`` check #1). It then restores each trainer's model
``state_dict`` (incl. encoder) **and** optimizer / target state into the
already-constructed networks — sidestepping the empty ``MODEL_REGISTRY`` in the
trainers' own ``load_checkpoint`` (the factory builds every model with an empty
``config``, so direct ``state_dict`` injection into the live objects is the
correct round-trip).

This module imports :mod:`simulation.config`,
:mod:`simulation.registry.signature`, and torch (for ``torch.load`` of the
trainer checkpoints). It does not import the bridge or the world.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

try:
    import torch
except Exception:  # pragma: no cover - torch always present at runtime
    torch = None  # type: ignore

from simulation.registry.signature import (
    Signature,
    SignatureMismatchError,
    build_signature,
)


# Canonical trainer-role keys an agent may expose (single or list-valued).
_SINGLE_ROLES: Tuple[str, ...] = (
    "propose_trainer",
    "respond_trainer",
    "threshold_trainer",
)
_LIST_ROLES: Tuple[str, ...] = (
    "iop_trainers_propose",
    "iop_trainers_respond",
)
_SIGNATURE_FILE: str = "signature.json"
_META_FILE: str = "agent_meta.json"


def _enumerate_trainers(agent: Any) -> List[Tuple[str, Any]]:
    """Return ``[(checkpoint_key, trainer), ...]`` for every trainer on an agent.

    Single-valued roles keep their role name; list-valued roles get an indexed
    suffix (e.g. ``iop_trainers_propose__0``). Trainers without a
    ``save_checkpoint`` are skipped (baseline agents have none).

    Example:
        >>> class A:
        ...     trainers = {"propose_trainer": object(),
        ...                 "iop_trainers_propose": [object(), object()]}
        >>> keys = [k for k, _ in _enumerate_trainers(A())]
        >>> sorted(keys)
        ['iop_trainers_propose__0', 'iop_trainers_propose__1', 'propose_trainer']
    """

    trainers_map = getattr(agent, "trainers", None) or {}
    out: List[Tuple[str, Any]] = []
    for role in _SINGLE_ROLES:
        trainer = trainers_map.get(role)
        if trainer is not None:
            out.append((role, trainer))
    for role in _LIST_ROLES:
        items = trainers_map.get(role) or []
        for i, trainer in enumerate(items):
            if trainer is not None:
                out.append((f"{role}__{i}", trainer))
    return out


def _load_state_into_trainer(trainer: Any, checkpoint: Dict[str, Any], device: str) -> None:
    """Inject a checkpoint's ``state_dict``s into an existing trainer's models.

    Loads each model's ``state_dict`` (incl. encoder sub-modules, G5) into the
    live model object, each optimizer's state into the live optimizer, and any
    DDPG ``target_models`` into the live targets — without re-instantiating from
    ``MODEL_REGISTRY`` (which the factory leaves empty).

    Args:
        trainer: An already-built trainer with attached ``models`` / ``optimizers``.
        checkpoint: The dict produced by the trainer's ``save_checkpoint``.
        device: Torch device string for ``map_location`` consistency.

    Example:
        >>> _load_state_into_trainer.__name__
        '_load_state_into_trainer'
    """

    models = getattr(trainer, "models", None) or {}
    optimizers = getattr(trainer, "optimizers", None) or {}
    target_models = getattr(trainer, "target_models", None) or {}

    for key, info in (checkpoint.get("models", {}) or {}).items():
        entry = models.get(key)
        state = info.get("state_dict") if isinstance(info, dict) else None
        if entry is not None and state is not None:
            entry["model"].load_state_dict(state)
    for key, info in (checkpoint.get("optimizers", {}) or {}).items():
        optimizer = optimizers.get(key)
        state = info.get("state_dict") if isinstance(info, dict) else None
        if optimizer is not None and state is not None:
            optimizer.load_state_dict(state)
    for key, state in (checkpoint.get("target_models", {}) or {}).items():
        target = target_models.get(key)
        if target is not None and state is not None:
            target.load_state_dict(state)


def _side_config_to_dict(config: Any, side: str) -> Dict[str, Any]:
    """Serialize the agent/trainer/mode config for one side (``06-02`` meta).

    Example:
        >>> from simulation.config.default import default_config
        >>> _side_config_to_dict(default_config(), "fleet")["agent_type"]
        'RLNegotiator'
    """

    side_cfg = getattr(config.agent, side)
    enrichment = getattr(side_cfg, "critic_enrichment", None)
    return {
        "agent_type": str(side_cfg.agent_type),
        "trainer_type": str(side_cfg.trainer_type),
        "mode": str(side_cfg.mode),
        "critic_enrichment": {
            "critic_sees_candidate_pool": bool(
                getattr(enrichment, "critic_sees_candidate_pool", False)
            ),
            "critic_privileged_info": bool(
                getattr(enrichment, "critic_privileged_info", False)
            ),
        },
    }


class AgentRegistry:
    """Save / load trained agents + signatures under ``training.registry_path``.

    Parameters
    ----------
    registry_path:
        Root directory for artifacts. Defaults to ``data/registry``; pass
        ``config.training.registry_path`` to honor the config (G14).
    device:
        Torch device used for ``torch.load`` (``"cpu"`` by default).

    Example:
        >>> reg = AgentRegistry(registry_path="data/registry")
        >>> reg.registry_path.endswith("registry")
        True
    """

    def __init__(self, registry_path: str = "data/registry", device: str = "cpu") -> None:
        """Create a registry rooted at ``registry_path`` (created on first save).

        Example:
            >>> AgentRegistry("data/registry").device
            'cpu'
        """

        self.registry_path = str(registry_path)
        self.device = str(device)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save_agent(
        self,
        agent: Any,
        signature: Signature,
        agent_id: str,
        version: Optional[str] = None,
        agent_meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Persist an agent's trainer checkpoints + signature + meta (``06-02``).

        Each trainer is checkpointed via its own ``save_checkpoint`` (state_dict
        incl. encoder G5 + optimizer / target state), alongside a ``signature.json``
        and ``agent_meta.json``. The artifact lives at
        ``<registry_path>/<agent_id>/<version>/``; ``version`` defaults to a
        UTC timestamp so repeated saves are versioned (``06-02``).

        Args:
            agent: The negotiator to persist.
            signature: Its :class:`Signature` (Contract C7).
            agent_id: Logical id (e.g. ``"fleet"`` / a model name).
            version: Optional version label; a timestamp is used if omitted.
            agent_meta: Optional extra config dict stored verbatim into the meta.

        Returns:
            The absolute artifact directory path.

        Example:
            >>> AgentRegistry.save_agent.__qualname__
            'AgentRegistry.save_agent'
        """

        if torch is None:  # pragma: no cover
            raise RuntimeError("torch is required to save agent checkpoints.")
        version = version or time.strftime("%Y%m%dT%H%M%S", time.gmtime())
        artifact_dir = os.path.join(self.registry_path, agent_id, version)
        os.makedirs(artifact_dir, exist_ok=True)

        trainer_keys: List[str] = []
        for key, trainer in _enumerate_trainers(agent):
            save = getattr(trainer, "save_checkpoint", None)
            if not callable(save):
                continue
            save(os.path.join(artifact_dir, f"{key}.pt"))
            trainer_keys.append(key)

        with open(os.path.join(artifact_dir, _SIGNATURE_FILE), "w", encoding="utf-8") as fh:
            json.dump(signature.to_dict(), fh, indent=2)

        meta: Dict[str, Any] = {
            "agent_id": agent_id,
            "version": version,
            "agent_type": type(agent).__name__,
            "mode": getattr(agent, "mode", None),
            "trainer_keys": trainer_keys,
        }
        if agent_meta:
            meta["config"] = agent_meta
        with open(os.path.join(artifact_dir, _META_FILE), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)

        return os.path.abspath(artifact_dir)

    def save_side(
        self, agent: Any, config: Any, side: str, version: Optional[str] = None
    ) -> str:
        """Convenience: build the side's signature from config + feature dims, save.

        Imports the bridge ``feature_builder`` lazily so the registry module does
        not hard-depend on the bridge at import time. The artifact id is the side
        name.

        Args:
            agent: The side's negotiator.
            config: The :class:`SimulationConfig`.
            side: ``"fleet"`` / ``"spklu"``.
            version: Optional version label.

        Returns:
            The artifact directory path.

        Example:
            >>> AgentRegistry.save_side.__qualname__
            'AgentRegistry.save_side'
        """

        from simulation.bridge.feature_builder import feature_dims

        signature = build_signature(config, side, feature_dims(config, side))
        meta = _side_config_to_dict(config, side)
        return self.save_agent(agent, signature, agent_id=side, version=version, agent_meta=meta)

    # ------------------------------------------------------------------
    # Load (with signature check — Contract C7)
    # ------------------------------------------------------------------

    def load_signature(self, agent_id: str, version: str) -> Signature:
        """Read the stored :class:`Signature` for an artifact (no model load).

        Example:
            >>> AgentRegistry.load_signature.__qualname__
            'AgentRegistry.load_signature'
        """

        artifact_dir = os.path.join(self.registry_path, agent_id, version)
        with open(os.path.join(artifact_dir, _SIGNATURE_FILE), "r", encoding="utf-8") as fh:
            return Signature.from_dict(json.load(fh))

    def load_into_agent(
        self,
        agent: Any,
        agent_id: str,
        version: str,
        current_signature: Signature,
        load_optimizer: bool = True,
    ) -> Signature:
        """Validate signatures, then restore trainer state into ``agent`` (``06-02``).

        First compares the stored signature against ``current_signature`` (the
        live agent's signature for the current config). On any structural mismatch
        a :class:`SignatureMismatchError` is raised naming the field (Contract C7).
        On success each trainer's checkpoint is loaded straight into the
        already-built networks (state_dict incl. encoder G5) + optimizer / target
        state.

        Args:
            agent: The freshly-built agent to load weights into (same type/dims).
            agent_id: The artifact id.
            version: The artifact version.
            current_signature: The live signature to validate against.
            load_optimizer: Whether to restore optimizer state (default True).

        Returns:
            The stored (now-validated) :class:`Signature`.

        Raises:
            SignatureMismatchError: If the stored and current signatures differ.

        Example:
            >>> AgentRegistry.load_into_agent.__qualname__
            'AgentRegistry.load_into_agent'
        """

        if torch is None:  # pragma: no cover
            raise RuntimeError("torch is required to load agent checkpoints.")
        stored = self.load_signature(agent_id, version)
        # Contract C7: the stored agent's signature must match the current config.
        stored.assert_compatible(current_signature)

        artifact_dir = os.path.join(self.registry_path, agent_id, version)
        for key, trainer in _enumerate_trainers(agent):
            checkpoint_path = os.path.join(artifact_dir, f"{key}.pt")
            if not os.path.exists(checkpoint_path):
                continue
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            _load_state_into_trainer(trainer, checkpoint, self.device)
        return stored

    def list_versions(self, agent_id: str) -> List[str]:
        """Return the sorted version labels saved for ``agent_id`` (newest last).

        Example:
            >>> AgentRegistry.list_versions.__qualname__
            'AgentRegistry.list_versions'
        """

        base = os.path.join(self.registry_path, agent_id)
        if not os.path.isdir(base):
            return []
        return sorted(
            d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))
        )

    def latest_version(self, agent_id: str) -> Optional[str]:
        """Return the most recent version label for ``agent_id`` (or ``None``).

        Example:
            >>> AgentRegistry.latest_version.__qualname__
            'AgentRegistry.latest_version'
        """

        versions = self.list_versions(agent_id)
        return versions[-1] if versions else None


if __name__ == "__main__":  # pragma: no cover - lightweight self-check
    import doctest

    failures, _ = doctest.testmod()
    if failures:
        raise SystemExit(f"{failures} doctest failure(s)")
    print("AgentRegistry doctests passed")
