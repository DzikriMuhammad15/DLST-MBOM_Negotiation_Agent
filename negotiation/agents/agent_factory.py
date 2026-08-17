"""Agent factory — build any negotiator by **type + trainer + mode** (``08-01 P``).

Single entry point :func:`build_agent` that constructs a fully-wired negotiator
(agent + per-phase trainers + models + dimensions + ``k_max``) for one side of a
negotiation, from a flat :class:`AgentBuildConfig`. Every agent type listed in
``08-01 P`` can be built:

* ``RLNegotiator``
* ``MBOM`` (+ ``MBOMNonZeroSum`` / ``MBOMNewMechanism``)
* ``DLST``
* ``DLSTMBOM`` (+ ``DLSTMBOMNonZeroSum`` / ``DLSTMBOMNewMechanism``)
* ``RandomIOP`` (baseline)

Wiring rules (Invariant #1 enforced by construction):

* discrete agents (RL / MBOM family) get a **separate**
  :class:`DiscreteActorNet` (+ :class:`CriticNet` for PPO) for propose and for
  respond, each in its own PG / PPO trainer (``propose_trainer`` /
  ``respond_trainer``) with its own reward shaper — no shared state;
* DLST family gets **three** DDPG trainers (``threshold_trainer`` /
  ``propose_trainer`` / ``respond_trainer``), each its own actor-critic + reward
  shaper, with ``_state_for`` set so the DDPG batch rebuilds the right flat state
  (head AS-IS, G13);
* MBOM / DLST-MBOM additionally get ``iop_trainers_propose`` /
  ``iop_trainers_respond`` (``M`` :class:`IOPNetwork` cross-entropy trainers each);
* MBOM sizes ``FeatureDims.f_cand`` ``+1`` (per-candidate IOP-propose column) and
  ``f_progress`` ``+2`` (IOP-respond block); DLST-MBOM concatenates
  ``[dlst | iop_respond(2) | iop_propose(K)]`` and sizes each DDPG actor's
  ``state_dim`` as ``dlst_state_dim + 2 + K``.
* The ``*NewMechanism`` variants use the **N·N fan-out** instead of the single
  block: ``MBOMNewMechanism`` sizes ``f_progress`` ``+= 2N + N*N`` (and does NOT
  widen ``f_cand``); ``DLSTMBOMNewMechanism`` sizes each DDPG actor's ``state_dim``
  as ``dlst_state_dim + 2N + N*N`` (``N = len(action_library)``).

The factory only depends on the ``negotiation`` layer; the caller (bridge /
orchestrator) supplies the per-side world ufun (a ``BundleUFun``) and ``FeatureDims``
emitted by the bridge ``feature_builder`` (Contract C7). ``mode`` only changes the
network front-end (Gotcha G13).
"""

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

from negotiation.agents.dlst.strategy_template import StrategyTemplateSpec
from negotiation.agents.dlst.opponent_model import FrequencyOpponentModel
from negotiation.agents.dlst_agent import DLSTNegotiator
from negotiation.agents.dlst_mbom_agent import DLSTMBOMAgent
from negotiation.agents.dlst_mbom_new_mechanism_agent import DLSTMBOMNewMechanism
from negotiation.agents.dlst_mbom_directed_agent import DLSTMBOMDirected, DirectedConfig
from negotiation.agents.dlst_mbom_non_zero_sum_agent import DLSTMBOMNonZeroSum
from negotiation.agents.iop_agent import RandomIOPAgent
from negotiation.agents.mbom_agent import MBOMAgent
from negotiation.agents.mbom_agent_new_mechanism import MBOMNewMechanism
from negotiation.agents.mbom_agent_non_zero_sum import MBOMAgentNonZeroSum
from negotiation.agents.rl_agent import RLNegotiator
from negotiation.agents.rule_agent import RuleBasedAgent
from negotiation.models.ddpg_networks import (
    DDPGActorStrategy,
    DDPGActorThreshold,
    DDPGCritic,
)
from negotiation.models.directed_actors import (
    DirectedActorStrategy,
    DirectedActorThreshold,
)
from negotiation.models.encoders.feature_encoder import (
    CriticNet,
    DiscreteActorNet,
    FeatureDims,
)
from negotiation.models.iop_model import IOPNetwork
from negotiation.reward_shaper.reward_shaper import (
    ProposeRewardConfig,
    RespondRewardConfig,
    RewardShaper,
    ThresholdRewardConfig,
)
from negotiation.trainers.ddpg_trainer import DDPGTrainer
from negotiation.trainers.directed_ddpg_trainer import DirectedDDPGTrainer
from negotiation.trainers.iop_trainer import IOPTrainer
from negotiation.trainers.policy_gradient_trainer import PolicyGradientTrainer
from negotiation.trainers.ppo_trainer import PPOTrainer

# Canonical agent-type names (``08-01 P``).
AGENT_RL = "RLNegotiator"
AGENT_MBOM = "MBOM"
AGENT_MBOM_NZS = "MBOMNonZeroSum"
AGENT_MBOM_NEW = "MBOMNewMechanism"
AGENT_DLST = "DLST"
AGENT_DLST_MBOM = "DLSTMBOM"
AGENT_DLST_MBOM_NZS = "DLSTMBOMNonZeroSum"
AGENT_DLST_MBOM_NEW = "DLSTMBOMNewMechanism"
AGENT_DLST_MBOM_DIRECTED = "DLSTMBOMDirected"
AGENT_RANDOM_IOP = "RandomIOP"
AGENT_RULE = "Rule"

# Trainer kinds.
TRAINER_PG = "PG"
TRAINER_PPO = "PPO"
TRAINER_DDPG = "DDPG"

_DISCRETE_TYPES = {AGENT_RL, AGENT_MBOM, AGENT_MBOM_NZS, AGENT_MBOM_NEW}
_MBOM_TYPES = {AGENT_MBOM, AGENT_MBOM_NZS, AGENT_MBOM_NEW}
_DLST_TYPES = {AGENT_DLST, AGENT_DLST_MBOM, AGENT_DLST_MBOM_NZS, AGENT_DLST_MBOM_NEW,
               AGENT_DLST_MBOM_DIRECTED}
_DLST_MBOM_TYPES = {AGENT_DLST_MBOM, AGENT_DLST_MBOM_NZS, AGENT_DLST_MBOM_NEW,
                    AGENT_DLST_MBOM_DIRECTED}

_MBOM_CLASS = {
    AGENT_MBOM: MBOMAgent,
    AGENT_MBOM_NZS: MBOMAgentNonZeroSum,
    AGENT_MBOM_NEW: MBOMNewMechanism,
}
_DLST_MBOM_CLASS = {
    AGENT_DLST_MBOM: DLSTMBOMAgent,
    AGENT_DLST_MBOM_NZS: DLSTMBOMNonZeroSum,
    AGENT_DLST_MBOM_NEW: DLSTMBOMNewMechanism,
    AGENT_DLST_MBOM_DIRECTED: DLSTMBOMDirected,
}


# =============================================================================
# Build config
# =============================================================================

@dataclass
class TrainerHyperparams:
    """Trainer hyperparameters (``08-01 P``); defaults are a runnable baseline.

    Example:
        >>> TrainerHyperparams().gamma
        0.99
    """

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    gradient_clip_norm: float = 0.5
    epoch: int = 1
    batch_size: int = 32
    learning_rate: float = 3e-4
    tau: float = 0.005  # DDPG soft update
    hidden_dim: int = 128
    hidden_layers: int = 2


@dataclass
class AgentBuildConfig:
    """Everything needed to build one side's negotiator (``08-01 P``).

    Parameters
    ----------
    name, side:
        Negotiator name and ``"fleet"`` / ``"spklu"``.
    agent_type:
        One of the canonical agent-type names (``08-01 P``).
    trainer:
        ``"PG"`` / ``"PPO"`` (discrete) — ``"DDPG"`` is implied for DLST.
    mode:
        ``"PADDED"`` / ``"ENCODER"``.
    feature_dims:
        Bridge-emitted :class:`FeatureDims` (Contract C7). For MBOM the factory
        widens ``f_cand`` / ``f_progress`` internally.
    world_ufun:
        Per-side world ufun callable (wrapped into a ``BundleUFun`` by the caller
        and passed as ``ufun``).
    ufun:
        The NegMAS ``BundleUFun`` attached to the negotiator.
    domain_spec:
        The :class:`DomainSpec` (carries ``action_library`` etc.).
    k_max:
        Maximum action-library size (PADDED head width); defaults to
        ``feature_dims.k_max``.
    hyperparams:
        :class:`TrainerHyperparams`.
    critic_sees_candidate_pool / critic_privileged_info:
        PPO critic enrichment flags (G6, default off; V(s) only).
    propose_reward_mode / respond_reward_mode / threshold_reward_mode:
        Reward-shaper mode names (``02-09``). ``None`` means *unset* and is
        resolved per family at build time: non-DLST agents default to ``"mode1"``;
        the DLST family defaults the **respond** mode to ``"dlst"`` (paper r_acc,
        Eq. 10) — propose stays Eq. 9 and threshold stays Eq. 8 (``"mode1"``).
    device:
        Torch device.
    mbom_M / rollout_k / mbom_gamma / max_steps / n_rollout_samples:
        MBOM / DLST-MBOM hyperparameters.
    bidding_spec / acceptance_spec / u_fixed:
        DLST strategy-template specs + fixed acceptance threshold.
    value_lists:
        Per-issue value lists for the frequency opponent model.

    Example:
        >>> cfg = AgentBuildConfig(name="x", side="fleet",
        ...     agent_type="RLNegotiator", trainer="PPO", mode="PADDED",
        ...     feature_dims=None, world_ufun=None, ufun=None, domain_spec=None)
        >>> cfg.agent_type
        'RLNegotiator'
    """

    name: str
    side: str
    agent_type: str
    trainer: str
    mode: str
    feature_dims: Optional[FeatureDims]
    world_ufun: Optional[Callable[[Any], float]]
    ufun: Any
    domain_spec: Any
    k_max: Optional[int] = None
    hyperparams: TrainerHyperparams = field(default_factory=TrainerHyperparams)
    critic_sees_candidate_pool: bool = False
    critic_privileged_info: bool = False
    propose_reward_mode: Optional[str] = None
    respond_reward_mode: Optional[str] = None
    threshold_reward_mode: Optional[str] = None
    device: str = "cpu"
    mbom_M: int = 3
    rollout_k: int = 2
    mbom_gamma: float = 0.99
    max_steps: int = 10
    n_rollout_samples: int = 5
    state_dim_basis: int = 3
    bidding_spec: Optional[StrategyTemplateSpec] = None
    acceptance_spec: Optional[StrategyTemplateSpec] = None
    u_fixed: float = 0.6
    value_lists: Optional[Dict[str, List[Any]]] = None
    directed: DirectedConfig = field(default_factory=DirectedConfig)


# =============================================================================
# Internal builders
# =============================================================================

def _optimizer(params, lr: float):
    """Adam optimizer over ``params`` (kept tiny for testability)."""
    return torch.optim.Adam(params, lr=lr)


def _utility_bounds(cfg: AgentBuildConfig) -> tuple:
    """Per-side reward-scaling bounds from the domain spec (C6)."""
    spec = cfg.domain_spec
    if cfg.side == "fleet":
        umin = getattr(spec, "utility_min_fleet", 0.0)
        umax = getattr(spec, "utility_max_fleet", 1.0)
    else:
        umin = getattr(spec, "utility_min_spklu", 0.0)
        umax = getattr(spec, "utility_max_spklu", 1.0)
    return float(umin), float(umax)


def _resolve_reward_mode(explicit: Optional[str], phase: str, is_dlst_family: bool) -> str:
    """Resolve an (optionally unset) reward mode to a concrete name (``02-09``).

    Unset (``None``) defaults to ``"mode1"`` for every phase, EXCEPT the DLST
    family's **respond** phase, which defaults to ``"dlst"`` (paper r_acc, Eq. 10)
    so the opponent-model-aware acceptance reward is actually selected. An
    explicit config value always wins (no override of non-DLST agents).
    """
    if explicit is not None:
        return explicit
    if is_dlst_family and phase == "respond":
        return "dlst"
    return "mode1"


def _reward_shaper(
    cfg: AgentBuildConfig, phase: str, opponent_model: Any = None, is_dlst_family: bool = False
) -> RewardShaper:
    """Build the per-phase reward shaper with the per-side bounds (``02-09``)."""
    umin, umax = _utility_bounds(cfg)
    if phase == "propose":
        mode = _resolve_reward_mode(cfg.propose_reward_mode, "propose", is_dlst_family)
        return RewardShaper(ProposeRewardConfig(mode=mode), umin, umax, for_="propose")
    if phase == "respond":
        mode = _resolve_reward_mode(cfg.respond_reward_mode, "respond", is_dlst_family)
        return RewardShaper(
            RespondRewardConfig(mode=mode), umin, umax,
            for_="respond", opponent_model=opponent_model,
        )
    mode = _resolve_reward_mode(cfg.threshold_reward_mode, "utility_threshold", is_dlst_family)
    return RewardShaper(ThresholdRewardConfig(mode=mode), umin, umax, for_="utility_threshold")


def _discrete_dims(cfg: AgentBuildConfig, is_mbom: bool, k_max: int) -> FeatureDims:
    """Return the FeatureDims for a discrete agent (MBOM widens cand / progress).

    MBOM / MBOMNonZeroSum fold the single-state IOP block in as a ``+1``
    per-candidate ``cand`` column plus a ``+2`` IOP-respond progress block.
    MBOMNewMechanism instead uses the **N·N fan-out**: the whole fan-out block
    (``iop_respond(2N) | iop_propose(N*N)``) rides ``progress_feats``, so
    ``f_progress`` is widened by ``2N + N*N`` and ``f_cand`` is left untouched.

    The propose action-space size ``N`` is the construction-time ``k_max``
    (``= max_action_library``), NOT ``len(domain_spec.action_library)``: the real
    ``DomainSpec`` / ``action_library`` is injected per-negotiation (Contract C3),
    so it is unavailable (``domain_spec`` is typically ``None``) when the factory
    sizes the networks. ``k_max`` is the fixed upper bound the bridge enforces
    equal to ``max_action_library`` (G1 index-direct; same ``K_max`` as the
    discrete propose head).
    """
    dims = cfg.feature_dims
    if not is_mbom:
        return dims
    if cfg.agent_type == AGENT_MBOM_NEW:
        n = max(1, k_max)
        fanout = n * 2 + n * n  # iop_respond(2N) + iop_propose(N*N)
        return replace(dims, f_progress=dims.f_progress + fanout)
    # +1 per-candidate IOP-propose column; +2 IOP-respond progress block.
    return replace(dims, f_cand=dims.f_cand + 1, f_progress=dims.f_progress + 2)


def _build_discrete_trainer(cfg: AgentBuildConfig, dims: FeatureDims, phase: str, opponent_model: Any = None):
    """Build a PG / PPO trainer with separate actor (+ critic) for one phase."""
    hp = cfg.hyperparams
    actor = DiscreteActorNet(dims, cfg.mode, phase, hidden_dim=hp.hidden_dim,
                             n_layers=hp.hidden_layers)
    reward_shaper = _reward_shaper(cfg, phase, opponent_model)
    if cfg.trainer == TRAINER_PPO:
        critic = CriticNet(
            dims, cfg.mode, hidden_dim=hp.hidden_dim,
            critic_sees_candidate_pool=cfg.critic_sees_candidate_pool,
            critic_privileged_info=cfg.critic_privileged_info,
            n_layers=hp.hidden_layers,
        )
        trainer = PPOTrainer(
            hp.gamma, hp.gae_lambda, hp.clip_ratio, hp.value_coef, hp.entropy_coef,
            hp.gradient_clip_norm, cfg.device, hp.epoch, hp.batch_size, reward_shaper, mode=cfg.mode,
        )
        trainer.attach_models({
            "policy": {"model": actor, "optimizer": _optimizer(actor.parameters(), hp.learning_rate), "config": {}},
            "value": {"model": critic, "optimizer": _optimizer(critic.parameters(), hp.learning_rate), "config": {}},
        })
    elif cfg.trainer == TRAINER_PG:
        trainer = PolicyGradientTrainer(hp.gamma, hp.epoch, cfg.device, hp.batch_size, reward_shaper, mode=cfg.mode)
        trainer.attach_models({
            "policy": {"model": actor, "optimizer": _optimizer(actor.parameters(), hp.learning_rate), "config": {}},
        })
    else:
        raise ValueError(f"Discrete agents require trainer 'PG' or 'PPO', not {cfg.trainer!r}.")
    return trainer


def _build_iop_trainers(cfg: AgentBuildConfig, action_dim: int) -> List[IOPTrainer]:
    """Build ``M`` IOP cross-entropy trainers over the flat dim-3 state basis."""
    hp = cfg.hyperparams
    trainers: List[IOPTrainer] = []
    for _ in range(cfg.mbom_M):
        net = IOPNetwork(cfg.state_dim_basis, action_dim, hidden_dim=hp.hidden_dim,
                         n_layers=hp.hidden_layers)
        t = IOPTrainer(hp.epoch, cfg.device, hp.batch_size, None, hp.gradient_clip_norm)
        t.attach_models({
            "policy": {"model": net, "optimizer": _optimizer(net.parameters(), hp.learning_rate), "config": {}},
        })
        trainers.append(t)
    return trainers


def _build_ddpg_trainer(cfg: AgentBuildConfig, phase: str, state_dim: int, action_dim: int, split_sizes, opponent_model: Any = None, trainer_cls: Any = None, trainer_extra: Any = None, directed_hygiene: bool = False, init_scale: float = 3e-3):
    """Build a DDPG trainer + actor-critic for one DLST head (head AS-IS, G13).

    ``trainer_cls`` (aditif): kelas trainer alternatif (mis. ``DirectedDDPGTrainer`` untuk
    head propose DLSTMBOMDirected + L3 shaping); default ``DDPGTrainer`` (perilaku AS-IS).
    ``trainer_extra``: kwargs tambahan untuk trainer_cls.
    ``directed_hygiene`` (aditif): True → actor pakai varian L5 (LayerNorm B2 + init kecil B1);
    default False → actor AS-IS."""
    hp = cfg.hyperparams
    if phase == "utility_threshold":
        actor = (DirectedActorThreshold(state_dim, hidden_dim=hp.hidden_dim,
                                        n_layers=hp.hidden_layers, init_scale=init_scale)
                 if directed_hygiene else
                 DDPGActorThreshold(state_dim, hidden_dim=hp.hidden_dim,
                                    n_layers=hp.hidden_layers))
        critic_action_dim = 1
    else:
        actor = (DirectedActorStrategy(state_dim, action_dim, split_sizes,
                                       hidden_dim=hp.hidden_dim, n_layers=hp.hidden_layers,
                                       init_scale=init_scale)
                 if directed_hygiene else
                 DDPGActorStrategy(state_dim, action_dim, split_sizes, hidden_dim=hp.hidden_dim,
                                   n_layers=hp.hidden_layers))
        critic_action_dim = action_dim
    critic = DDPGCritic(state_dim, critic_action_dim, hidden_dim=hp.hidden_dim,
                        n_layers=hp.hidden_layers)
    reward_shaper = _reward_shaper(cfg, phase, opponent_model, is_dlst_family=True)
    tc = trainer_cls or DDPGTrainer
    trainer = tc(
        hp.gamma, hp.tau, cfg.device, hp.epoch, hp.batch_size, reward_shaper,
        gradient_clip_norm=hp.gradient_clip_norm, action_dim=critic_action_dim,
        **(trainer_extra or {}),
    )
    trainer.attach_models({
        "actor": {"model": actor, "optimizer": _optimizer(actor.parameters(), hp.learning_rate), "config": {}},
        "critic": {"model": critic, "optimizer": _optimizer(critic.parameters(), hp.learning_rate), "config": {}},
    })
    trainer._state_for = phase
    return trainer


# =============================================================================
# Parameter counting
# =============================================================================

def count_agent_parameters(agent: Any) -> Dict[str, int]:
    """Count total trainable parameters for every network in an agent.

    Returns a dict with a key per trainer role and a ``"total"`` key summing
    all parameters. Encoder phi parameters are counted automatically because
    they are sub-modules of ``DiscreteActorNet`` / ``CriticNet`` and therefore
    covered by ``model.parameters()`` recursively (Gotcha G5 / Q5 decision).

    Args:
        agent: Any negotiating agent with a ``trainers`` attribute dict.

    Returns:
        Dict containing ``"total"`` (int) plus one key per trainer.
        ``"iop_trainers_propose"`` and ``"iop_trainers_respond"`` aggregate
        across all indexed IOP trainers (``iop_trainers_propose__0``, …).

    Example:
        >>> count_agent_parameters.__name__
        'count_agent_parameters'
    """
    # Import here to avoid circular imports at module level.
    from simulation.registry.agent_registry import _enumerate_trainers  # type: ignore

    totals: Dict[str, int] = {}
    grand_total = 0

    for key, trainer in _enumerate_trainers(agent):
        models = getattr(trainer, "models", None) or {}
        trainer_total = 0
        for entry in models.values():
            model = entry.get("model") if isinstance(entry, dict) else None
            if model is not None:
                trainer_total += sum(p.numel() for p in model.parameters())
        totals[key] = trainer_total
        grand_total += trainer_total

    # Aggregate IOP trainer groups (sum iop_trainers_propose__* and iop_trainers_respond__*).
    iop_propose_total = sum(v for k, v in totals.items()
                            if k.startswith("iop_trainers_propose__"))
    iop_respond_total = sum(v for k, v in totals.items()
                            if k.startswith("iop_trainers_respond__"))

    result: Dict[str, int] = {"total": grand_total}
    # Single-role trainers use their key directly.
    for key, val in totals.items():
        if not (key.startswith("iop_trainers_propose__") or
                key.startswith("iop_trainers_respond__")):
            result[key] = val
    # Aggregate IOP groups.
    if iop_propose_total > 0:
        result["iop_trainers_propose"] = iop_propose_total
    if iop_respond_total > 0:
        result["iop_trainers_respond"] = iop_respond_total

    return result


# =============================================================================
# Public entry point
# =============================================================================

def _k_max(cfg: AgentBuildConfig) -> int:
    """Resolve the propose-head width (``k_max``) from the build config."""
    return cfg.k_max if cfg.k_max is not None else (cfg.feature_dims.k_max if cfg.feature_dims else 0)


def construct_agent_shell(cfg: AgentBuildConfig):
    """Build **only** the agent object (no trainers / models attached).

    This is the wrapper-only half of :func:`build_agent`: it constructs the
    negotiator object exactly as ``build_agent`` does, but attaches **no** trainers.
    It exists so a fresh agent shell can be built per negotiation episode while the
    expensive, stateful trainers (models + optimizers) are preserved and re-attached
    via :func:`rebuild_agent_shell_with_trainers` — mirroring the per-episode
    new-agent / persistent-trainer pattern of the old ``main_train_*.py`` scripts.

    The agent reads its networks through ``self.trainers`` at inference time
    (it never caches model references at construction), so a shell built here is
    inert until trainers are attached. The caller still calls
    ``agent.configure_negotiation(...)`` per negotiation (Contract C3).

    Example:
        >>> construct_agent_shell.__name__
        'construct_agent_shell'
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError("torch is required to build agents.")

    at = cfg.agent_type
    k_max = _k_max(cfg)

    if at == AGENT_RANDOM_IOP:
        return RandomIOPAgent(name=cfg.name, ufun=cfg.ufun, domain_spec=cfg.domain_spec)

    if at == AGENT_RULE:
        return RuleBasedAgent(name=cfg.name, ufun=cfg.ufun, domain_spec=cfg.domain_spec)

    if at in _DISCRETE_TYPES:
        if at == AGENT_RL:
            return RLNegotiator(cfg.name, cfg.ufun, cfg.domain_spec, mode=cfg.mode, k_max=k_max)
        return _MBOM_CLASS[at](
            cfg.name, cfg.ufun, cfg.domain_spec,
            state_dim=cfg.state_dim_basis, M=cfg.mbom_M, rollout_k=cfg.rollout_k,
            gamma=cfg.mbom_gamma, n_rollout_samples=cfg.n_rollout_samples,
            max_steps=cfg.max_steps, mode=cfg.mode, k_max=k_max,
        )

    if at in _DLST_TYPES:
        bidding_spec = cfg.bidding_spec or _default_bidding_spec()
        acceptance_spec = cfg.acceptance_spec or _default_acceptance_spec()
        if at in _DLST_MBOM_TYPES:
            # DLSTMBOMDirected menerima config switch `directed` (aditif); tipe lain tidak.
            extra = {"directed": cfg.directed} if at == AGENT_DLST_MBOM_DIRECTED else {}
            return _DLST_MBOM_CLASS[at](
                cfg.name, cfg.ufun, cfg.domain_spec, bidding_spec, acceptance_spec,
                u_fixed=cfg.u_fixed, mode=cfg.mode, state_dim=cfg.state_dim_basis,
                M=cfg.mbom_M, rollout_k=cfg.rollout_k, gamma=cfg.mbom_gamma,
                n_rollout_samples=cfg.n_rollout_samples, max_steps=cfg.max_steps,
                k_max=k_max, **extra,
            )
        return DLSTNegotiator(
            cfg.name, cfg.ufun, cfg.domain_spec, bidding_spec, acceptance_spec,
            u_fixed=cfg.u_fixed, mode=cfg.mode,
        )

    raise ValueError(f"Unknown agent_type {at!r}. Known: {sorted(ALL_AGENT_TYPES)}.")


def _build_trainers(cfg: AgentBuildConfig) -> Dict[str, Any]:
    """Build the fresh ``{role: trainer}`` wiring for one side (no agent attached).

    Holds all the trainer-construction logic that used to live inline in
    ``build_agent``: separate propose/respond (PG/PPO) trainers for the discrete
    family, three DDPG trainers (+ optional IOP) for the DLST family, and the
    shared frequency opponent model for the DLST respond reward shaper. Baselines
    (``RandomIOP`` / ``Rule``) have no trainers and yield an empty mapping.

    The owning agent is **not** attached here; :func:`build_agent` (and the rebuild
    path) call ``trainer.attach_agent(agent)`` afterwards. Returns an empty dict for
    non-learner agent types.
    """
    at = cfg.agent_type
    k_max = _k_max(cfg)

    if at in (AGENT_RANDOM_IOP, AGENT_RULE):
        return {}

    if at in _DISCRETE_TYPES:
        is_mbom = at in _MBOM_TYPES
        dims = _discrete_dims(cfg, is_mbom, k_max)
        wiring: Dict[str, Any] = {
            "propose_trainer": _build_discrete_trainer(cfg, dims, "propose"),
            "respond_trainer": _build_discrete_trainer(cfg, dims, "respond"),
        }
        if is_mbom:
            # IOP propose head width = k_max (the fixed propose action space), NOT
            # len(action_library) (injected per-negotiation via C3; domain_spec is
            # None at construction). k_max == max_action_library (bridge-enforced).
            wiring["iop_trainers_propose"] = _build_iop_trainers(cfg, max(1, k_max))
            wiring["iop_trainers_respond"] = _build_iop_trainers(cfg, 2)
        return wiring

    if at in _DLST_TYPES:
        bidding_spec = cfg.bidding_spec or _default_bidding_spec()
        acceptance_spec = cfg.acceptance_spec or _default_acceptance_spec()
        is_hybrid = at in _DLST_MBOM_TYPES

        # DLST state dims (enriched by IOP blocks for the hybrid). The IOP propose
        # action space is sized from k_max (== max_action_library, bridge-enforced),
        # NOT len(action_library): the real library is injected per-negotiation
        # (C3) and domain_spec is typically None at construction.
        iop_propose_dim = max(1, k_max)
        iop_respond_dim = 2
        if not is_hybrid:
            extra = 0
        elif at == AGENT_DLST_MBOM_NEW:
            # N·N fan-out: [dlst | iop_respond(2N) | iop_propose(N*N)].
            n = iop_propose_dim
            extra = n * iop_respond_dim + n * iop_propose_dim
        elif at == AGENT_DLST_MBOM_DIRECTED:
            # L1 ON → concat RINGKAS-8 (spec §2b); L1 OFF → identik NewMechanism (fan-out 2N+N²).
            if cfg.directed.l1:
                extra = 8
            else:
                n = iop_propose_dim
                extra = n * iop_respond_dim + n * iop_propose_dim
        else:
            # Single-state block: [dlst | iop_respond(2) | iop_propose(K)].
            extra = iop_propose_dim + iop_respond_dim
        th_dim = DLSTNegotiator.STATE_DIM_THRESHOLD + extra
        re_dim = DLSTNegotiator.STATE_DIM_RESPOND + extra
        pr_dim = DLSTNegotiator.STATE_DIM_PROPOSE + extra

        # Shared frequency opponent model (used by the respond reward shaper).
        # value_lists span the propose action space (k_max indices) when not given
        # explicitly — the real library is injected per-negotiation (C3).
        opponent_model = FrequencyOpponentModel(
            issue_names=[getattr(cfg.domain_spec, "issue_name", "bundle_index") or "bundle_index"],
            value_lists=cfg.value_lists or {"bundle_index": list(range(max(1, k_max)))},
        )

        # DLSTMBOMDirected: head PROPOSE pakai DirectedDDPGTrainer (L3 PBRS shaping, §6);
        # head lain tetap DDPGTrainer AS-IS. Config-gated (shaping default OFF).
        propose_cls = propose_extra = None
        is_directed = at == AGENT_DLST_MBOM_DIRECTED
        hyg = bool(cfg.directed.hygiene) if is_directed else False   # L5 B1/B2 actor
        isc = float(cfg.directed.init_scale) if is_directed else 3e-3
        if is_directed:
            propose_cls = DirectedDDPGTrainer
            propose_extra = {
                "shaping_enabled": cfg.directed.shaping,
                "beta": cfg.directed.beta,
                "gamma_shaping": cfg.directed.gamma_shaping,
            }
        wiring = {
            "threshold_trainer": _build_ddpg_trainer(cfg, "utility_threshold", th_dim, 1, (1, 0, 0), directed_hygiene=hyg, init_scale=isc),
            "propose_trainer": _build_ddpg_trainer(cfg, "propose", pr_dim, bidding_spec.action_dim, bidding_spec.split_sizes, trainer_cls=propose_cls, trainer_extra=propose_extra, directed_hygiene=hyg, init_scale=isc),
            "respond_trainer": _build_ddpg_trainer(cfg, "respond", re_dim, acceptance_spec.action_dim, acceptance_spec.split_sizes, opponent_model=opponent_model, directed_hygiene=hyg, init_scale=isc),
        }
        if is_hybrid:
            wiring["iop_trainers_propose"] = _build_iop_trainers(cfg, iop_propose_dim)
            wiring["iop_trainers_respond"] = _build_iop_trainers(cfg, iop_respond_dim)
        return wiring

    raise ValueError(f"Unknown agent_type {at!r}. Known: {sorted(ALL_AGENT_TYPES)}.")


def _iter_trainer_objs(wiring: Dict[str, Any]):
    """Yield every distinct trainer object in a ``{role: trainer}`` wiring.

    Flattens the list-valued MBOM/DLST-MBOM IOP roles (``iop_trainers_*``).
    """
    for value in (wiring or {}).values():
        if isinstance(value, (list, tuple)):
            for t in value:
                if t is not None:
                    yield t
        elif value is not None:
            yield value


def _attach_trainers(agent: Any, wiring: Dict[str, Any]) -> None:
    """Attach a ``{role: trainer}`` wiring to ``agent`` and back-link each trainer.

    Sets ``agent.trainers`` (via ``attach_trainer``) and calls
    ``trainer.attach_agent(agent)`` on every trainer that supports it (DDPG / IOP /
    discrete) so the owning-agent back-reference points at this exact agent.
    """
    if not wiring:
        return
    agent.attach_trainer(wiring)
    for trainer in _iter_trainer_objs(wiring):
        attach = getattr(trainer, "attach_agent", None)
        if callable(attach):
            attach(agent)


def build_agent(cfg: AgentBuildConfig):
    """Build and fully wire one negotiator for one side (``08-01 P``).

    Returns the constructed agent with **fresh** trainers attached and
    ``attach_agent`` set where a trainer needs the owning agent (DDPG / IOP). The
    caller still calls ``agent.configure_negotiation(...)`` per negotiation
    (Contract C3).

    Implemented as :func:`construct_agent_shell` + :func:`_build_trainers` +
    :func:`_attach_trainers`; see :func:`rebuild_agent_shell_with_trainers` for the
    per-episode variant that preserves existing trainers.

    Example:
        >>> build_agent.__name__
        'build_agent'
    """
    agent = construct_agent_shell(cfg)
    _attach_trainers(agent, _build_trainers(cfg))
    return agent


def rebuild_agent_shell_with_trainers(cfg: AgentBuildConfig, trainers: Dict[str, Any]):
    """Build a **fresh** agent shell and re-attach **existing** trainers to it.

    This is the per-episode rebuild used to guarantee that no NegMAS-internal agent
    state (e.g. the SAO proposal cache) leaks across negotiations: every episode
    gets a brand-new agent object, but the trained trainers (models + optimizers)
    from the previous agent are carried over, so learning is never lost. Mirrors the
    old ``main_train_*.py`` loop (new agent each episode, same trainers).

    ``trainers`` is the previous agent's ``trainers`` mapping (``{role: trainer}``).
    For non-learner baselines (``RandomIOP`` / ``Rule``) it is typically empty and
    the call reduces to a fresh shell. The caller still invokes
    ``agent.configure_negotiation(...)`` per negotiation (Contract C3).

    Example:
        >>> rebuild_agent_shell_with_trainers.__name__
        'rebuild_agent_shell_with_trainers'
    """
    agent = construct_agent_shell(cfg)
    _attach_trainers(agent, trainers)
    return agent


# =============================================================================
# Defaults for the DLST strategy templates
# =============================================================================

def _default_bidding_spec() -> StrategyTemplateSpec:
    """A small, runnable default bidding strategy template (``02-05``)."""
    return StrategyTemplateSpec(
        n_phases=2,
        tactics_per_phase=[2, 2],
        tactic_names_per_phase=[["boulware", "pareto"], ["b_opp", "random_above"]],
    )


def _default_acceptance_spec() -> StrategyTemplateSpec:
    """A small, runnable default acceptance strategy template (``02-05``)."""
    return StrategyTemplateSpec(
        n_phases=2,
        tactics_per_phase=[2, 2],
        tactic_names_per_phase=[["u_bar", "u_fixed"], ["quantile", "U_future"]],
    )


ALL_AGENT_TYPES = {
    AGENT_RL, AGENT_MBOM, AGENT_MBOM_NZS, AGENT_MBOM_NEW,
    AGENT_DLST, AGENT_DLST_MBOM, AGENT_DLST_MBOM_NZS, AGENT_DLST_MBOM_NEW,
    AGENT_DLST_MBOM_DIRECTED,
    AGENT_RANDOM_IOP, AGENT_RULE,
}
