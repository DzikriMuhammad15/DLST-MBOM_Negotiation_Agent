"""DDPG trainer for the DLST continuous head (``02-08``) — AS-IS formulas.

Ported AS-IS from the paper code (paper Sec. 5.2). The DLST template head is a
**continuous** actor-critic and is NOT affected by the input mode (G13): only the
front-end that produces the flat state vector differs, and that vector is built by
the owning agent's ``build_state`` (so this trainer stays mode-agnostic and keeps
consuming flat vectors, exactly as AS-IS). The DDPG critic is ``Q(s, a)`` and is
**not** enriched with the candidate pool (G6 applies to the PPO ``V(s)`` only).

Formulas (paper Eq. 16-19):

* critic loss (Eq. 18-19): ``L = MSE(Q(s, a), y)`` with
  ``y = r + gamma * (1 - done) * Q'(s', mu'(s'))``;
* actor update (Eq. 16-17, gradient ascent on ``J = E[Q(s, mu(s))]``),
  implemented as ``actor_loss = -mean(Q(s, mu(s)))``;
* soft target update: ``theta' <- tau * theta + (1 - tau) * theta'``.

Used three times by DLST (threshold / propose / respond) as **separate**
instances with **different reward shapers** (G2): no shared models / optimizers /
episodes / target networks between them.
"""

import copy
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from negotiation.models.ddpg_networks import (
    DDPGActorStrategy,
    DDPGActorThreshold,
    DDPGCritic,
)
from negotiation.trainers.trainer import Trainer, TrainerInterface

# Registry for checkpoint reconstruction.
MODEL_REGISTRY: Dict[str, Any] = {
    "DDPGActorThreshold": DDPGActorThreshold,
    "DDPGActorStrategy": DDPGActorStrategy,
    "DDPGCritic": DDPGCritic,
}


class DDPGTrainer(Trainer, TrainerInterface):
    """DDPG trainer for one DLST actor-critic pair (threshold / propose / respond).

    ``self.models`` format (via :meth:`attach_models`)::

        {"actor":  {"model": actor_net,  "optimizer": opt_a, "config": {...}},
         "critic": {"model": critic_net, "optimizer": opt_c, "config": {...}}}

    Example:
        >>> t = DDPGTrainer(gamma=0.99, tau=0.005, device="cpu", epoch=1,
        ...                 batch_size=8, reward_shaper=None)
        >>> t.tau
        0.005
    """

    def __init__(
        self,
        gamma: float,
        tau: float,
        device: str,
        epoch: int,
        batch_size: int,
        reward_shaper: Any,
        gradient_clip_norm: float = 1.0,
        action_dim: Optional[int] = None,
        models: Optional[Dict[str, Any]] = None,
        optimizers: Optional[Dict[str, Any]] = None,
        losses: Optional[Dict[str, Any]] = None,
        episodes: Optional[List[Any]] = None,
        agent: Any = None,
    ) -> None:
        super().__init__(
            models=models,
            optimizers=optimizers,
            reward_shaper=reward_shaper,
            losses=losses,
            episodes=episodes,
            batch_size=batch_size,
            agent=agent,
        )
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.epoch = epoch
        self.gradient_clip_norm = gradient_clip_norm
        self.action_dim = action_dim
        # Set by the agent so build_state knows which DLST head this trainer is for.
        self._state_for: Optional[str] = None
        self.target_models: Dict[str, Any] = {}

    # =====================================================
    # INTERFACE IMPLEMENTATIONS
    # =====================================================

    def store_episode(self, episode: Any) -> None:
        self.episodes.append(self._stamp_reward_bounds(episode))

    def attach_models(self, dict_of_models: Dict[str, Any]) -> None:
        """Attach actor/critic and build frozen target networks (AS-IS)."""
        self.models = {}
        self.optimizers = {}
        self.target_models = {}
        for key, value in dict_of_models.items():
            model = value["model"].to(self.device)
            self.models[key] = {
                "model": model,
                "class_name": model.__class__.__name__,
                "config": value.get("config", {}),
            }
            self.optimizers[key] = value["optimizer"]
            target = copy.deepcopy(model).to(self.device)
            for p in target.parameters():
                p.requires_grad_(False)
            self.target_models[key] = target

        if self.action_dim is None:
            actor_cfg = self.models.get("actor", {}).get("config", {})
            self.action_dim = actor_cfg.get("action_dim", 1)

    def attach_losses(self, dict_of_losses: Dict[str, Any]) -> None:
        self.losses = dict_of_losses

    def attach_agent(self, agent: Any) -> None:
        self.agent = agent

    # =====================================================
    # DEFAULT LOSSES (paper Eq. 16-19)
    # =====================================================

    def default_critic_loss(self, q_pred, y_target):
        """MSE between ``Q(s, a)`` and target ``y`` (Eq. 18) — AS-IS."""
        return F.mse_loss(q_pred, y_target)

    def default_actor_loss(self, q_for_policy):
        """Actor loss ``-mean(Q(s, mu(s)))`` (maximize J, Eq. 16) — AS-IS."""
        return -q_for_policy.mean()

    # =====================================================
    # SOFT UPDATE
    # =====================================================

    def _soft_update(self, key: str) -> None:
        """``theta' <- tau * theta + (1 - tau) * theta'`` for model ``key`` (AS-IS)."""
        main = self.models[key]["model"]
        target = self.target_models[key]
        with torch.no_grad():
            for tp, mp in zip(target.parameters(), main.parameters()):
                tp.data.mul_(1.0 - self.tau)
                tp.data.add_(self.tau * mp.data)

    # =====================================================
    # EPISODE -> BATCH (flat state vectors via agent.build_state)
    # =====================================================

    def _build_state(self, state: Any) -> Any:
        """Build the flat state vector via the owning agent (mode handled there)."""
        for_ = getattr(self, "_state_for", None)
        if for_ is not None:
            return self.agent.build_state(state, for_=for_)
        return self.agent.build_state(state)

    def _extract_states_actions(self, episode: Any):
        """Return ``(state_vecs, action_vectors)`` for one episode (AS-IS)."""
        state_vecs = [self._build_state(s) for s in episode.states]
        actions = episode.actions if episode.actions is not None else [s.action for s in episode.steps]
        return state_vecs, actions

    def episode_to_batch(self, episode: Any) -> Optional[Dict[str, Any]]:
        """Pack one episode into ``(states, actions, rewards, next_states, dones)`` (AS-IS)."""
        rewards = self.reward_shaper.compute_reward_sequence(episode)
        states, actions = self._extract_states_actions(episode)

        n = len(states)
        if n == 0:
            return None

        next_states = states[1:] + [states[-1]]
        dones = [False] * n
        dones[-1] = True

        states_np = np.array(states, dtype=np.float32)
        next_states_np = np.array(next_states, dtype=np.float32)
        actions_np = np.array(
            [np.atleast_1d(np.asarray(a, dtype=np.float32)) for a in actions],
            dtype=np.float32,
        )

        return {
            "states": torch.from_numpy(states_np).to(self.device),
            "actions": torch.from_numpy(actions_np).to(self.device),
            "rewards": torch.tensor(rewards, dtype=torch.float32, device=self.device),
            "next_states": torch.from_numpy(next_states_np).to(self.device),
            "dones": torch.tensor(dones, dtype=torch.float32, device=self.device),
        }

    def _episode_to_batch(self, episode: Any) -> Optional[Dict[str, Any]]:
        """Alias to :meth:`episode_to_batch` (satisfies the ABC name)."""
        return self.episode_to_batch(episode)

    def build_batch(self):
        """Concatenate all episodes into a shuffled replay ``DataLoader`` (AS-IS)."""
        from torch.utils.data import DataLoader, TensorDataset

        all_states, all_actions, all_rewards, all_next, all_dones = [], [], [], [], []
        for ep in self.episodes:
            pack = self.episode_to_batch(ep)
            if pack is None:
                continue
            all_states.append(pack["states"])
            all_actions.append(pack["actions"])
            all_rewards.append(pack["rewards"])
            all_next.append(pack["next_states"])
            all_dones.append(pack["dones"])

        if len(all_states) == 0:
            return None

        states = torch.cat(all_states, dim=0)
        actions = torch.cat(all_actions, dim=0)
        rewards = torch.cat(all_rewards, dim=0)
        next_states = torch.cat(all_next, dim=0)
        dones = torch.cat(all_dones, dim=0)
        if states.size(0) == 0:
            return None

        dataset = TensorDataset(states, actions, rewards, next_states, dones)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

    # =====================================================
    # UPDATE (DDPG, AS-IS)
    # =====================================================

    def update(self) -> Dict[str, Any]:
        """Run DDPG critic + actor updates with soft target updates (AS-IS)."""
        actor_entry = self.models.get("actor") if self.models else None
        critic_entry = self.models.get("critic") if self.models else None
        if actor_entry is None or critic_entry is None:
            return {"actor_loss": 0.0, "critic_loss": 0.0, "steps": 0}

        actor = actor_entry["model"]
        critic = critic_entry["model"]
        actor_target = self.target_models["actor"]
        critic_target = self.target_models["critic"]
        actor_opt = self.optimizers.get("actor")
        critic_opt = self.optimizers.get("critic")
        if actor_opt is None or critic_opt is None:
            return {"actor_loss": 0.0, "critic_loss": 0.0, "steps": 0}

        critic_loss_fn = self.losses.get("critic_loss", self.default_critic_loss)
        actor_loss_fn = self.losses.get("actor_loss", self.default_actor_loss)

        loader = self.build_batch()
        if loader is None:
            return {"actor_loss": 0.0, "critic_loss": 0.0, "steps": 0}

        actor.train()
        critic.train()
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_steps = 0
        # Per-epoch mean of actor/critic loss (training-log tracing only; the loss
        # formula is untouched — records the same values grouped by epoch).
        epoch_losses: List[Dict[str, float]] = []

        for _ in range(self.epoch):
            epoch_actor_sum = 0.0
            epoch_critic_sum = 0.0
            epoch_step_sum = 0
            for batch in loader:
                states, actions, rewards, next_states, dones = batch
                states = states.to(self.device)
                actions = actions.to(self.device)
                rewards = rewards.to(self.device)
                next_states = next_states.to(self.device)
                dones = dones.to(self.device)

                # Critic update (Eq. 18-19).
                with torch.no_grad():
                    next_actions = actor_target(next_states)
                    q_next = critic_target(next_states, next_actions)
                    y = rewards + self.gamma * (1.0 - dones) * q_next

                q_pred = critic(states, actions)
                critic_loss = critic_loss_fn(q_pred, y)

                critic_opt.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), self.gradient_clip_norm)
                critic_opt.step()

                # Actor update (Eq. 16-17).
                pred_actions = actor(states)
                q_for_policy = critic(states, pred_actions)
                actor_loss = actor_loss_fn(q_for_policy)

                actor_opt.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), self.gradient_clip_norm)
                actor_opt.step()

                self._soft_update("actor")
                self._soft_update("critic")

                bs = states.size(0)
                total_actor_loss += float(actor_loss.item()) * bs
                total_critic_loss += float(critic_loss.item()) * bs
                total_steps += bs
                epoch_actor_sum += float(actor_loss.item()) * bs
                epoch_critic_sum += float(critic_loss.item()) * bs
                epoch_step_sum += bs
            if epoch_step_sum > 0:
                epoch_losses.append({
                    "actor_loss": epoch_actor_sum / epoch_step_sum,
                    "critic_loss": epoch_critic_sum / epoch_step_sum,
                })

        if total_steps == 0:
            return {"actor_loss": 0.0, "critic_loss": 0.0, "steps": 0,
                    "epoch_losses": []}
        return {
            "actor_loss": total_actor_loss / total_steps,
            "critic_loss": total_critic_loss / total_steps,
            "steps": total_steps,
            "epoch_losses": epoch_losses,
        }

    def reset_episodes(self) -> None:
        self.episodes = []

    # =====================================================
    # CHECKPOINT (AS-IS)
    # =====================================================

    def save_checkpoint(self, path: str, extra: Any = None) -> None:
        """Persist trainer state + models + target networks + optimizers (AS-IS)."""
        checkpoint: Dict[str, Any] = {
            "trainer_state": {
                "gamma": self.gamma,
                "tau": self.tau,
                "epoch": self.epoch,
                "batch_size": self.batch_size,
                "gradient_clip_norm": self.gradient_clip_norm,
                "action_dim": self.action_dim,
            },
            "models": {},
            "optimizers": {},
            "target_models": {},
        }
        if self.models is not None:
            for key, entry in self.models.items():
                checkpoint["models"][key] = {
                    "class_name": entry["class_name"],
                    "config": entry["config"],
                    "state_dict": entry["model"].state_dict(),
                }
        for key, target in self.target_models.items():
            checkpoint["target_models"][key] = target.state_dict()
        if self.optimizers is not None:
            for key, optimizer in self.optimizers.items():
                checkpoint["optimizers"][key] = {
                    "class_name": optimizer.__class__.__name__,
                    "state_dict": optimizer.state_dict(),
                }
        if extra is not None:
            checkpoint["extra"] = extra
        torch.save(checkpoint, path)
        print(f"[INFO] Checkpoint saved: {path}")

    def load_checkpoint(self, path: str, load_optimizer: bool = True) -> None:
        """Restore trainer/model/target/optimizer state from ``path`` (AS-IS)."""
        checkpoint = torch.load(path, map_location=self.device)
        trainer_state = checkpoint.get("trainer_state", {})
        self.gamma = trainer_state.get("gamma", self.gamma)
        self.tau = trainer_state.get("tau", self.tau)
        self.epoch = trainer_state.get("epoch", self.epoch)
        self.batch_size = trainer_state.get("batch_size", self.batch_size)
        self.gradient_clip_norm = trainer_state.get("gradient_clip_norm", self.gradient_clip_norm)
        self.action_dim = trainer_state.get("action_dim", self.action_dim)

        self.models = {}
        self.target_models = {}
        for key, info in checkpoint.get("models", {}).items():
            class_name = info["class_name"]
            if class_name not in MODEL_REGISTRY:
                raise ValueError(
                    f"Model class '{class_name}' not found in MODEL_REGISTRY"
                )
            model = MODEL_REGISTRY[class_name](**info["config"]).to(self.device)
            model.load_state_dict(info["state_dict"])
            self.models[key] = {
                "model": model,
                "class_name": class_name,
                "config": info["config"],
            }
            target = copy.deepcopy(model).to(self.device)
            for p in target.parameters():
                p.requires_grad_(False)
            target_sd = checkpoint.get("target_models", {}).get(key)
            if target_sd is not None:
                target.load_state_dict(target_sd)
            self.target_models[key] = target
            print(f"[INFO] Loaded model: {key}")

        self.optimizers = {}
        if load_optimizer:
            for key, info in checkpoint.get("optimizers", {}).items():
                if key not in self.models:
                    print(f"[WARNING] No model found for optimizer '{key}'")
                    continue
                optimizer_class_name = info["class_name"]
                if not hasattr(torch.optim, optimizer_class_name):
                    raise ValueError(
                        f"Optimizer class '{optimizer_class_name}' not found in torch.optim"
                    )
                optimizer_class = getattr(torch.optim, optimizer_class_name)
                optimizer = optimizer_class(self.models[key]["model"].parameters())
                optimizer.load_state_dict(info["state_dict"])
                self.optimizers[key] = optimizer
                print(f"[INFO] Loaded optimizer: {key}")
        print(f"[INFO] Checkpoint loaded from: {path}")
