"""PPO trainer (``02-08``) with mode-aware batch plumbing + log_prob recompute (``03-05``).

All formulas are **exactly** the paper / AS-IS PPO:

* GAE: ``delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)``;
  ``gae = delta_t + gamma * lambda * (1 - done_t) * gae``; ``returns = adv + V``.
* clipped surrogate policy loss:
  ``ratio = exp(log_prob - old_log_prob)``;
  ``loss = -min(ratio * adv, clip(ratio, 1-clip, 1+clip) * adv).mean()``.
* value loss: ``MSE(values, returns)``.
* entropy bonus: ``Categorical(logits).entropy().mean()``.
* total: ``policy_loss + value_coef * value_loss - entropy_coef * entropy``,
  gradient clipping ``gradient_clip_norm``.

The only change vs. AS-IS is *plumbing*: the policy is the mode-aware
:class:`~negotiation.models.encoders.feature_encoder.DiscreteActorNet` and the
value net is :class:`~negotiation.models.encoders.feature_encoder.CriticNet`,
both consuming a batched packed-observation dict. At update time the stored
per-step observation dicts plus their ``action_mask`` (PADDED) /
``candidate_features`` (ENCODER) are re-collated and re-forwarded so the
recomputed ``log_prob`` uses the **identical** ``Categorical`` distribution as
sampling (Contract C4 / ``03-05``). No loss formula is touched.

Per phase (G2): separate propose / respond instances, no shared state. The
critic enrichment flags (``critic_sees_candidate_pool`` / ``critic_privileged_info``)
live on the ``CriticNet`` itself (G6, default off) and affect only ``V(s)``.
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from negotiation.trainers.trainer import Trainer, TrainerInterface

MODEL_REGISTRY: Dict[str, Any] = {}


class PPOTrainer(Trainer, TrainerInterface):
    """PPO trainer for one phase (propose **or** respond).

    ``self.models`` format (via :meth:`attach_models`)::

        {"policy": {"model": actor_net, "optimizer": opt_p, "config": {...}},
         "value":  {"model": critic_net, "optimizer": opt_v, "config": {...}}}

    Example:
        >>> t = PPOTrainer(gamma=0.99, gae_lambda=0.95, clip_ratio=0.2,
        ...                value_coef=0.5, entropy_coef=0.01,
        ...                gradient_clip_norm=0.5, device="cpu", epoch=1,
        ...                batch_size=8, reward_shaper=None, mode="ENCODER")
        >>> t.clip_ratio, t.mode
        (0.2, 'ENCODER')
    """

    def __init__(
        self,
        gamma: float,
        gae_lambda: float,
        clip_ratio: float,
        value_coef: float,
        entropy_coef: float,
        gradient_clip_norm: float,
        device: str,
        epoch: int,
        batch_size: int,
        reward_shaper: Any,
        mode: str = "PADDED",
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
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.gradient_clip_norm = gradient_clip_norm
        self.device = device
        self.epoch = epoch
        self.mode = mode

    # =====================================================
    # PPO LOSSES (AS-IS)
    # =====================================================

    def default_policy_loss(self, logits, actions, old_log_probs, advantages, clip_ratio):
        """Clipped surrogate policy loss (AS-IS).

        Example:
            >>> import torch
            >>> t = PPOTrainer(0.99, 0.95, 0.2, 0.5, 0.01, 0.5, "cpu", 1, 8, None)
            >>> logits = torch.zeros(2, 3)
            >>> a = torch.tensor([0, 1])
            >>> olp = torch.distributions.Categorical(logits=logits).log_prob(a)
            >>> loss = t.default_policy_loss(logits, a, olp,
            ...                              torch.tensor([1.0, 1.0]), 0.2)
            >>> round(float(loss.item()), 4)
            -1.0
        """
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        ratio = torch.exp(log_probs - old_log_probs)
        unclipped = ratio * advantages
        clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
        loss = -torch.min(unclipped, clipped).mean()
        return loss

    def default_value_loss(self, values, returns_t):
        """Value MSE loss (AS-IS)."""
        return F.mse_loss(values, returns_t)

    def default_entropy_loss(self, logits):
        """Entropy bonus ``Categorical(logits).entropy().mean()`` (AS-IS)."""
        dist = torch.distributions.Categorical(logits=logits)
        return dist.entropy().mean()

    def _compute_gae(self, rewards, values, dones):
        """Return ``(returns, advantages)`` via GAE (AS-IS).

        Example:
            >>> t = PPOTrainer(1.0, 1.0, 0.2, 0.5, 0.0, 0.5, "cpu", 1, 8, None)
            >>> ret, adv = t._compute_gae([1.0, 1.0], [0.0, 0.0], [False, True])
            >>> [round(float(x), 4) for x in ret]
            [2.0, 1.0]
        """
        advantages: List[float] = []
        gae = 0.0
        values = list(values) + [0.0]
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1.0 - float(dones[t])) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1.0 - float(dones[t])) * gae
            advantages.insert(0, gae)
        returns = [a + v for a, v in zip(advantages, values[:-1])]
        return returns, advantages

    # =====================================================
    # INTERFACE IMPLEMENTATIONS
    # =====================================================

    def store_episode(self, episode: Any) -> None:
        self.episodes.append(self._stamp_reward_bounds(episode))

    def attach_models(self, dict_of_models: Dict[str, Any]) -> None:
        self.models = {}
        self.optimizers = {}
        for key, value in dict_of_models.items():
            model = value["model"].to(self.device)
            self.models[key] = {
                "model": model,
                "class_name": model.__class__.__name__,
                "config": value.get("config", {}),
            }
            self.optimizers[key] = value["optimizer"]

    def attach_losses(self, dict_of_losses: Dict[str, Any]) -> None:
        self.losses = dict_of_losses

    def attach_agent(self, agent: Any) -> None:
        self.agent = agent

    # =====================================================
    # EPISODE -> per-episode pack (mode-aware, dict states)
    # =====================================================

    def _episode_to_batch(self, episode: Any) -> Optional[Dict[str, Any]]:
        """Pack one finalized episode: dict states, masks, actions, old log_probs, GAE.

        ``values`` for GAE are the stored sampling-time ``V(s)`` (``step.value``);
        the recomputed ``V(s)`` for the value loss is produced at update time by
        re-forwarding the critic over the re-collated obs.
        """
        rewards = self.reward_shaper.compute_reward_sequence(episode)
        steps = episode.steps
        if not steps:
            return None

        states = episode.states if episode.states is not None else [s.state for s in steps]
        actions = episode.actions if episode.actions is not None else [int(s.action) for s in steps]
        old_log_probs = episode.log_probs if episode.log_probs is not None else [s.log_prob for s in steps]
        values = episode.values if episode.values is not None else [s.value for s in steps]
        action_masks = [s.action_mask for s in steps]
        candidate_features = [s.candidate_features for s in steps]

        if len(states) == 0:
            return None

        values = [0.0 if v is None else float(v) for v in values]
        dones = [False] * len(rewards)
        if dones:
            dones[-1] = True
        returns, advantages = self._compute_gae(rewards, values, dones)

        return {
            "states": states,
            "actions": [int(a) for a in actions],
            "old_log_probs": [float(lp) for lp in old_log_probs],
            "action_masks": action_masks,
            "candidate_features": candidate_features,
            "returns": returns,
            "advantages": advantages,
        }

    # =====================================================
    # BUILD BATCH (re-collate ragged dict states; 03-05)
    # =====================================================

    def build_batch(self) -> Optional[Dict[str, Any]]:
        """Re-collate all stored episodes into one batched obs dict + PPO targets."""
        all_states: List[Dict[str, Any]] = []
        all_actions: List[int] = []
        all_old_lp: List[float] = []
        all_masks: List[Any] = []
        all_cand_feats: List[Any] = []
        all_returns: List[float] = []
        all_adv: List[float] = []

        for ep in self.episodes:
            pack = self._episode_to_batch(ep)
            if pack is None:
                continue
            all_states.extend(pack["states"])
            all_actions.extend(pack["actions"])
            all_old_lp.extend(pack["old_log_probs"])
            all_masks.extend(pack["action_masks"])
            all_cand_feats.extend(pack["candidate_features"])
            all_returns.extend(pack["returns"])
            all_adv.extend(pack["advantages"])

        if len(all_states) == 0:
            return None

        obs = self.collate_states(
            all_states,
            action_masks=all_masks if self.mode == "PADDED" else None,
            candidate_features=all_cand_feats if self.mode == "ENCODER" else None,
            mode=self.mode,
        )
        return {
            "obs": obs,
            "actions": torch.tensor(all_actions, dtype=torch.long, device=self.device),
            "old_log_probs": torch.tensor(all_old_lp, dtype=torch.float32, device=self.device),
            "returns": torch.tensor(all_returns, dtype=torch.float32, device=self.device),
            "advantages": torch.tensor(all_adv, dtype=torch.float32, device=self.device),
        }

    # =====================================================
    # UPDATE (AS-IS loss; recompute log_prob over identical dist)
    # =====================================================

    def update(self) -> Dict[str, Any]:
        """Run PPO updates; all losses are the AS-IS formulas (C4 recompute)."""
        if not self.models or "policy" not in self.models or "value" not in self.models:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy_loss": 0.0}

        policy = self.models["policy"]["model"]
        value = self.models["value"]["model"]
        policy_optimizer = self.optimizers["policy"]
        value_optimizer = self.optimizers["value"]

        policy_loss_fn = self.losses.get("policy_loss", self.default_policy_loss)
        value_loss_fn = self.losses.get("value_loss", self.default_value_loss)
        entropy_loss_fn = self.losses.get("entropy_loss", self.default_entropy_loss)

        batch = self.build_batch()
        if batch is None:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy_loss": 0.0}

        obs = batch["obs"]
        actions = batch["actions"]
        old_log_probs = batch["old_log_probs"]
        returns = batch["returns"]
        advantages = batch["advantages"]
        n = int(actions.shape[0])
        # AS-IS minibatching (02-08): shuffle once per epoch and iterate
        # ``batch_size`` minibatches. The obs dict is already collated to one
        # batch (every value has leading dim ``n``), so a minibatch is a plain
        # index-select over dim 0 — the per-row padding/masks are unchanged, so
        # the re-forwarded ``Categorical`` stays identical to sampling (C4).
        batch_size = self.batch_size if self.batch_size and self.batch_size > 0 else n

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        total_steps = 0
        # Per-epoch mean of each loss component (training-log tracing only; the
        # loss formula is untouched — this records the same values grouped by epoch).
        epoch_losses: List[Dict[str, float]] = []

        policy.train()
        value.train()
        for _ in range(self.epoch):
            perm = torch.randperm(n, device=self.device)
            epoch_policy_sum = 0.0
            epoch_value_sum = 0.0
            epoch_entropy_sum = 0.0
            epoch_step_sum = 0
            for start in range(0, n, batch_size):
                mb_idx = perm[start : start + batch_size]
                mb_obs = {key: tensor[mb_idx] for key, tensor in obs.items()}
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_returns = returns[mb_idx]
                mb_advantages = advantages[mb_idx]

                policy_optimizer.zero_grad()
                value_optimizer.zero_grad()

                # Re-forward the mode-aware nets -> identical Categorical to sampling.
                logits = policy(mb_obs)
                values = value(mb_obs).view(-1)

                policy_loss = policy_loss_fn(
                    logits, mb_actions, mb_old_log_probs, mb_advantages, self.clip_ratio
                )
                value_loss = value_loss_fn(values, mb_returns)
                entropy_loss = entropy_loss_fn(logits)

                total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss
                total_loss.backward()

                torch.nn.utils.clip_grad_norm_(policy.parameters(), self.gradient_clip_norm)
                torch.nn.utils.clip_grad_norm_(value.parameters(), self.gradient_clip_norm)

                policy_optimizer.step()
                value_optimizer.step()

                mb_n = int(mb_idx.shape[0])
                total_policy_loss += float(policy_loss.item()) * mb_n
                total_value_loss += float(value_loss.item()) * mb_n
                total_entropy_loss += float(entropy_loss.item()) * mb_n
                total_steps += mb_n
                epoch_policy_sum += float(policy_loss.item()) * mb_n
                epoch_value_sum += float(value_loss.item()) * mb_n
                epoch_entropy_sum += float(entropy_loss.item()) * mb_n
                epoch_step_sum += mb_n
            if epoch_step_sum > 0:
                epoch_losses.append({
                    "policy_loss": epoch_policy_sum / epoch_step_sum,
                    "value_loss": epoch_value_sum / epoch_step_sum,
                    "entropy_loss": epoch_entropy_sum / epoch_step_sum,
                })

        if total_steps == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy_loss": 0.0,
                    "epoch_losses": []}
        return {
            "policy_loss": total_policy_loss / total_steps,
            "value_loss": total_value_loss / total_steps,
            "entropy_loss": total_entropy_loss / total_steps,
            "epoch_losses": epoch_losses,
        }

    def reset_episodes(self) -> None:
        self.episodes = []

    # =====================================================
    # CHECKPOINT (AS-IS)
    # =====================================================

    def save_checkpoint(self, path: str, extra: Any = None) -> None:
        """Persist trainer state + each model/optimizer state_dict (AS-IS)."""
        checkpoint: Dict[str, Any] = {
            "trainer_state": {
                "gamma": self.gamma,
                "gae_lambda": self.gae_lambda,
                "clip_ratio": self.clip_ratio,
                "value_coef": self.value_coef,
                "entropy_coef": self.entropy_coef,
                "gradient_clip_norm": self.gradient_clip_norm,
                "epoch": self.epoch,
                "mode": self.mode,
            },
            "models": {},
            "optimizers": {},
        }
        if self.models is not None:
            for key, entry in self.models.items():
                checkpoint["models"][key] = {
                    "class_name": entry["class_name"],
                    "config": entry["config"],
                    "state_dict": entry["model"].state_dict(),
                }
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
        """Restore trainer/model/optimizer state from ``path`` (AS-IS)."""
        checkpoint = torch.load(path, map_location=self.device)
        trainer_state = checkpoint.get("trainer_state", {})
        self.gamma = trainer_state.get("gamma", self.gamma)
        self.gae_lambda = trainer_state.get("gae_lambda", self.gae_lambda)
        self.clip_ratio = trainer_state.get("clip_ratio", self.clip_ratio)
        self.value_coef = trainer_state.get("value_coef", self.value_coef)
        self.entropy_coef = trainer_state.get("entropy_coef", self.entropy_coef)
        self.gradient_clip_norm = trainer_state.get("gradient_clip_norm", self.gradient_clip_norm)
        self.epoch = trainer_state.get("epoch", self.epoch)
        self.mode = trainer_state.get("mode", self.mode)

        self.models = {}
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
