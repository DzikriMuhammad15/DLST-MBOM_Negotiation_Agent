"""REINFORCE / Policy-Gradient trainer (``02-08``) with mode-aware batch plumbing (``03-05``).

Loss & returns are **exactly** the paper / AS-IS formulas:

* discounted returns ``G_t = r_t + gamma * G_{t+1}``;
* advantages ``= returns - mean``, divided by ``std`` when ``std > 1e-8``;
* loss ``= -(log_prob * returns).mean()`` with
  ``log_prob = Categorical(logits).log_prob(actions)``.

The only change vs. AS-IS is the *plumbing*: the policy is now the mode-aware
:class:`~negotiation.models.encoders.feature_encoder.DiscreteActorNet`, which
consumes a batched packed-observation dict instead of a flat state vector. At
update time the stored per-step observation dicts plus their ``action_mask``
(PADDED) / ``candidate_features`` (ENCODER) are re-collated via
:meth:`~negotiation.trainers.trainer.Trainer.collate_states` and re-forwarded, so
the ``Categorical`` distribution is reconstructed identically to sampling
(Contract C4). No loss formula is touched.

This trainer is **per phase** (G2): an agent uses a separate instance for propose
and for respond, with no shared models / optimizers / episode buffer / reward
shaper.
"""

from typing import Any, Dict, List, Optional

import torch

from negotiation.trainers.trainer import Trainer, TrainerInterface

# Model registry for checkpoint reconstruction (filled by the agent factory).
MODEL_REGISTRY: Dict[str, Any] = {}


class PolicyGradientTrainer(Trainer, TrainerInterface):
    """REINFORCE trainer for one phase (propose **or** respond).

    ``self.models`` format (via :meth:`attach_models`)::

        {"policy": {"model": actor_net, "optimizer": opt, "config": {...}}}

    ``model`` is the mode-aware ``DiscreteActorNet`` whose front-end / score-head
    sub-modules ride in ``state_dict`` and are covered by ``optimizer`` (G5).

    Example:
        >>> t = PolicyGradientTrainer(gamma=0.99, epoch=1, device="cpu",
        ...                           batch_size=8, reward_shaper=None,
        ...                           mode="PADDED")
        >>> t.gamma, t.mode
        (0.99, 'PADDED')
    """

    def __init__(
        self,
        gamma: float,
        epoch: int,
        device: str,
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
        self.epochs = epoch
        self.device = device
        self.batch_size = batch_size
        self.mode = mode

    # =====================================================
    # LOSS (AS-IS)
    # =====================================================

    def default_loss(self, logits, actions, returns):
        """REINFORCE loss ``-(log_prob * returns).mean()`` (AS-IS).

        Example:
            >>> import torch
            >>> t = PolicyGradientTrainer(0.99, 1, "cpu", 8, None)
            >>> logits = torch.zeros(2, 3)
            >>> loss = t.default_loss(logits, torch.tensor([0, 1]),
            ...                       torch.tensor([1.0, -1.0]))
            >>> float(loss.item()) == 0.0
            True
        """
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        loss = -(log_probs * returns).mean()
        return loss

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
    # ADVANTAGES / RETURNS (AS-IS)
    # =====================================================

    def compute_advantages_and_returns(self, rewards):
        """Return ``(advantages, returns)`` exactly as AS-IS REINFORCE.

        Example:
            >>> t = PolicyGradientTrainer(1.0, 1, "cpu", 8, None)
            >>> adv, ret = t.compute_advantages_and_returns([0.0, 1.0])
            >>> [round(float(x), 4) for x in ret]
            [1.0, 1.0]
        """
        returns: List[float] = []
        g = 0.0
        for r in reversed(rewards):
            g = r + self.gamma * g
            returns.append(g)
        returns.reverse()

        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        advantages = returns_t - returns_t.mean()
        std = advantages.std(unbiased=False)
        if std > 1e-8:
            advantages = advantages / std
        return advantages, returns_t

    # =====================================================
    # EPISODE -> per-episode pack (mode-aware, dict states)
    # =====================================================

    def _episode_to_batch(self, episode: Any) -> Optional[Dict[str, Any]]:
        """Pack one finalized episode into per-step training arrays.

        Returns a dict with the per-step packed-observation dicts, action masks /
        candidate features, integer actions, and the (paper-exact) ``returns``.
        States are kept as packed-obs dicts (not flattened) so the mode-aware net
        can be re-forwarded at update time.
        """
        rewards = self.reward_shaper.compute_reward_sequence(episode)
        steps = episode.steps
        if not steps:
            return None

        states = episode.states if episode.states is not None else [s.state for s in steps]
        actions = episode.actions if episode.actions is not None else [int(s.action) for s in steps]
        action_masks = [s.action_mask for s in steps]
        candidate_features = [s.candidate_features for s in steps]

        if len(states) == 0 or len(actions) == 0:
            return None

        _, returns = self.compute_advantages_and_returns(rewards)
        return {
            "states": states,
            "actions": [int(a) for a in actions],
            "action_masks": action_masks,
            "candidate_features": candidate_features,
            "returns": returns,
        }

    # =====================================================
    # BUILD BATCH (re-collate ragged dict states; 03-05)
    # =====================================================

    def build_batch(self) -> Optional[Dict[str, Any]]:
        """Re-collate all stored episodes into one batched obs dict + targets.

        The mode-aware actor consumes a batched packed-obs dict, so this returns a
        single batch (no flat ``TensorDataset``): the per-step observation dicts,
        masks / candidate features, integer actions, and concatenated returns.
        """
        all_states: List[Dict[str, Any]] = []
        all_actions: List[int] = []
        all_masks: List[Any] = []
        all_cand_feats: List[Any] = []
        all_returns: List[torch.Tensor] = []

        for ep in self.episodes:
            pack = self._episode_to_batch(ep)
            if pack is None:
                continue
            all_states.extend(pack["states"])
            all_actions.extend(pack["actions"])
            all_masks.extend(pack["action_masks"])
            all_cand_feats.extend(pack["candidate_features"])
            all_returns.append(pack["returns"])

        if len(all_states) == 0:
            return None

        returns = torch.cat(all_returns, dim=0).to(self.device)
        actions = torch.tensor(all_actions, dtype=torch.long, device=self.device)
        obs = self.collate_states(
            all_states,
            action_masks=all_masks if self.mode == "PADDED" else None,
            candidate_features=all_cand_feats if self.mode == "ENCODER" else None,
            mode=self.mode,
        )
        return {
            "obs": obs,
            "actions": actions,
            "returns": returns,
        }

    # =====================================================
    # TRAINING (AS-IS loss, mode-aware forward)
    # =====================================================

    def update(self) -> Dict[str, Any]:
        """Run REINFORCE updates; loss = ``-(log_prob * returns).mean()`` (AS-IS)."""
        policy_entry = self.models.get("policy") if self.models else None
        if policy_entry is None:
            return {"loss": 0.0, "steps": 0}
        optimizer = self.optimizers.get("policy")
        if optimizer is None:
            return {"loss": 0.0, "steps": 0}

        policy = policy_entry["model"]
        loss_fn = self.losses.get("policy_loss", self.default_loss)

        batch = self.build_batch()
        if batch is None:
            return {"loss": 0.0, "steps": 0}

        obs = batch["obs"]
        actions = batch["actions"]
        returns = batch["returns"]
        n = int(actions.shape[0])
        # AS-IS minibatching (02-08): shuffle once per epoch and iterate
        # ``batch_size`` minibatches. The collated obs dict has leading dim ``n``
        # on every value, so a minibatch is a plain index-select over dim 0; the
        # per-row padding/masks are preserved, keeping the logits identical to
        # sampling (C4).
        batch_size = self.batch_size if self.batch_size and self.batch_size > 0 else n

        total_loss = 0.0
        total_steps = 0
        # Per-epoch mean loss (for training-log tracing; does NOT touch the loss
        # formula — only records the same per-minibatch losses grouped by epoch).
        epoch_losses: List[float] = []
        policy.train()
        for _ in range(self.epochs):
            perm = torch.randperm(n, device=self.device)
            epoch_loss_sum = 0.0
            epoch_step_sum = 0
            for start in range(0, n, batch_size):
                mb_idx = perm[start : start + batch_size]
                mb_obs = {key: tensor[mb_idx] for key, tensor in obs.items()}
                mb_actions = actions[mb_idx]
                mb_returns = returns[mb_idx]
                logits = policy(mb_obs)
                optimizer.zero_grad()
                loss = loss_fn(logits, mb_actions, mb_returns)
                loss.backward()
                optimizer.step()
                mb_n = int(mb_idx.shape[0])
                total_loss += float(loss.item()) * mb_n
                total_steps += mb_n
                epoch_loss_sum += float(loss.item()) * mb_n
                epoch_step_sum += mb_n
            if epoch_step_sum > 0:
                epoch_losses.append(epoch_loss_sum / epoch_step_sum)

        if total_steps == 0:
            return {"loss": 0.0, "steps": 0, "epoch_losses": []}
        return {
            "loss": total_loss / total_steps,
            "steps": total_steps,
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
                "epochs": self.epochs,
                "batch_size": self.batch_size,
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

    def load_checkpoint(
        self,
        path: str,
        load_optimizers: bool = True,
        strict: bool = True,
    ) -> None:
        """Restore trainer/model/optimizer state from ``path`` (AS-IS)."""
        checkpoint = torch.load(path, map_location=self.device)
        trainer_state = checkpoint.get("trainer_state", {})
        self.gamma = trainer_state.get("gamma", self.gamma)
        self.epochs = trainer_state.get("epochs", self.epochs)
        self.batch_size = trainer_state.get("batch_size", self.batch_size)
        self.mode = trainer_state.get("mode", self.mode)

        self.models = {}
        for key, info in checkpoint.get("models", {}).items():
            class_name = info["class_name"]
            if class_name not in MODEL_REGISTRY:
                raise ValueError(
                    f"Model class '{class_name}' not found in MODEL_REGISTRY"
                )
            model = MODEL_REGISTRY[class_name](**info["config"]).to(self.device)
            model.load_state_dict(info["state_dict"], strict=strict)
            self.models[key] = {
                "model": model,
                "class_name": class_name,
                "config": info["config"],
            }
            print(f"[INFO] Loaded model: {key}")

        self.optimizers = {}
        if load_optimizers:
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
