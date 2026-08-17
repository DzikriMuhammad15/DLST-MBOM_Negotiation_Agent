"""IOP trainer (``02-08``) — AS-IS cross-entropy opponent-action prediction.

Ported AS-IS from the paper code. The Imagined-Opponent-Policy network
(:class:`~negotiation.models.iop_model.IOPNetwork`) is trained **supervised** to
predict the opponent's chosen action, with the exact AS-IS loss:

    ``loss = -mean( log_softmax(logits)[range, action] )``   (cross-entropy / NLL)

IOP works on a flat per-step memory (``add_step`` / ``reset_step``) rather than
finalized episodes. Each stored :class:`~negotiation.data_type.step.Step` carries
the opponent state and the observed opponent action; the state is turned into a
flat vector by the owning agent (``build_state_basis``) when stored as a dict, or
used directly when already a vector. ``copy_iop_policy_weight`` synchronises
weights across MBOM recursion levels (same architecture / init params).

The IOP head is a plain MLP over a flat state vector and is **not** affected by
the input mode (the front-end that builds the state vector lives in the agent).
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from negotiation.data_type.step import Step
from negotiation.trainers.trainer import Trainer, TrainerInterface

MODEL_REGISTRY: Dict[str, Any] = {}


class IOPTrainer(Trainer, TrainerInterface):
    """Cross-entropy trainer for the IOP network (opponent action prediction).

    ``self.models`` format (via :meth:`attach_models`)::

        {"policy": {"model": iop_net, "optimizer": opt, "config": {...}}}
        # optionally a "value" entry (ignored by the loss, kept for symmetry).

    Example:
        >>> t = IOPTrainer(epoch=1, device="cpu", batch_size=8,
        ...                reward_shaper=None, gradient_clip_norm=1.0)
        >>> t.memory
        []
    """

    def __init__(
        self,
        epoch: int,
        device: str,
        batch_size: int,
        reward_shaper: Any,
        gradient_clip_norm: float,
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
        self.epoch = epoch
        self.device = device
        self.memory: List[Step] = []
        self.agent = agent
        self.gradient_clip_norm = gradient_clip_norm

    # =====================================================
    # STEP MEMORY
    # =====================================================

    def reset_step(self) -> None:
        """Clear the per-step supervised memory."""
        self.memory = []

    def add_step(self, step: Step) -> None:
        """Append one (opponent state, opponent action) :class:`Step` to memory."""
        self.memory.append(step)

    # =====================================================
    # LOSS (AS-IS cross-entropy / NLL)
    # =====================================================

    def default_loss(self, logits: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Cross-entropy NLL over the selected opponent action
        """
        log_probs = F.log_softmax(logits, dim=-1)
        selected_log_probs = log_probs.gather(
            dim=1, index=action.unsqueeze(1)
        ).squeeze(1)
        loss = -selected_log_probs.mean()
        return loss

    # =====================================================
    # INTERFACE IMPLEMENTATIONS
    # =====================================================

    def store_episode(self, episode: Any) -> None:
        """IOP uses per-step memory (``add_step``); episodes are not stored."""

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

    def attach_agent(self, agent: Any) -> None:
        self.agent = agent

    def attach_losses(self, dict_of_losses: Dict[str, Any]) -> None:
        self.losses = dict_of_losses

    def _episode_to_batch(self, episode: Any) -> Any:
        """IOP is step-based; episode conversion is unused (kept for the ABC)."""
        return None

    # =====================================================
    # BUILD BATCH (flat state vectors)
    # =====================================================

    def build_batch(self):
        """Build a ``(states, actions)`` ``DataLoader`` from step memory (AS-IS)."""
        from torch.utils.data import DataLoader, TensorDataset

        all_states: List[Any] = []
        all_actions: List[Any] = []
        for step in self.memory:
            if isinstance(step.state, dict):
                state_vec = self.agent.build_state_basis(step.state)
            else:
                state_vec = step.state
            all_states.append(state_vec)
            all_actions.append(step.action)

        if len(all_states) == 0:
            return None

        states = torch.as_tensor(all_states, dtype=torch.float32)
        actions = torch.as_tensor(all_actions, dtype=torch.long)
        dataset = TensorDataset(states, actions)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    # =====================================================
    # UPDATE (AS-IS)
    # =====================================================

    def update(self) -> Dict[str, Any]:
        """Run supervised cross-entropy updates over the IOP policy (AS-IS)."""
        if not self.models or "policy" not in self.models:
            return {"loss": 0.0}
        policy = self.models["policy"]["model"]
        policy_optimizer = self.optimizers["policy"]

        loss_fn = self.losses.get("iop_loss", self.default_loss)

        data_loader = self.build_batch()
        if data_loader is None:
            return {"loss": 0.0}

        total_loss = 0.0
        total_steps = 0
        # Per-epoch mean loss (training-log tracing only; loss formula untouched).
        epoch_losses: List[float] = []
        policy.train()
        for _ in range(self.epoch):
            epoch_loss_sum = 0.0
            epoch_step_sum = 0
            for batch in data_loader:
                states, actions = batch
                states = states.to(self.device)
                actions = actions.to(self.device)

                logits = policy(states)
                loss = loss_fn(logits, actions)
                policy_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), self.gradient_clip_norm)
                policy_optimizer.step()

                total_loss += float(loss.item()) * states.size(0)
                total_steps += states.size(0)
                epoch_loss_sum += float(loss.item()) * states.size(0)
                epoch_step_sum += states.size(0)
            if epoch_step_sum > 0:
                epoch_losses.append(epoch_loss_sum / epoch_step_sum)

        if total_steps == 0:
            return {"loss": 0.0, "epoch_losses": []}
        return {"loss": total_loss / total_steps, "epoch_losses": epoch_losses}

    def reset_episodes(self) -> None:
        """No-op: IOP uses ``reset_step`` for its step memory."""

    def copy_iop_policy_weight(self, iop_policy: Any) -> None:
        """Copy weights from another IOP policy (same architecture/init) — AS-IS.

        Used by MBOM to synchronise the policy across recursion levels.
        """
        self.models.get("policy").get("model").load_state_dict(iop_policy.state_dict())

    # =====================================================
    # CHECKPOINT (AS-IS)
    # =====================================================

    def save_checkpoint(self, path: str, extra: Any = None) -> None:
        """Persist trainer state + each model/optimizer state_dict (AS-IS)."""
        checkpoint: Dict[str, Any] = {
            "trainer_state": {
                "gradient_clip_norm": self.gradient_clip_norm,
                "epoch": self.epoch,
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
        self.gradient_clip_norm = trainer_state.get("gradient_clip_norm", self.gradient_clip_norm)
        self.epoch = trainer_state.get("epoch", self.epoch)

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
