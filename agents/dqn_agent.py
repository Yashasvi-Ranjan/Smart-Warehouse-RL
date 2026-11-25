"""Deep Q-Network agent implemented from scratch in PyTorch (no RL library).

Vanilla DQN (Mnih et al. 2015): MLP Q-network, separate target network (hard update every
``target_update_every`` gradient steps, or Polyak soft update with ``tau``), epsilon-greedy with linear decay
over environment steps, uniform experience replay, Huber loss, Adam, gradient clipping.
Optional extras (off by default): ``double_dqn`` and ``use_action_mask``.

The agent only reads the observation and (if masking is on) ``env.unwrapped.action_masks()``.
"""
from __future__ import annotations

import json
import os
from typing import Any, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.base import Agent
from agents.replay_buffer import ReplayBuffer

N_ACTIONS = 7
HPARAM_KEYS = ("obs_dim", "n_actions", "hidden", "gamma", "lr", "batch_size", "buffer_size", "learning_starts",
               "target_update_every", "tau", "epsilon_start", "epsilon_end", "epsilon_decay_steps",
               "double_dqn", "grad_clip", "loss", "use_action_mask", "train_freq")


def resolve_device(device: str = "cpu") -> torch.device:
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but not available")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return torch.device(device)


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: Sequence[int] = (128, 128)):
        super().__init__()
        layers, d = [], obs_dim
        for h in hidden:
            layers += [nn.Linear(d, int(h)), nn.ReLU()]
            d = int(h)
        layers.append(nn.Linear(d, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent(Agent):
    name = "dqn"

    def __init__(self, obs_dim: int = 43, n_actions: int = N_ACTIONS, hidden: Sequence[int] = (128, 128),
                 gamma: float = 0.99, lr: float = 1e-3, batch_size: int = 64, buffer_size: int = 100_000,
                 learning_starts: int = 1000, target_update_every: int = 500, tau: Optional[float] = None,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.05, epsilon_decay_steps: int = 60_000,
                 double_dqn: bool = False, grad_clip: Optional[float] = 10.0, loss: str = "huber",
                 use_action_mask: bool = False, train_freq: int = 1, seed: Optional[int] = None,
                 device: str = "cpu"):
        if loss not in ("huber", "mse"):
            raise ValueError("loss must be 'huber' or 'mse'")
        self.obs_dim, self.n_actions = int(obs_dim), int(n_actions)
        self.hidden = [int(h) for h in hidden]
        self.gamma, self.lr = float(gamma), float(lr)
        self.batch_size, self.buffer_size = int(batch_size), int(buffer_size)
        self.learning_starts = int(learning_starts)
        self.target_update_every = int(target_update_every)
        self.tau = None if tau is None else float(tau)
        self.epsilon_start, self.epsilon_end = float(epsilon_start), float(epsilon_end)
        self.epsilon_decay_steps = int(epsilon_decay_steps)
        self.double_dqn = bool(double_dqn)
        self.grad_clip = None if grad_clip is None else float(grad_clip)
        self.loss_name = loss
        self.use_action_mask = bool(use_action_mask)
        self.train_freq = int(train_freq)
        self.seed = seed
        self.device = resolve_device(device)

        if seed is not None:
            torch.manual_seed(int(seed))  # network init
        self.q_net = QNetwork(self.obs_dim, self.n_actions, self.hidden).to(self.device)
        self.target_net = QNetwork(self.obs_dim, self.n_actions, self.hidden).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.lr)
        self.buffer: Optional[ReplayBuffer] = None
        self._rng = np.random.default_rng(seed)
        self.env_steps = 0      # environment steps seen (drives epsilon)
        self.grad_steps = 0     # gradient updates performed (drives target updates)
        self.epsilon = self.epsilon_start

    # ------------------------------------------------------------------ helpers
    def hparams(self) -> dict:
        d = {k: getattr(self, k) for k in HPARAM_KEYS if k != "loss"}
        d["loss"] = self.loss_name
        return d

    def make_buffer(self) -> ReplayBuffer:
        seed = None if self.seed is None else int(self.seed) + 12345
        self.buffer = ReplayBuffer(self.buffer_size, self.obs_dim, self.n_actions, seed=seed, device=self.device)
        return self.buffer

    def epsilon_at(self, step: int) -> float:
        """Linear decay epsilon_start -> epsilon_end over ``epsilon_decay_steps`` env steps, then constant."""
        n = self.epsilon_decay_steps
        if n <= 0:
            return self.epsilon_end
        frac = min(1.0, max(0.0, step / n))
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    @torch.no_grad()
    def q_values(self, obs: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(np.asarray(obs, np.float32), device=self.device).unsqueeze(0)
        return self.q_net(x)[0].cpu().numpy()

    def greedy_action(self, obs: np.ndarray, mask: Optional[np.ndarray] = None) -> int:
        q = self.q_values(obs)
        if mask is not None and mask.any():
            q = np.where(mask, q, -np.inf)
        return int(np.argmax(q))

    def select_action(self, obs: np.ndarray, epsilon: float, mask: Optional[np.ndarray] = None) -> int:
        """Epsilon-greedy; with masking the exploration is uniform over valid actions."""
        if epsilon > 0 and self._rng.random() < epsilon:
            if mask is not None and mask.any():
                return int(self._rng.choice(np.flatnonzero(mask)))
            return int(self._rng.integers(self.n_actions))
        return self.greedy_action(obs, mask)

    # ------------------------------------------------------------------ learning
    @torch.no_grad()
    def compute_targets(self, batch: dict) -> torch.Tensor:
        """y = r + gamma * max_a' Q_target(s', a') * (1 - terminated).

        Only ``terminated`` (delivered / battery depleted: true terminal states) cuts the bootstrap. Time-limit
        *truncation* is not stored as done, so we keep bootstrapping through it: the state after a time-out still
        has value; treating it as terminal would teach the agent a fake end of the world.
        """
        nxt = self.target_net(batch["next_obs"])
        mask = batch.get("next_mask")
        if self.use_action_mask and mask is not None:
            # (rows with an all-False mask cannot occur: env guarantees a non-empty mask)
            nxt_masked = nxt.masked_fill(~mask, float("-inf"))
        else:
            nxt_masked = nxt
        if self.double_dqn:
            # online net selects the action, target net evaluates it
            online = self.q_net(batch["next_obs"])
            if self.use_action_mask and mask is not None:
                online = online.masked_fill(~mask, float("-inf"))
            a_star = online.argmax(dim=1, keepdim=True)
            max_next = nxt.gather(1, a_star).squeeze(1)
        else:
            max_next = nxt_masked.max(dim=1).values
        return batch["rewards"] + self.gamma * max_next * (1.0 - batch["terminated"])

    def learn_on_batch(self, batch: dict) -> float:
        y = self.compute_targets(batch)
        q = self.q_net(batch["obs"]).gather(1, batch["actions"].unsqueeze(1)).squeeze(1)
        loss = F.smooth_l1_loss(q, y) if self.loss_name == "huber" else F.mse_loss(q, y)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.grad_clip:
            nn.utils.clip_grad_norm_(self.q_net.parameters(), self.grad_clip)
        self.optimizer.step()
        self.grad_steps += 1
        self._update_target()
        return float(loss.item())

    def _update_target(self) -> None:
        if self.tau is not None:
            with torch.no_grad():  # soft (Polyak) update every gradient step
                for tp, p in zip(self.target_net.parameters(), self.q_net.parameters()):
                    tp.mul_(1.0 - self.tau).add_(p, alpha=self.tau)
        elif self.grad_steps % self.target_update_every == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())  # hard update

    def observe(self, obs, action, reward, next_obs, terminated, next_mask=None) -> Optional[float]:
        """Store a transition, advance the step counter and (if due) do one gradient step. Returns loss or None."""
        if self.buffer is None:
            self.make_buffer()
        self.buffer.add(obs, action, reward, next_obs, terminated, next_mask)
        self.env_steps += 1
        self.epsilon = self.epsilon_at(self.env_steps)
        if (len(self.buffer) >= max(self.learning_starts, self.batch_size)
                and self.env_steps % self.train_freq == 0):
            return self.learn_on_batch(self.buffer.sample(self.batch_size))
        return None

    # ------------------------------------------------------------------ Agent API
    def act(self, obs: np.ndarray, env: Optional[Any] = None, deterministic: bool = True) -> int:
        mask = None
        if self.use_action_mask and env is not None:
            mask = env.unwrapped.action_masks()
        eps = 0.0 if deterministic else self.epsilon
        return self.select_action(obs, eps, mask)

    # ------------------------------------------------------------------ persistence
    def save(self, path: str, meta: Optional[dict] = None, extra: Optional[dict] = None) -> None:
        path = str(path)
        if not path.endswith(".pt"):
            path += ".pt"
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({"q_net": self.q_net.state_dict(), "target_net": self.target_net.state_dict(),
                    "optimizer": self.optimizer.state_dict(), "hparams": self.hparams(),
                    "epsilon": self.epsilon, "env_steps": self.env_steps, "grad_steps": self.grad_steps,
                    "seed": self.seed, "rng_state": self._rng.bit_generator.state,
                    "extra": extra or {}}, path)
        if meta is not None:
            with open(os.path.splitext(path)[0] + ".meta.json", "w") as fh:
                json.dump(meta, fh, indent=2)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "DQNAgent":
        ck = torch.load(str(path), map_location="cpu", weights_only=False)
        hp = dict(ck["hparams"])
        agent = cls(seed=ck.get("seed"), device=device, **hp)
        agent.q_net.load_state_dict(ck["q_net"])
        agent.target_net.load_state_dict(ck["target_net"])
        agent.optimizer.load_state_dict(ck["optimizer"])
        agent.epsilon = float(ck["epsilon"])
        agent.env_steps = int(ck["env_steps"])
        agent.grad_steps = int(ck["grad_steps"])
        if ck.get("rng_state") is not None:
            agent._rng.bit_generator.state = ck["rng_state"]
        agent.q_net.eval()
        agent.loaded_extra = ck.get("extra", {})
        return agent
