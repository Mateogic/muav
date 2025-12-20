from marl_models.base_model import MARLModel, ExperienceBatch
from marl_models.mappo.agents import ActorNetwork, CriticNetwork
import config
import numpy as np
import os
import torch
from torch.distributions import Normal


class MAPPO(MARLModel):
    def __init__(self, model_name: str, num_agents: int, obs_dim: int, action_dim: int, state_dim: int, device: str) -> None:
        super().__init__(model_name, num_agents, obs_dim, action_dim, device)
        self.state_dim: int = state_dim

        # Create networks
        self.actors: ActorNetwork = ActorNetwork(obs_dim, action_dim).to(device)
        self.critics: CriticNetwork = CriticNetwork(state_dim, num_agents).to(device)

        # Create optimizers
        self.actor_optimizer: torch.optim.AdamW = torch.optim.AdamW(self.actors.parameters(), lr=config.ACTOR_LR)
        self.critic_optimizer: torch.optim.AdamW = torch.optim.AdamW(self.critics.parameters(), lr=config.CRITIC_LR)

    def select_actions(self, observations: list[np.ndarray], exploration: bool) -> np.ndarray:
        # Convert observations to tensor once (avoid repeated conversions)
        obs_array: np.ndarray = np.array(observations, dtype=np.float32)
        obs_tensor: torch.Tensor = torch.from_numpy(obs_array).to(self.device, non_blocking=True)
        with torch.no_grad():
            dist: Normal = self.actors(obs_tensor)
            if exploration:
                actions: torch.Tensor = dist.sample()  # Stochastic actions for exploration
            else:
                actions = dist.mean  # Deterministic actions for evaluation
            # Clip on GPU before transferring to CPU
            actions = torch.clamp(actions, -1.0, 1.0)

        return actions.cpu().numpy()

    def get_action_and_value(self, obs: np.ndarray, state: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample raw actions for PPO; caller is responsible for clipping before env step."""
        # Use non_blocking for async data transfer
        obs_tensor: torch.Tensor = torch.from_numpy(obs.astype(np.float32)).to(self.device, non_blocking=True)
        state_tensor: torch.Tensor = torch.from_numpy(state.astype(np.float32)).to(self.device, non_blocking=True)

        with torch.no_grad():
            dist: Normal = self.actors(obs_tensor)
            raw_actions: torch.Tensor = dist.sample()
            # PPO log_prob must correspond to the raw (unclipped) actions
            log_probs: torch.Tensor = dist.log_prob(raw_actions).sum(dim=-1)

            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(0)
            values: torch.Tensor = self.critics(state_tensor).squeeze(0)

        return raw_actions.cpu().numpy(), log_probs.cpu().numpy(), values.cpu().numpy()

    def update(self, batch: ExperienceBatch) -> None:
        assert isinstance(batch, dict), "MAPPO expects OnPolicyExperienceBatch (dict)"
        obs_batch: torch.Tensor = batch["obs"]
        actions_batch: torch.Tensor = batch["actions"]
        old_log_probs_batch: torch.Tensor = batch["old_log_probs"]
        advantages_batch: torch.Tensor = batch["advantages"]
        returns_batch: torch.Tensor = batch["returns"]
        states_batch: torch.Tensor = batch["states"]
        old_values_batch: torch.Tensor = batch["old_values"]
        agent_indices_batch: torch.Tensor = batch["agent_indices"]

        # Normalize advantages with clipping for numerical stability
        advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)
        advantages_batch = torch.clamp(advantages_batch, -10.0, 10.0)

        # Critic Loss
        # The states_batch contains repeated states for each agent after shuffling.
        # We use agent_indices_batch to correctly extract the value for each agent.
        batch_size = states_batch.shape[0]
        values_all: torch.Tensor = self.critics(states_batch)  # Shape: (batch_size, num_agents)
        # Extract the value for the corresponding agent using the tracked indices
        values: torch.Tensor = values_all[torch.arange(batch_size, device=values_all.device), agent_indices_batch]
        # Value clipping - apply clipping to the extracted values
        values_clipped: torch.Tensor = old_values_batch + torch.clamp(values - old_values_batch, -config.PPO_VALUE_CLIP_EPS, config.PPO_VALUE_CLIP_EPS)
        vf_loss1: torch.Tensor = (values - returns_batch).pow(2)
        vf_loss2: torch.Tensor = (values_clipped - returns_batch).pow(2)
        critic_loss: torch.Tensor = 0.5 * torch.max(vf_loss1, vf_loss2).mean()

        # Actor Loss
        dist: Normal = self.actors(obs_batch)
        new_log_probs: torch.Tensor = dist.log_prob(actions_batch).sum(dim=-1)
        ratio: torch.Tensor = torch.exp(new_log_probs - old_log_probs_batch)

        # PPO surrogate loss
        surr1: torch.Tensor = ratio * advantages_batch
        surr2: torch.Tensor = torch.clamp(ratio, 1.0 - config.PPO_CLIP_EPS, 1.0 + config.PPO_CLIP_EPS) * advantages_batch
        actor_loss: torch.Tensor = -torch.min(surr1, surr2).mean()

        # Adding entropy bonus for exploration
        entropy_loss: torch.Tensor = dist.entropy().mean()
        actor_loss -= config.PPO_ENTROPY_COEF * entropy_loss

        # Combined update: zero gradients once, compute both losses, then step
        # Using set_to_none=True is faster than setting to zero
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        
        # Combined backward pass for better GPU utilization
        total_loss: torch.Tensor = actor_loss + critic_loss
        total_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.actors.parameters(), config.MAX_GRAD_NORM)
        torch.nn.utils.clip_grad_norm_(self.critics.parameters(), config.MAX_GRAD_NORM)
        self.actor_optimizer.step()
        self.critic_optimizer.step()

    def reset(self) -> None:
        pass  # Nothing to reset

    def save(self, directory: str) -> None:
        torch.save(
            {
                "actor": self.actors.state_dict(),
                "critic": self.critics.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
            },
            os.path.join(directory, "mappo.pth"),
        )

    def load(self, directory: str) -> None:
        path: str = os.path.join(directory, "mappo.pth")
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ Model file not found: {path}")
        checkpoint: dict = torch.load(path, map_location=self.device)
        self.actors.load_state_dict(checkpoint["actor"])
        self.critics.load_state_dict(checkpoint["critic"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
