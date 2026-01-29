from marl_models.base_model import MARLModel, ExperienceBatch
from marl_models.maddpg.agents import ActorNetwork, CriticNetwork
from marl_models.buffer_and_helpers import soft_update, GaussianNoise
import config
import torch
import torch.nn.functional as F
import numpy as np
import os

# 条件导入注意力网络
if config.USE_ATTENTION:
    from marl_models.maddpg.agents import ActorNetworkWithAttention, CriticNetworkWithAttention


class MADDPG(MARLModel):
    def __init__(self, model_name: str, num_agents: int, obs_dim: int, action_dim: int, device: str) -> None:
        super().__init__(model_name, num_agents, obs_dim, action_dim, device)
        self.total_obs_dim: int = num_agents * obs_dim
        self.total_action_dim: int = num_agents * action_dim

        # Create networks for each agent (根据配置选择网络类型)
        if config.USE_ATTENTION:
            # 创建共享的编码器（所有Critic共享，避免参数爆炸：N²→N）
            from marl_models.attention import AttentionEncoder
            self.shared_encoders = torch.nn.ModuleList([AttentionEncoder().to(device) for _ in range(num_agents)])
            self.target_shared_encoders = torch.nn.ModuleList([AttentionEncoder().to(device) for _ in range(num_agents)])
            
            # 使用注意力网络
            self.actors = [ActorNetworkWithAttention(obs_dim, action_dim).to(device) for _ in range(num_agents)]
            self.critics = [CriticNetworkWithAttention(num_agents, obs_dim, action_dim, self.shared_encoders).to(device) for _ in range(num_agents)]
            self.target_actors = [ActorNetworkWithAttention(obs_dim, action_dim).to(device) for _ in range(num_agents)]
            self.target_critics = [CriticNetworkWithAttention(num_agents, obs_dim, action_dim, self.target_shared_encoders).to(device) for _ in range(num_agents)]
        else:
            # 使用原始 MLP 网络
            self.actors = [ActorNetwork(obs_dim, action_dim).to(device) for _ in range(num_agents)]
            self.critics = [CriticNetwork(self.total_obs_dim, self.total_action_dim).to(device) for _ in range(num_agents)]
            self.target_actors = [ActorNetwork(obs_dim, action_dim).to(device) for _ in range(num_agents)]
            self.target_critics = [CriticNetwork(self.total_obs_dim, self.total_action_dim).to(device) for _ in range(num_agents)]
        self._init_target_networks()

        # Create optimizers
        self.actor_optimizers: list[torch.optim.AdamW] = [torch.optim.AdamW(actor.parameters(), lr=config.ACTOR_LR) for actor in self.actors]
        if config.USE_ATTENTION:
            # 修复优化器 Bug: 共享编码器使用单独的优化器（避免 N 次更新）
            self.shared_encoder_optimizers: list[torch.optim.AdamW] = [torch.optim.AdamW(encoder.parameters(), lr=config.CRITIC_LR) for encoder in self.shared_encoders]
            # Critic MLP 优化器（仅包含 MLP 参数）
            self.critic_optimizers: list[torch.optim.AdamW] = [torch.optim.AdamW(critic.mlp_parameters(), lr=config.CRITIC_LR) for critic in self.critics]
        else:
            self.critic_optimizers: list[torch.optim.AdamW] = [torch.optim.AdamW(critic.parameters(), lr=config.CRITIC_LR) for critic in self.critics]

        # Exploration Noise
        self.noise: list[GaussianNoise] = [GaussianNoise() for _ in range(num_agents)]

    def select_actions(self, observations: list[np.ndarray], exploration: bool) -> np.ndarray:
        # Batch all observations and process together for better GPU utilization
        obs_array: np.ndarray = np.array(observations, dtype=np.float32)
        obs_tensor: torch.Tensor = torch.from_numpy(obs_array).to(self.device, non_blocking=True)
        
        actions: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(self.num_agents):
                action: torch.Tensor = self.actors[i](obs_tensor[i:i+1]).squeeze(0)
                if exploration:
                    action_np: np.ndarray = action.cpu().numpy() + self.noise[i].sample()
                else:
                    action_np = action.cpu().numpy()
                actions.append(np.clip(action_np, -1.0, 1.0))

        return np.array(actions)

    def update(self, batch: ExperienceBatch) -> dict:
        assert isinstance(batch, tuple) and len(batch) == 5, "MADDPG expects OffPolicyExperienceBatch (tuple of 5 elements)"
        obs_batch, actions_batch, rewards_batch, next_obs_batch, dones_batch = batch
        obs_tensor: torch.Tensor = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
        actions_tensor: torch.Tensor = torch.as_tensor(actions_batch, dtype=torch.float32, device=self.device)
        rewards_tensor: torch.Tensor = torch.as_tensor(rewards_batch, dtype=torch.float32, device=self.device)
        next_obs_tensor: torch.Tensor = torch.as_tensor(next_obs_batch, dtype=torch.float32, device=self.device)
        dones_tensor: torch.Tensor = torch.as_tensor(dones_batch, dtype=torch.float32, device=self.device)

        batch_size: int = obs_tensor.shape[0]  # Get batch size from the data
        obs_flat: torch.Tensor = obs_tensor.reshape(batch_size, -1)
        next_obs_flat: torch.Tensor = next_obs_tensor.reshape(batch_size, -1)
        actions_flat: torch.Tensor = actions_tensor.reshape(batch_size, -1)

        # Track training statistics
        total_actor_loss: float = 0.0
        total_critic_loss: float = 0.0
        total_actor_grad_norm: float = 0.0
        total_critic_grad_norm: float = 0.0
        q_values: list[float] = []
        
        # Pre-compute encodings for attention networks OUTSIDE no_grad (需要梯度流)
        if config.USE_ATTENTION:
            joint_encoded = self.critics[0].encode_observations(obs_flat)
        else:
            joint_encoded = None
        
        # Pre-compute all actions once (without gradients) for efficiency
        # This reduces actor forward passes from O(N²) to O(2N)
        with torch.no_grad():
            all_actions_detached = [self.actors[i](obs_tensor[:, i, :]) for i in range(self.num_agents)]
            # Pre-compute target actions for all agents (used in critic updates)
            next_actions: list[torch.Tensor] = [self.target_actors[i](next_obs_tensor[:, i, :]) for i in range(self.num_agents)]
            next_actions_tensor: torch.Tensor = torch.cat(next_actions, dim=1)
            
            # Pre-compute target encodings (不需要梯度)
            if config.USE_ATTENTION:
                target_joint_encoded = self.target_critics[0].encode_observations(next_obs_flat)
            else:
                target_joint_encoded = None

        # Zero gradients for shared encoders (if using attention)
        if config.USE_ATTENTION:
            for opt in self.shared_encoder_optimizers:
                opt.zero_grad(set_to_none=True)
        
        for agent_idx in range(self.num_agents):
            # Determine if this is the last backward pass (for graph retention)
            is_last_agent = (agent_idx == self.num_agents - 1)
            
            # Update Critic
            with torch.no_grad():
                target_q_value: torch.Tensor = self.target_critics[agent_idx](next_obs_flat, next_actions_tensor, target_joint_encoded)
                agent_reward: torch.Tensor = rewards_tensor[:, agent_idx].unsqueeze(1)
                agent_done: torch.Tensor = dones_tensor[:, agent_idx].unsqueeze(1)
                y: torch.Tensor = agent_reward + config.DISCOUNT_FACTOR * target_q_value * (1 - agent_done)

            current_q_value: torch.Tensor = self.critics[agent_idx](obs_flat, actions_flat, joint_encoded)

            critic_loss: torch.Tensor = F.mse_loss(current_q_value, y)
            self.critic_optimizers[agent_idx].zero_grad(set_to_none=True)
            # In attention mode, the critics share encoder modules whose parameters are updated
            # via their own optimizers after all agents have backpropagated.
            # To accumulate gradients on these shared encoders across agents, we must retain the
            # computation graph for every agent *except* the last one. The last backward call can
            # safely free the graph.
            #
            # When attention is disabled, critic networks do not share encoders in this way, so
            # each agent's computation graph is independent and there is no need to retain it
            # across backward calls.
            retain_graph = config.USE_ATTENTION and not is_last_agent
            critic_loss.backward(retain_graph=retain_graph)
            # Only clip MLP parameters (shared encoders clipped once outside loop)
            if config.USE_ATTENTION:
                critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critics[agent_idx].mlp_parameters(), config.MAX_GRAD_NORM)
            else:
                critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critics[agent_idx].parameters(), config.MAX_GRAD_NORM)
            self.critic_optimizers[agent_idx].step()  # Only updates MLP parameters

            # Update Actor
            # Use pre-computed actions for other agents, recompute only current agent's action with gradients
            # This optimization reduces forward passes from N² to 2N
            current_joint_actions: list[torch.Tensor] = []
            for i in range(self.num_agents):
                if i == agent_idx:
                    action = self.actors[i](obs_tensor[:, i, :])  # Compute with gradients
                else:
                    action = all_actions_detached[i]  # Use pre-computed detached action
                current_joint_actions.append(action)
            
            pred_actions_tensor: torch.Tensor = torch.stack(current_joint_actions, dim=1)
            pred_actions_flat: torch.Tensor = pred_actions_tensor.reshape(batch_size, -1)

            # Detach joint_encoded to prevent actor from modifying critic's state representation
            actor_joint_encoded = joint_encoded.detach() if config.USE_ATTENTION else joint_encoded
            actor_loss: torch.Tensor = -self.critics[agent_idx](obs_flat, pred_actions_flat, actor_joint_encoded).mean()
            self.actor_optimizers[agent_idx].zero_grad(set_to_none=True)
            # No need to retain graph: each actor's computation graph is independent
            actor_loss.backward()
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actors[agent_idx].parameters(), config.MAX_GRAD_NORM)
            self.actor_optimizers[agent_idx].step()
            
            # Collect statistics
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_actor_grad_norm += float(actor_grad_norm)
            total_critic_grad_norm += float(critic_grad_norm)
            q_values.extend(current_q_value.detach().cpu().numpy().flatten().tolist())
        
        # Step shared encoder optimizers after all gradients have accumulated
        if config.USE_ATTENTION:
            for i, opt in enumerate(self.shared_encoder_optimizers):
                torch.nn.utils.clip_grad_norm_(self.shared_encoders[i].parameters(), config.MAX_GRAD_NORM)
                opt.step()
        
        # Soft update target networks AFTER all agents have updated their main networks
        if config.USE_ATTENTION:
            # 修复 N×τ Bug: 共享编码器只更新一次（循环外）
            for i in range(self.num_agents):
                soft_update(self.target_shared_encoders[i], self.shared_encoders[i], config.UPDATE_FACTOR)
            # 各智能体 Actor 和 Critic (仅 MLP 部分) 分别更新
            for agent_idx in range(self.num_agents):
                soft_update(self.target_actors[agent_idx], self.actors[agent_idx], config.UPDATE_FACTOR)
                # 仅更新 MLP 参数，不包括共享编码器（通过 mlp_parameters() 隔离）
                soft_update(self.target_critics[agent_idx].mlp_parameters(), 
                            self.critics[agent_idx].mlp_parameters(), config.UPDATE_FACTOR)
        else:
            # 原始 MLP 模式：正常更新所有参数
            for agent_idx in range(self.num_agents):
                soft_update(self.target_actors[agent_idx], self.actors[agent_idx], config.UPDATE_FACTOR)
                soft_update(self.target_critics[agent_idx], self.critics[agent_idx], config.UPDATE_FACTOR)
        
        # Return averaged statistics
        return {
            "actor_loss": total_actor_loss / self.num_agents,
            "critic_loss": total_critic_loss / self.num_agents,
            "actor_grad_norm": total_actor_grad_norm / self.num_agents,
            "critic_grad_norm": total_critic_grad_norm / self.num_agents,
            "q_value_mean": float(np.mean(q_values)),
            "q_value_std": float(np.std(q_values)),
            "noise_scale": self.noise[0].scale,  # All agents share same noise scale
        }

    def _init_target_networks(self) -> None:
        if config.USE_ATTENTION:
            # 初始化共享编码器
            for encoder, target_encoder in zip(self.shared_encoders, self.target_shared_encoders):
                target_encoder.load_state_dict(encoder.state_dict())
        for actor, target_actor in zip(self.actors, self.target_actors):
            target_actor.load_state_dict(actor.state_dict())
        for critic, target_critic in zip(self.critics, self.target_critics):
            target_critic.load_state_dict(critic.state_dict())

    def reset(self) -> None:
        for n in self.noise:
            n.reset()

    def save(self, directory: str) -> None:
        # Save shared encoders once (if using attention)
        if config.USE_ATTENTION:
            shared_encoder_state = {
                f"encoder_{i}": self.shared_encoders[i].state_dict()
                for i in range(self.num_agents)
            }
            shared_encoder_state.update({
                f"target_encoder_{i}": self.target_shared_encoders[i].state_dict()
                for i in range(self.num_agents)
            })
            shared_encoder_state.update({
                f"encoder_optimizer_{i}": self.shared_encoder_optimizers[i].state_dict()
                for i in range(self.num_agents)
            })
            torch.save(shared_encoder_state, os.path.join(directory, "shared_encoders.pth"))
        
        # Save per-agent networks
        for i in range(self.num_agents):
            # Filter out shared encoders from critic state_dict to avoid duplication
            if config.USE_ATTENTION:
                critic_state = {k: v for k, v in self.critics[i].state_dict().items() if not k.startswith('encoders.')}
                target_critic_state = {k: v for k, v in self.target_critics[i].state_dict().items() if not k.startswith('encoders.')}
            else:
                critic_state = self.critics[i].state_dict()
                target_critic_state = self.target_critics[i].state_dict()
            
            torch.save(
                {
                    "actor": self.actors[i].state_dict(),
                    "critic": critic_state,
                    "target_actor": self.target_actors[i].state_dict(),
                    "target_critic": target_critic_state,
                    "actor_optimizer": self.actor_optimizers[i].state_dict(),
                    "critic_optimizer": self.critic_optimizers[i].state_dict(),
                    "noise_scale": self.noise[i].scale,
                },
                os.path.join(directory, f"agent_{i}.pth"),
            )

    def load(self, directory: str) -> None:
        if not os.path.exists(directory):
            raise FileNotFoundError(f"❌ Model directory not found: {directory}")

        # Load shared encoders (if using attention)
        if config.USE_ATTENTION:
            encoder_path = os.path.join(directory, "shared_encoders.pth")
            if os.path.exists(encoder_path):
                encoder_checkpoint = torch.load(encoder_path, map_location=self.device)
                for i in range(self.num_agents):
                    self.shared_encoders[i].load_state_dict(encoder_checkpoint[f"encoder_{i}"])
                    self.target_shared_encoders[i].load_state_dict(encoder_checkpoint[f"target_encoder_{i}"])
                    self.shared_encoder_optimizers[i].load_state_dict(encoder_checkpoint[f"encoder_optimizer_{i}"])

        for i in range(self.num_agents):
            agent_path: str = os.path.join(directory, f"agent_{i}.pth")
            if not os.path.exists(agent_path):
                raise FileNotFoundError(f"❌ Model file not found: {agent_path}")
            checkpoint: dict = torch.load(agent_path, map_location=self.device)
            self.actors[i].load_state_dict(checkpoint["actor"])
            # Filter out encoders from old checkpoints to prevent overwriting shared encoders
            if config.USE_ATTENTION:
                critic_state = {k: v for k, v in checkpoint["critic"].items() if not k.startswith('encoders.')}
                target_critic_state = {k: v for k, v in checkpoint["target_critic"].items() if not k.startswith('encoders.')}
                self.critics[i].load_state_dict(critic_state, strict=False)
                self.target_critics[i].load_state_dict(target_critic_state, strict=False)
            else:
                self.critics[i].load_state_dict(checkpoint["critic"])
                self.target_critics[i].load_state_dict(checkpoint["target_critic"])
            self.actor_optimizers[i].load_state_dict(checkpoint["actor_optimizer"])
            self.critic_optimizers[i].load_state_dict(checkpoint["critic_optimizer"])
            # Restore noise scale (backward compatible: use default if not present)
            if "noise_scale" in checkpoint:
                self.noise[i].scale = checkpoint["noise_scale"]
