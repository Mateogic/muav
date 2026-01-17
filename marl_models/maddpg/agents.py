import config
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Network architecture optimizations for training stability:
# 1. Orthogonal initialization - better gradient flow at init
# 2. LayerNorm - stable for RL (batch-independent normalization)
# 3. LeakyReLU - avoids dead neurons problem
# 4. Residual connections (Critic) - improves gradient flow in deep networks

# 条件导入注意力模块
if config.USE_ATTENTION:
    from marl_models.attention import AttentionEncoder


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super(ActorNetwork, self).__init__()
        self.fc1: nn.Linear = layer_init(nn.Linear(obs_dim, config.MLP_HIDDEN_DIM))
        self.ln1: nn.LayerNorm = nn.LayerNorm(config.MLP_HIDDEN_DIM)
        self.fc2: nn.Linear = layer_init(nn.Linear(config.MLP_HIDDEN_DIM, config.MLP_HIDDEN_DIM))
        self.ln2: nn.LayerNorm = nn.LayerNorm(config.MLP_HIDDEN_DIM)
        self.out: nn.Linear = layer_init(nn.Linear(config.MLP_HIDDEN_DIM, action_dim), std=0.01)
        
        self.activation = nn.LeakyReLU(0.01)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = self.activation(self.ln1(self.fc1(input)))
        x = self.activation(self.ln2(self.fc2(x)))
        return torch.tanh(self.out(x))


class CriticNetwork(nn.Module):
    def __init__(self, total_obs_dim: int, total_action_dim: int) -> None:
        super(CriticNetwork, self).__init__()
        # First layer: input projection (no residual, dimension change)
        self.fc1: nn.Linear = layer_init(nn.Linear(total_obs_dim + total_action_dim, config.MLP_HIDDEN_DIM))
        self.ln1: nn.LayerNorm = nn.LayerNorm(config.MLP_HIDDEN_DIM)
        
        # Hidden layers with residual connections (same dimension)
        self.fc2: nn.Linear = layer_init(nn.Linear(config.MLP_HIDDEN_DIM, config.MLP_HIDDEN_DIM))
        self.ln2: nn.LayerNorm = nn.LayerNorm(config.MLP_HIDDEN_DIM)
        self.fc3: nn.Linear = layer_init(nn.Linear(config.MLP_HIDDEN_DIM, config.MLP_HIDDEN_DIM))
        self.ln3: nn.LayerNorm = nn.LayerNorm(config.MLP_HIDDEN_DIM)
        
        # Output layer
        self.out: nn.Linear = layer_init(nn.Linear(config.MLP_HIDDEN_DIM, 1))
        
        self.activation = nn.LeakyReLU(0.01)

    def forward(self, joint_obs: torch.Tensor, joint_action: torch.Tensor, _joint_encoded: torch.Tensor | None = None) -> torch.Tensor:
        x: torch.Tensor = torch.cat([joint_obs, joint_action], dim=1)
        
        # First layer (no residual due to dimension change)
        x = self.activation(self.ln1(self.fc1(x)))
        
        # Second layer with residual connection
        residual = x
        x = self.activation(self.ln2(self.fc2(x)))
        x = x + residual  # Residual connection
        
        # Third layer with residual connection
        residual = x
        x = self.activation(self.ln3(self.fc3(x)))
        x = x + residual  # Residual connection
        
        return self.out(x)


class ActorNetworkWithAttention(nn.Module):
    """使用注意力机制的 Actor 网络"""

    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        # 注意力编码器
        self.encoder = AttentionEncoder()
        encoder_output_dim = self.encoder.output_dim  # 256

        # MLP 层
        self.fc1 = layer_init(nn.Linear(encoder_output_dim, config.MLP_HIDDEN_DIM))
        self.ln1 = nn.LayerNorm(config.MLP_HIDDEN_DIM)
        self.fc2 = layer_init(nn.Linear(config.MLP_HIDDEN_DIM, config.MLP_HIDDEN_DIM))
        self.ln2 = nn.LayerNorm(config.MLP_HIDDEN_DIM)
        self.out = layer_init(nn.Linear(config.MLP_HIDDEN_DIM, action_dim), std=0.01)

        self.activation = nn.LeakyReLU(0.01)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # 注意力编码
        encoded = self.encoder(obs)
        # MLP 处理
        x = self.activation(self.ln1(self.fc1(encoded)))
        x = self.activation(self.ln2(self.fc2(x)))
        return torch.tanh(self.out(x))


class CriticNetworkWithAttention(nn.Module):
    """使用注意力机制的 Critic 网络"""

    def __init__(self, num_agents: int, obs_dim: int, action_dim: int, shared_encoders: nn.ModuleList) -> None:
        super().__init__()
        self.num_agents = num_agents
        self.obs_dim = obs_dim

        # 使用外部传入的共享编码器（所有Critic共享，避免参数爆炸）
        self.encoders = shared_encoders
        encoder_output_dim = self.encoders[0].output_dim  # 256

        # 总输入维度：所有 agent 的编码 + 所有 agent 的动作
        total_encoded_dim = num_agents * encoder_output_dim
        total_action_dim = num_agents * action_dim

        # MLP 层（带残差连接）
        self.fc1 = layer_init(nn.Linear(total_encoded_dim + total_action_dim, config.MLP_HIDDEN_DIM))
        self.ln1 = nn.LayerNorm(config.MLP_HIDDEN_DIM)
        self.fc2 = layer_init(nn.Linear(config.MLP_HIDDEN_DIM, config.MLP_HIDDEN_DIM))
        self.ln2 = nn.LayerNorm(config.MLP_HIDDEN_DIM)
        self.fc3 = layer_init(nn.Linear(config.MLP_HIDDEN_DIM, config.MLP_HIDDEN_DIM))
        self.ln3 = nn.LayerNorm(config.MLP_HIDDEN_DIM)
        self.out = layer_init(nn.Linear(config.MLP_HIDDEN_DIM, 1))

        self.activation = nn.LeakyReLU(0.01)

    def encode_observations(self, joint_obs: torch.Tensor) -> torch.Tensor:
        """Pre-compute encodings for all agents (optimization for MADDPG update loop)."""
        batch_size = joint_obs.shape[0]
        encodings = []
        for i in range(self.num_agents):
            agent_obs = joint_obs[:, i * self.obs_dim:(i + 1) * self.obs_dim]
            encoded = self.encoders[i](agent_obs)
            encodings.append(encoded)
        return torch.cat(encodings, dim=-1)

    def forward(self, joint_obs: torch.Tensor, joint_action: torch.Tensor, 
                joint_encoded: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            joint_obs: [batch, num_agents * obs_dim] 所有 agent 的观测
            joint_action: [batch, num_agents * action_dim] 所有 agent 的动作
            joint_encoded: [batch, num_agents * encoder_dim] 预计算的编码（可选）
        """
        # Use pre-computed encodings if provided, otherwise compute them
        if joint_encoded is None:
            joint_encoded = self.encode_observations(joint_obs)

        # 拼接编码和动作
        x = torch.cat([joint_encoded, joint_action], dim=-1)

        # MLP 处理（带残差连接）
        x = self.activation(self.ln1(self.fc1(x)))
        residual = x
        x = self.activation(self.ln2(self.fc2(x)))
        x = x + residual
        residual = x
        x = self.activation(self.ln3(self.fc3(x)))
        x = x + residual

        return self.out(x)
    
    def mlp_parameters(self):
        """返回仅属于 MLP 的参数（不包括共享编码器）"""
        return list(self.fc1.parameters()) + list(self.ln1.parameters()) + \
               list(self.fc2.parameters()) + list(self.ln2.parameters()) + \
               list(self.fc3.parameters()) + list(self.ln3.parameters()) + \
               list(self.out.parameters())
