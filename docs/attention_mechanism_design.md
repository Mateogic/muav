# 注意力机制设计方案

## 1. 问题背景

### 原始问题：UE 观测截断
- UAV 只能观测固定数量的 UE（如 20 个），但实际可能服务更多
- 截断导致观测与实际状态不一致，影响决策质量
- 统计数据显示单个 UAV 最多可覆盖 26 个 UE

### 解决方案：注意力机制
- 使用 Cross-Attention 处理可变长度的 UE 列表
- 通过 padding + mask 支持动态数量的 UE
- 注意力机制自动学习关注重要的 UE

## 2. 架构设计：Cross-Attention

### 设计理念
- **Q (Query)**: 来自 UAV 自身状态，表示"UAV 想要关注什么"
- **K (Key)**: 来自 UE 特征，表示"每个 UE 的关键信息"
- **V (Value)**: 来自 UE 特征，表示"每个 UE 的实际内容"

### 语义解释
UAV 作为决策主体，主动查询（Query）UE 列表，根据每个 UE 的关键特征（Key）计算注意力权重，然后加权聚合 UE 的价值信息（Value）。

## 3. Embedding 设计

### 为什么需要 Embedding

1. **维度对齐**
   - UE 原始特征只有 5 维，维度太低
   - 注意力机制需要 64-128 维来捕捉复杂关系
   - 点积注意力要求 Q, K 维度一致

2. **特征类型差异**
   - **位置特征**（连续值）：需要线性变换
   - **文件 ID**（离散值）：需要 Embedding 层学习文件相似性
   - **缓存命中**（二值）：需要独立处理
   - **缓存位图**（多维二值）：需要学习缓存模式

3. **学习能力**
   - Embedding 可以学习特征之间的非线性关系
   - 例如：学习热门文件与冷门文件的区别

### Embedding 模块

#### UEEmbedding (5 → 128)
```
输入: [batch, num_ues, 5]
  - pos (3): 相对位置（归一化）
  - file_id (1): 请求文件 ID（归一化到 [0,1)，需要反归一化）
  - cache_hit (1): 缓存命中标志

处理:
  - pos → Linear(3, 64) → LeakyReLU
  - file_id → round() → Embedding(NUM_FILES, 32)  # 使用 round() 避免浮点精度问题
  - cache_hit → Linear(1, 32) → LeakyReLU
  - concat → LayerNorm

输出: [batch, num_ues, 128]
```

#### NeighborEmbedding (25 → 64) - 混合方案
```
输入: [batch, num_neighbors, 25]
  - pos (3): 相对位置
  - cache (20): 原始缓存位图（让注意力机制学习）
  - immediate_help (1): 即时帮助能力（预处理特征）
  - complementarity (1): 缓存互补性（预处理特征）

处理:
  - pos → Linear(3, 16) → LeakyReLU
  - cache → Linear(20, 32) → LeakyReLU  # 最多维度，信息最丰富
  - processed → Linear(2, 16) → LeakyReLU
  - concat → LayerNorm

输出: [batch, num_neighbors, 64]

设计理念：
  - 原始 cache bitmap 让注意力机制自由学习协作模式
  - 预处理特征 (immediate_help, complementarity) 编码领域知识，加速收敛
```

#### UAVEmbedding (23 → 64)
```
输入:
  - pos: [batch, 3] 归一化位置
  - cache: [batch, NUM_FILES] 缓存位图

处理:
  - pos → Linear(3, 32) → LeakyReLU
  - cache → Linear(20, 32) → LeakyReLU
  - concat → LayerNorm

输出: [batch, 64]
```

## 4. Cross-Attention 实现

### 多头注意力
```
Q (Query) - 来自 UAV 状态:
  输入: UAV embedding [batch, 64]
  处理: Linear(64, kv_dim)
  输出: [batch, 1, kv_dim]

K (Key) - 来自 UE/Neighbor 特征:
  输入: embeddings [batch, seq_len, kv_dim]
  处理: Linear projection
  输出: [batch, seq_len, kv_dim]

V (Value) - 来自 UE/Neighbor 特征:
  输入: embeddings [batch, seq_len, kv_dim]
  处理: Linear projection
  输出: [batch, seq_len, kv_dim]

注意力计算:
  Q, K, V = reshape to multi-head format
  scores = Q @ K^T / sqrt(head_dim)
  scores = masked_fill(scores, mask==0, -inf)
  weights = softmax(scores)
  weights = nan_to_num(weights)  # 处理全 mask 情况
  output = weights @ V
  output = reshape and project
```

### Mask 机制
- 观测中包含 `neighbor_count` 和 `ue_count` 字段（仅 USE_ATTENTION=True 时）
- 动态生成 mask：`mask[i] = 1 if i < count else 0`
- Padding 位置的注意力分数设为 -inf，softmax 后为 0

## 5. 完整编码器架构

### AttentionEncoder
```
输入: obs [batch, 375]

1. 解析观测 → 结构化数据
   - uav_pos: [batch, 3]
   - uav_cache: [batch, 20]
   - neighbor_features: [batch, 4, 25]  # 混合特征
   - neighbor_count: [batch]
   - ue_features: [batch, 50, 5]
   - ue_count: [batch]

2. UAV Embedding → [batch, 64]

3. UE Embedding + CrossAttention
   - UE features → UEEmbedding → [batch, 50, 128]
   - CrossAttention(UAV_emb, UE_emb, mask) → [batch, 128]

4. Neighbor Embedding + CrossAttention
   - Neighbor features → NeighborEmbedding → [batch, 4, 64]
   - CrossAttention(UAV_emb, Neighbor_emb, mask) → [batch, 64]

5. 拼接 → [batch, 256]

输出: encoded [batch, 256]
```

## 6. 参数配置

### 当前配置（最佳实践）
| 参数 | 值 | 说明 |
|------|-----|------|
| MAX_ASSOCIATED_UES | 50 | 最多观测 UE 数（覆盖 >99.9% 情况）|
| MAX_UAV_NEIGHBORS | 4 | 最多观测邻居数 |
| NEIGHBOR_STATE_DIM | 25 | 邻居特征维度（混合方案）|
| ATTENTION_EMBED_DIM | 128 | UE 注意力 embedding 维度 |
| ATTENTION_UAV_EMBED_DIM | 64 | UAV embedding 维度 |
| ATTENTION_NEIGHBOR_DIM | 64 | Neighbor 注意力维度 |
| ATTENTION_NUM_HEADS | 2 | 多头注意力头数 |
| ATTENTION_DROPOUT | 0.1 | Dropout 率 |

### 维度分析
| 组件 | heads | head_dim | 评价 |
|------|-------|----------|------|
| UE attention | 2 | 64 | 优秀（最佳实践 32-64）|
| Neighbor attention | 2 | 32 | 良好（最佳实践 32-64）|

### 输入/输出比
- 输入维度: 375
- 输出维度: 256
- 压缩比: 1.46（有效压缩，提取关键信息）

## 7. 观测结构

### 新观测格式 (375 维)
```
[uav_pos(3), uav_cache(20),
 neighbor_features(4×25=100), neighbor_count(1),
 ue_features(50×5=250), ue_count(1)]
```

### 邻居特征结构 (25 维/邻居)
```
[pos(3), cache_bitmap(20), immediate_help(1), complementarity(1)]
```

### 与旧格式的区别
1. 增加了 `neighbor_count` 和 `ue_count` 字段用于生成 mask（仅注意力模式）
2. `MAX_ASSOCIATED_UES` 从 20 增加到 50，减少截断
3. `MAX_UAV_NEIGHBORS` 从 3 增加到 4
4. **邻居特征采用混合方案**：原始 cache bitmap + 预处理特征

## 8. 网络集成

### ActorNetworkWithAttention
```
obs [batch, 375]
  → AttentionEncoder → [batch, 256]
  → Linear(256, 768) → LayerNorm → LeakyReLU
  → Linear(768, 768) → LayerNorm → LeakyReLU
  → Linear(768, 5) → Tanh
  → action [batch, 5]
```

### CriticNetworkWithAttention
```
joint_obs [batch, 10×375]
  → 每个 agent 独立 AttentionEncoder → [batch, 10×256]
  → concat with joint_action [batch, 10×5]
  → Linear(2560+50, 768) → LayerNorm → LeakyReLU
  → Residual blocks (×2)
  → Linear(768, 1)
  → Q-value [batch, 1]
```

## 9. 关键实现细节

### Bug 修复：File ID 反归一化 + round()
```python
# 错误实现（所有 file_id 都被映射到 0）
file_id = ue_features[:, :, 3].long()  # 0.5.long() = 0 ← 错误!

# 改进实现（使用 round() 避免浮点精度问题）
file_id = (ue_features[:, :, 3] * config.NUM_FILES).round().long().clamp(0, config.NUM_FILES - 1)
# 0.4999999 * 20 = 9.999998 → round() = 10 ← 正确!
```

### 全 Mask 处理
```python
# 当所有位置都是 padding 时，softmax 会产生 nan
attn_weights = F.softmax(attn_scores, dim=-1)
attn_weights = torch.nan_to_num(attn_weights, nan=0.0)  # 将 nan 替换为 0
```

### 条件观测格式
```python
# env.py: 根据 USE_ATTENTION 决定是否包含 count 字段
if config.USE_ATTENTION:
    obs = np.concatenate([..., neighbor_count, ..., ue_count])
else:
    obs = np.concatenate([...])  # 无 count 字段
```

## 10. 使用方式

### 配置开关
```python
# config.py
USE_ATTENTION: bool = True  # 启用注意力机制
```

### 条件导入
```python
# agents.py
if config.USE_ATTENTION:
    from marl_models.attention import AttentionEncoder

# maddpg.py
if config.USE_ATTENTION:
    self.actors = [ActorNetworkWithAttention(...) for _ in range(num_agents)]
else:
    self.actors = [ActorNetwork(...) for _ in range(num_agents)]
```

## 11. 性能考虑

### 计算复杂度
- UE attention: O(50 × 128) per UAV
- Neighbor attention: O(4 × 64) per UAV
- 总体增加约 25% 计算量，但信息完整性大幅提升

### 内存占用
- 每个 Critic 有 10 个 AttentionEncoder（每个 agent 一个）
- 总参数量增加约 2.5M

## 12. 设计决策：混合邻居特征

### 为什么采用混合方案？

| 方案 | 维度 | 信息量 | 学习难度 |
|------|------|--------|----------|
| 仅预处理特征 | 5 | 部分丢失 | 低 |
| 仅原始 cache | 23 | 完整 | 高 |
| **混合方案** | 25 | 完整+先验 | 中 |

### 混合方案的优势
1. **原始 cache bitmap**：让注意力机制自由学习协作模式
2. **预处理特征**：编码领域知识（即时帮助、互补性），加速收敛
3. **类似 ResNet shortcut**：预处理特征提供"捷径"，原始数据提供细节

## 13. 未来改进方向

1. **位置编码**: 为 UE 添加相对位置编码，增强空间感知
2. **多层注意力**: 堆叠多层 attention 捕捉更复杂的关系
3. **自注意力**: UE 之间的自注意力，捕捉 UE 间的相互关系
4. **共享编码器**: Critic 中共享 encoder 减少参数量
5. **动态邻居选择**: 根据任务需求动态调整关注的邻居数量
