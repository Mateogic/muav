"""
诊断脚本 v2：深入分析奖励各分量的实际数量级
运行方式：python analyze_reward_scale_v2.py --num_episodes 5
"""

import numpy as np
import torch
import argparse
from environment.env import Env
from marl_models.utils import get_model
import config


class InstrumentedEnv(Env):
    """添加诊断功能的环境类，用于收集奖励分量的详细信息"""
    
    def __init__(self):
        super().__init__()
        # 收集器
        self.r_latency_history = []
        self.r_energy_history = []
        self.r_rate_history = []
        self.r_fairness_history = []
        self.reward_no_penalty_history = []
        self.reward_with_penalty_history = []
        self.collision_penalty_history = []
        self.boundary_penalty_history = []
    
    def _get_rewards_and_metrics(self):
        """重写以收集详细的奖励分量信息"""
        total_latency = sum(ue.latency_current_request if ue.assigned else config.NON_SERVED_LATENCY_PENALTY 
                          for ue in self._ues)
        total_energy = sum(uav.energy for uav in self._uavs)
        total_rate = sum(uav.total_downlink_rate for uav in self._uavs)
        
        sc_metrics = np.array([ue.service_coverage for ue in self._ues])
        jfi = 0.0
        if sc_metrics.size > 0 and np.sum(sc_metrics**2) > 0:
            jfi = (np.sum(sc_metrics) ** 2) / (sc_metrics.size * np.sum(sc_metrics**2))

        # 动态归一化
        log_latency = np.log(total_latency + config.EPSILON)
        log_energy = np.log(total_energy + config.EPSILON)
        log_rate = np.log(total_rate + config.EPSILON)
        
        r_latency = config.ALPHA_1 * self._latency_normalizer.normalize(log_latency)
        r_energy = config.ALPHA_2 * self._energy_normalizer.normalize(log_energy)
        r_fairness = config.ALPHA_3 * np.clip((jfi - 0.6) * 5.0, -2.0, 2.0)
        r_rate = config.ALPHA_RATE * self._rate_normalizer.normalize(log_rate)
        
        # 收集分量（缩放前）
        self.r_latency_history.append(r_latency)
        self.r_energy_history.append(r_energy)
        self.r_rate_history.append(r_rate)
        self.r_fairness_history.append(r_fairness)
        
        # 基础奖励（无惩罚）
        reward_no_penalty = r_fairness + r_rate - r_latency - r_energy
        self.reward_no_penalty_history.append(reward_no_penalty)
        
        # 计算惩罚
        rewards = [reward_no_penalty] * config.NUM_UAVS
        total_collision_penalty = 0.0
        total_boundary_penalty = 0.0
        
        for uav in self._uavs:
            if uav.collision_violation:
                rewards[uav.id] -= config.COLLISION_PENALTY
                total_collision_penalty += config.COLLISION_PENALTY
            if uav.boundary_violation:
                rewards[uav.id] -= config.BOUNDARY_PENALTY
                total_boundary_penalty += config.BOUNDARY_PENALTY
        
        self.collision_penalty_history.append(total_collision_penalty)
        self.boundary_penalty_history.append(total_boundary_penalty)
        
        # 带惩罚的总奖励（缩放前）
        reward_with_penalty = reward_no_penalty - total_collision_penalty / config.NUM_UAVS - total_boundary_penalty / config.NUM_UAVS
        self.reward_with_penalty_history.append(reward_with_penalty)
        
        # 最终缩放
        rewards = [r * config.REWARD_SCALING_FACTOR for r in rewards]
        return rewards, (total_latency, total_energy, jfi, total_rate)


def analyze_detailed(num_episodes: int = 5):
    """运行诊断分析"""
    
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    
    env = InstrumentedEnv()
    
    raw_latencies = []
    raw_energies = []
    raw_rates = []
    raw_jfis = []
    collision_counts = []
    boundary_counts = []
    all_rewards = []
    
    print(f"\n{'='*70}")
    print(f"深度分析 {num_episodes} 个 Episode 的奖励分量（使用新参数）")
    print(f"{'='*70}")
    print(f"\n当前参数:")
    print(f"  REWARD_SCALING_FACTOR = {config.REWARD_SCALING_FACTOR}")
    print(f"  COLLISION_PENALTY = {config.COLLISION_PENALTY}")
    print(f"  BOUNDARY_PENALTY = {config.BOUNDARY_PENALTY}")
    print(f"  ALPHA_1 (latency) = {config.ALPHA_1}")
    print(f"  ALPHA_2 (energy) = {config.ALPHA_2}")
    print(f"  ALPHA_3 (fairness) = {config.ALPHA_3}")
    print(f"  ALPHA_RATE = {config.ALPHA_RATE}")
    print()
    
    for episode in range(1, num_episodes + 1):
        obs = env.reset()
        ep_rewards = []
        ep_collisions = 0
        ep_boundaries = 0
        
        for step in range(1, config.STEPS_PER_EPISODE + 1):
            actions = np.array([np.random.uniform(-1, 1, config.ACTION_DIM) 
                               for _ in range(config.NUM_UAVS)])
            
            next_obs, rewards, (total_latency, total_energy, jfi, total_rate, 
                               step_collisions, step_boundaries) = env.step(actions)
            
            raw_latencies.append(total_latency)
            raw_energies.append(total_energy)
            raw_rates.append(total_rate)
            raw_jfis.append(jfi)
            
            ep_rewards.extend(rewards)
            ep_collisions += step_collisions
            ep_boundaries += step_boundaries
            all_rewards.extend(rewards)
            
            obs = next_obs
        
        collision_counts.append(ep_collisions)
        boundary_counts.append(ep_boundaries)
        
        avg_reward = np.mean(ep_rewards)
        print(f"Episode {episode}: Avg Reward = {avg_reward:.4f}, "
              f"Col = {ep_collisions}, Bnd = {ep_boundaries}")
    
    # ========== 详细统计 ==========
    print(f"\n{'='*70}")
    print("1. 原始指标统计")
    print(f"{'='*70}")
    print(f"Latency:  mean={np.mean(raw_latencies):.2f}, std={np.std(raw_latencies):.2f}")
    print(f"Energy:   mean={np.mean(raw_energies):.2f}, std={np.std(raw_energies):.2f}")
    print(f"Rate:     mean={np.mean(raw_rates):.2e}, std={np.std(raw_rates):.2e}")
    print(f"JFI:      mean={np.mean(raw_jfis):.4f}, std={np.std(raw_jfis):.4f}")
    
    print(f"\n{'='*70}")
    print("2. 奖励各分量统计（缩放前）")
    print(f"{'='*70}")
    
    # 跳过前几步的异常值（归一化器冷启动）
    skip_steps = 100
    
    r_lat = np.array(env.r_latency_history[skip_steps:])
    r_eng = np.array(env.r_energy_history[skip_steps:])
    r_rate = np.array(env.r_rate_history[skip_steps:])
    r_fair = np.array(env.r_fairness_history[skip_steps:])
    r_no_pen = np.array(env.reward_no_penalty_history[skip_steps:])
    r_with_pen = np.array(env.reward_with_penalty_history[skip_steps:])
    col_pen = np.array(env.collision_penalty_history[skip_steps:])
    bnd_pen = np.array(env.boundary_penalty_history[skip_steps:])
    
    print(f"\n  r_latency (负向):   mean={np.mean(r_lat):>8.4f}, std={np.std(r_lat):.4f}, "
          f"[{np.percentile(r_lat, 5):.3f}, {np.percentile(r_lat, 95):.3f}]")
    print(f"  r_energy (负向):    mean={np.mean(r_eng):>8.4f}, std={np.std(r_eng):.4f}, "
          f"[{np.percentile(r_eng, 5):.3f}, {np.percentile(r_eng, 95):.3f}]")
    print(f"  r_rate (正向):      mean={np.mean(r_rate):>8.4f}, std={np.std(r_rate):.4f}, "
          f"[{np.percentile(r_rate, 5):.3f}, {np.percentile(r_rate, 95):.3f}]")
    print(f"  r_fairness (正向):  mean={np.mean(r_fair):>8.4f}, std={np.std(r_fair):.4f}, "
          f"[{np.percentile(r_fair, 5):.3f}, {np.percentile(r_fair, 95):.3f}]")
    
    print(f"\n  基础奖励 (无惩罚):  mean={np.mean(r_no_pen):>8.4f}, std={np.std(r_no_pen):.4f}")
    print(f"  碰撞惩罚/步:        mean={np.mean(col_pen):>8.4f}")
    print(f"  边界惩罚/步:        mean={np.mean(bnd_pen):>8.4f}")
    print(f"  总惩罚/步:          mean={np.mean(col_pen + bnd_pen):>8.4f}")
    
    print(f"\n{'='*70}")
    print("3. 缩放后的奖励统计")
    print(f"{'='*70}")
    
    base_scaled = np.mean(r_no_pen) * config.REWARD_SCALING_FACTOR
    penalty_scaled = np.mean(col_pen + bnd_pen) * config.REWARD_SCALING_FACTOR
    final_scaled = np.mean(all_rewards)
    
    print(f"\n  基础奖励 (缩放后):  {base_scaled:>8.4f}")
    print(f"  总惩罚 (缩放后):    {penalty_scaled:>8.4f}")
    print(f"  最终奖励 (缩放后):  {final_scaled:>8.4f}")
    print(f"\n  惩罚/基础奖励 比例: {abs(penalty_scaled / (base_scaled + 1e-8)):.2f}")
    
    print(f"\n{'='*70}")
    print("4. 归一化器状态")
    print(f"{'='*70}")
    print(f"\n  Latency normalizer: mean={env._latency_normalizer.mean:.4f}, "
          f"var={env._latency_normalizer.var:.4f}, count={env._latency_normalizer.count}")
    print(f"  Energy normalizer:  mean={env._energy_normalizer.mean:.4f}, "
          f"var={env._energy_normalizer.var:.4f}, count={env._energy_normalizer.count}")
    print(f"  Rate normalizer:    mean={env._rate_normalizer.mean:.4f}, "
          f"var={env._rate_normalizer.var:.4f}, count={env._rate_normalizer.count}")
    
    print(f"\n{'='*70}")
    print("5. 建议分析")
    print(f"{'='*70}")
    
    ratio = abs(penalty_scaled / (base_scaled + 1e-8))
    
    if ratio > 5:
        print(f"\n⚠️  惩罚/奖励比例 = {ratio:.1f}，惩罚严重主导！")
        suggested_penalty = abs(base_scaled) / (np.mean(col_pen + bnd_pen) / config.NUM_UAVS + 1e-8) * 0.5
        print(f"    建议将 COLLISION_PENALTY/BOUNDARY_PENALTY 降至 {suggested_penalty:.2f}")
        print(f"    或者将 REWARD_SCALING_FACTOR 提高至 {config.REWARD_SCALING_FACTOR * ratio / 2:.3f}")
    elif ratio > 2:
        print(f"\n⚠️  惩罚/奖励比例 = {ratio:.1f}，惩罚偏高")
        print(f"    可接受但建议微调，降低惩罚约 {int((ratio-1)*50)}%")
    elif ratio > 0.3:
        print(f"\n✅  惩罚/奖励比例 = {ratio:.1f}，比例合理")
    else:
        print(f"\n⚠️  惩罚/奖励比例 = {ratio:.1f}，惩罚可能过低")
        print(f"    智能体可能忽视约束，建议提高惩罚")
    
    # 建议 Q 值范围
    print(f"\n{'='*70}")
    print("6. 预期 Q 值范围估算")
    print(f"{'='*70}")
    
    avg_reward = final_scaled
    gamma = config.DISCOUNT_FACTOR
    # Q ≈ r / (1 - γ) for steady state (infinite horizon)
    # 但由于 episode 有限，实际 Q 值会更小
    horizon = config.STEPS_PER_EPISODE
    effective_horizon = min(horizon, 1 / (1 - gamma))  # ~100 for γ=0.99
    
    q_estimate_infinite = avg_reward / (1 - gamma)
    q_estimate_finite = avg_reward * (1 - gamma**horizon) / (1 - gamma)
    
    print(f"\n  平均每步奖励: {avg_reward:.4f}")
    print(f"  无限视野 Q 估计: {q_estimate_infinite:.2f}")
    print(f"  有限视野 Q 估计 (T={horizon}): {q_estimate_finite:.2f}")
    print(f"\n  如果 Q 值远大于此估计，说明 Critic 高估；远小于则低估")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes", type=int, default=5, help="分析的 episode 数量")
    args = parser.parse_args()
    
    analyze_detailed(args.num_episodes)
