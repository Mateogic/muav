
import sys
import numpy as np
import config
from environment.env import Env
import matplotlib.pyplot as plt

def analyze_stability():
    print("=" * 80)
    print("TRAINING STABILITY ANALYSIS")
    print("=" * 80)
    
    # Initialize environment
    env = Env()
    
    # Parameters for analysis
    num_episodes = 5
    steps_per_episode = 100
    
    # Data collectors
    data = {
        "latency": [],
        "energy": [],
        "jfi": [],
        "rate": [],
        "reward_latency": [],
        "reward_energy": [],
        "reward_fairness": [],
        "reward_rate": [],
        "total_reward": [],
        "coverage_ratio": [],
        "collisions": 0,
        "boundaries": 0
    }
    
    print(f"Running {num_episodes} episodes with {steps_per_episode} steps each (Random Policy)...")
    
    for episode in range(num_episodes):
        obs = env.reset()
        ep_collisions = 0
        ep_boundaries = 0
        
        for step in range(steps_per_episode):
            # Random actions
            actions = np.random.uniform(-1, 1, (config.NUM_UAVS, config.ACTION_DIM))
            
            next_obs, rewards, (latency, energy, jfi, rate) = env.step(actions)
            
            # Calculate reward components manually to verify weights
            # Note: These formulas must match env.py exactly
            r_fairness = config.ALPHA_3 * np.log(jfi + config.EPSILON)
            r_latency = config.ALPHA_1 * np.log(latency + config.EPSILON)
            r_energy = config.ALPHA_2 * np.log(energy + config.EPSILON)
            
            reference_rate = config.BANDWIDTH_EDGE * np.log2(1 + 100)
            normalized_rate = rate / (config.NUM_UES * reference_rate + config.EPSILON)
            r_rate = config.ALPHA_RATE * np.log(normalized_rate + config.EPSILON)
            
            # Collect raw metrics
            data["latency"].append(latency)
            data["energy"].append(energy)
            data["jfi"].append(jfi)
            data["rate"].append(rate)
            
            # Collect weighted reward components
            data["reward_latency"].append(r_latency)
            data["reward_energy"].append(r_energy)
            data["reward_fairness"].append(r_fairness)
            data["reward_rate"].append(r_rate)
            data["total_reward"].append(np.mean(rewards)) # Average reward across agents
            
            # Collect coverage
            assigned_ues = sum(1 for ue in env.ues if ue.assigned)
            data["coverage_ratio"].append(assigned_ues / config.NUM_UES)
            
            # Count violations
            for uav in env.uavs:
                if uav.collision_violation:
                    data["collisions"] += 1
                    ep_collisions += 1
                if uav.boundary_violation:
                    data["boundaries"] += 1
                    ep_boundaries += 1
            
            obs = next_obs
            
        print(f"Episode {episode+1}: Collisions={ep_collisions}, Boundaries={ep_boundaries}")

    # --- Analysis ---
    print("\n" + "-" * 80)
    print("STATISTICAL ANALYSIS")
    print("-" * 80)
    
    def print_stats(name, values, unit=""):
        mean = np.mean(values)
        std = np.std(values)
        min_v = np.min(values)
        max_v = np.max(values)
        print(f"{name:20s} | Mean: {mean:10.4f} {unit} | Std: {std:10.4f} | Range: [{min_v:.4f}, {max_v:.4f}]")
        return mean, std

    # 1. Physical Metrics
    print("\n[Physical Metrics]")
    print_stats("Latency", data["latency"], "s")
    print_stats("Energy", data["energy"], "J")
    print_stats("JFI (Fairness)", data["jfi"], "")
    print_stats("Total Rate", np.array(data["rate"]) / 1e6, "Mbps") # Convert to Mbps
    print_stats("Coverage Ratio", data["coverage_ratio"], "%")

    # 2. Reward Components (Weighted Log Values)
    print("\n[Reward Components (Weighted Log Values)]")
    # Note: Reward = Fairness + Rate - Latency - Energy
    # So we display them as they contribute to the sum
    mean_r_lat, std_r_lat = print_stats("R_Latency (-)", data["reward_latency"])
    mean_r_eng, std_r_eng = print_stats("R_Energy (-)", data["reward_energy"])
    mean_r_fair, std_r_fair = print_stats("R_Fairness (+)", data["reward_fairness"])
    mean_r_rate, std_r_rate = print_stats("R_Rate (+)", data["reward_rate"])
    
    # 3. Reward Contribution Analysis
    print("\n[Reward Contribution Analysis]")
    # Calculate absolute magnitude of each component's mean
    total_magnitude = abs(mean_r_lat) + abs(mean_r_eng) + abs(mean_r_fair) + abs(mean_r_rate)
    print(f"Latency Contribution:  {abs(mean_r_lat)/total_magnitude*100:.2f}%")
    print(f"Energy Contribution:   {abs(mean_r_eng)/total_magnitude*100:.2f}%")
    print(f"Fairness Contribution: {abs(mean_r_fair)/total_magnitude*100:.2f}%")
    print(f"Rate Contribution:     {abs(mean_r_rate)/total_magnitude*100:.2f}%")
    
    # 4. Stability Warnings
    print("\n[Stability Warnings & Suggestions]")
    warnings = []
    
    # Check Coverage
    avg_cov = np.mean(data["coverage_ratio"])
    if avg_cov < 0.3:
        warnings.append(f"CRITICAL: Low Coverage ({avg_cov*100:.1f}%). Random policy should achieve at least 30-40%. Check UAV_COVERAGE_RADIUS or NUM_UAVS.")
    
    # Check Collisions
    total_steps = num_episodes * steps_per_episode * config.NUM_UAVS
    collision_rate = data["collisions"] / total_steps
    if collision_rate > 0.1:
        warnings.append(f"HIGH COLLISION RATE: {collision_rate*100:.1f}% of UAV steps involved collisions. Consider increasing MIN_UAV_SEPARATION or checking movement logic.")

    # Check Reward Dominance
    # Ideally, no single component should contribute > 60-70% of the gradient magnitude (roughly)
    # Standard deviation is a better proxy for gradient magnitude in RL
    stds = {"Latency": std_r_lat, "Energy": std_r_eng, "Fairness": std_r_fair, "Rate": std_r_rate}
    max_std_name = max(stds, key=stds.get)
    max_std_val = stds[max_std_name]
    total_std = sum(stds.values())
    
    print(f"Standard Deviation Distribution (Gradient Proxy):")
    for k, v in stds.items():
        print(f"  {k}: {v:.4f} ({v/total_std*100:.1f}%)")
        
    if stds[max_std_name] / total_std > 0.6:
        warnings.append(f"IMBALANCED REWARDS: {max_std_name} dominates variance ({stds[max_std_name]/total_std*100:.1f}%). The agent may ignore other objectives. Consider reducing ALPHA for {max_std_name}.")

    if not warnings:
        print("No critical stability issues detected based on random sampling.")
    else:
        for w in warnings:
            print(f"! {w}")

if __name__ == "__main__":
    analyze_stability()
