from marl_models.base_model import MARLModel
from marl_models.buffer_and_helpers import RolloutBuffer, ReplayBuffer
from marl_models.utils import save_models
from environment.env import Env
from utils.logger import Logger, Log
from utils.plot_snapshots import plot_snapshot
from utils.plot_logs import generate_plots

# from utils.plot_snapshots import update_trajectories, reset_trajectories  # trajectory tracking, comment if not needed
import config
import torch
import numpy as np
import time


def train_on_policy(env: Env, model: MARLModel, logger: Logger, num_episodes: int) -> None:
    start_time: float = time.time()
    buffer: RolloutBuffer = RolloutBuffer(num_agents=config.NUM_UAVS, obs_dim=config.OBS_DIM_SINGLE, action_dim=config.ACTION_DIM, state_dim=config.STATE_DIM, buffer_size=config.PPO_ROLLOUT_LENGTH, device=model.device)
    max_time_steps: int = num_episodes * config.STEPS_PER_EPISODE
    num_updates: int = max_time_steps // config.PPO_ROLLOUT_LENGTH
    assert num_updates > 0, "num_updates is 0, please modify settings."
    save_freq: int = num_episodes // 10
    if num_episodes < 1000:
        save_freq = 100
    print(f"Total updates to be performed: {num_updates}")
    print(f"Each update has {config.PPO_ROLLOUT_LENGTH} steps.")
    print(f"Updates for {config.PPO_EPOCHS} epochs with batch size {config.PPO_BATCH_SIZE}.")
    rollout_log: Log = Log()

    for update in range(1, num_updates + 1):
        obs: list[np.ndarray] = env.reset()
        state: np.ndarray = np.concatenate(obs, axis=0)
        rollout_reward: float = 0.0
        rollout_latency: float = 0.0
        rollout_energy: float = 0.0
        rollout_fairness: float = 0.0
        rollout_rate: float = 0.0
        rollout_collisions: int = 0
        rollout_boundaries: int = 0
        # reset_trajectories(env)  # tracking code, comment if not needed
        plot_snapshot(env, update, 0, logger.log_dir, "update", logger.timestamp, True)

        for step in range(1, config.PPO_ROLLOUT_LENGTH + 1):
            if step % config.IMG_FREQ == 0:
                plot_snapshot(env, update, step, logger.log_dir, "update", logger.timestamp)

            obs_arr: np.ndarray = np.array(obs)
            raw_actions, log_probs, values = model.get_action_and_value(obs_arr, state)
            actions: np.ndarray = np.clip(raw_actions, -1.0, 1.0)

            next_obs, rewards, (total_latency, total_energy, jfi, total_rate) = env.step(actions)
            # update_trajectories(env)  # tracking code, comment if not needed
            next_state: np.ndarray = np.concatenate(next_obs, axis=0)
            # For time-limit truncation, we should not treat the episode as "done" for the value update.
            # We want to bootstrap from the next state's value.
            # Only use done=True if the episode terminated due to failure/completion, not timeout.
            done: bool = False 
            # Store raw actions so PPO log_prob computation remains consistent
            buffer.add(state, obs_arr, raw_actions, log_probs, rewards, done, values)

            obs = next_obs
            state = next_state

            rollout_reward += np.sum(rewards)
            rollout_latency += total_latency
            rollout_energy += total_energy
            rollout_fairness = jfi
            rollout_rate += total_rate
            
            for uav in env.uavs:
                if uav.collision_violation:
                    rollout_collisions += 1
                if uav.boundary_violation:
                    rollout_boundaries += 1

        with torch.no_grad():
            _, _, last_values = model.get_action_and_value(np.array(obs), state)

        buffer.compute_returns_and_advantages(last_values, config.DISCOUNT_FACTOR, config.PPO_GAE_LAMBDA)

        for _ in range(config.PPO_EPOCHS):
            for batch in buffer.get_batches(config.PPO_BATCH_SIZE):
                model.update(batch)

        buffer.clear()

        rollout_log.append(rollout_reward, rollout_latency, rollout_energy, rollout_fairness, rollout_rate, rollout_collisions, rollout_boundaries)
        if update % config.LOG_FREQ == 0:
            elapsed_time: float = time.time() - start_time
            logger.log_metrics(update, rollout_log, config.LOG_FREQ, elapsed_time, "update")
        if update % 100 == 0:
            generate_plots(f"{logger.log_dir}/log_data_{logger.timestamp}.json", f"train_plots/{config.MODEL}/", "train", logger.timestamp)
        if update % save_freq == 0 and update < num_episodes:
            save_models(model, update, "update", logger.timestamp)

    save_models(model, -1, "update", logger.timestamp, final=True)


def train_off_policy(env: Env, model: MARLModel, logger: Logger, num_episodes: int, total_step_count: int) -> None:
    start_time: float = time.time()
    buffer: ReplayBuffer = ReplayBuffer(config.REPLAY_BUFFER_SIZE)
    save_freq: int = num_episodes // 10
    if num_episodes < 1000:
        save_freq = 100
    episode_log: Log = Log()

    for episode in range(1, num_episodes + 1):
        obs = env.reset()
        model.reset()
        episode_reward: float = 0.0
        episode_latency: float = 0.0
        episode_energy: float = 0.0
        episode_fairness: float = 0.0
        episode_rate: float = 0.0
        episode_collisions: int = 0
        episode_boundaries: int = 0
        # reset_trajectories(env)  # tracking code, comment if not needed
        plot_snapshot(env, episode, 0, logger.log_dir, "episode", logger.timestamp, True)

        for step in range(1, config.STEPS_PER_EPISODE + 1):
            if step % config.IMG_FREQ == 0:
                plot_snapshot(env, episode, step, logger.log_dir, "episode", logger.timestamp)

            total_step_count += 1
            if total_step_count <= config.INITIAL_RANDOM_STEPS:
                actions: np.ndarray = np.array([np.random.uniform(-1, 1, config.ACTION_DIM) for _ in range(config.NUM_UAVS)])
            else:
                actions = model.select_actions(obs, exploration=True)

            next_obs, rewards, (total_latency, total_energy, jfi, total_rate) = env.step(actions)
            # update_trajectories(env)  # tracking code, comment if not needed
            
            # For time-limit truncation, we should not treat the episode as "done" for the value update.
            done: bool = False
            buffer.add(obs, actions, rewards, next_obs, done)

            if total_step_count > config.INITIAL_RANDOM_STEPS and step % config.LEARN_FREQ == 0 and len(buffer) > config.REPLAY_BATCH_SIZE:
                batch = buffer.sample(config.REPLAY_BATCH_SIZE)
                model.update(batch)

            obs = next_obs

            episode_reward += np.sum(rewards)
            episode_latency += total_latency
            episode_energy += total_energy
            episode_fairness = jfi
            episode_rate += total_rate
            
            for uav in env.uavs:
                if uav.collision_violation:
                    episode_collisions += 1
                if uav.boundary_violation:
                    episode_boundaries += 1
            
            if done:
                break

        # Decay exploration noise at the end of each episode (for MATD3/MADDPG)
        if hasattr(model, 'noise'):
            for n in model.noise:
                n.decay()

        episode_log.append(episode_reward, episode_latency, episode_energy, episode_fairness, episode_rate, episode_collisions, episode_boundaries)
        if episode % config.LOG_FREQ == 0:
            elapsed_time: float = time.time() - start_time
            logger.log_metrics(episode, episode_log, config.LOG_FREQ, elapsed_time, "episode")
        if episode % 100 == 0:
            generate_plots(f"{logger.log_dir}/log_data_{logger.timestamp}.json", f"train_plots/{config.MODEL}/", "train", logger.timestamp)
        if episode % save_freq == 0 and episode < num_episodes:
            save_models(model, episode, "episode", logger.timestamp, total_steps=total_step_count)

    save_models(model, -1, "episode", logger.timestamp, final=True, total_steps=total_step_count)


def train_random(env: Env, model: MARLModel, logger: Logger, num_episodes: int) -> None:
    start_time: float = time.time()
    episode_log: Log = Log()

    for episode in range(1, num_episodes + 1):
        obs = env.reset()
        episode_reward: float = 0.0
        episode_latency: float = 0.0
        episode_energy: float = 0.0
        episode_fairness: float = 0.0
        episode_rate: float = 0.0
        episode_collisions: int = 0
        episode_boundaries: int = 0
        # reset_trajectories(env)  # tracking code, comment if not needed
        plot_snapshot(env, episode, 0, logger.log_dir, "episode", logger.timestamp, True)

        for step in range(1, config.STEPS_PER_EPISODE + 1):
            if step % config.IMG_FREQ == 0:
                plot_snapshot(env, episode, step, logger.log_dir, "episode", logger.timestamp)

            actions: np.ndarray = model.select_actions(obs, exploration=False)
            next_obs, rewards, (total_latency, total_energy, jfi, total_rate) = env.step(actions)
            # update_trajectories(env)  # tracking code, comment if not needed
            done: bool = step >= config.STEPS_PER_EPISODE
            obs = next_obs

            episode_reward += np.sum(rewards)
            episode_latency += total_latency
            episode_energy += total_energy
            episode_fairness = jfi
            episode_rate += total_rate
            
            for uav in env.uavs:
                if uav.collision_violation:
                    episode_collisions += 1
                if uav.boundary_violation:
                    episode_boundaries += 1
            
            if done:
                break

        episode_log.append(episode_reward, episode_latency, episode_energy, episode_fairness, episode_rate, episode_collisions, episode_boundaries)
        if episode % config.LOG_FREQ == 0:
            elapsed_time: float = time.time() - start_time
            logger.log_metrics(episode, episode_log, config.LOG_FREQ, elapsed_time, "episode")
        if episode % 100 == 0:
            generate_plots(f"{logger.log_dir}/log_data_{logger.timestamp}.json", f"train_plots/{config.MODEL}/", "train", logger.timestamp)
