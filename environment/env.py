from environment.user_equipments import UE
from environment.uavs import UAV
import config
import numpy as np
# 基于强化学习框架，模拟多无人机（UAV）空中基站通信保障环境，管理 UAV、用户设备（UE）和宏基站（MBS）的状态、动作和奖励。

class Env:
    def __init__(self) -> None:
        self._mbs_pos: np.ndarray = config.MBS_POS
        UE.initialize_ue_class()
        self._ues: list[UE] = [UE(i) for i in range(config.NUM_UES)]
        self._uavs: list[UAV] = [UAV(i) for i in range(config.NUM_UAVS)]
        self._time_step: int = 0

    @property
    def uavs(self) -> list[UAV]:
        return self._uavs

    @property
    def ues(self) -> list[UE]:
        return self._ues

    def reset(self) -> list[np.ndarray]:
        """Resets the environment to an initial state and returns the initial observations."""
        self._ues = [UE(i) for i in range(config.NUM_UES)]
        self._uavs = [UAV(i) for i in range(config.NUM_UAVS)]
        self._time_step = 0
        self._prepare_for_next_step()
        return self._get_obs()

    def step(self, actions: np.ndarray) -> tuple[list[np.ndarray], list[float], tuple[float, float, float]]:
        """Execute one time step of the simulation."""
        self._time_step += 1

        # 0. Apply beam control actions first (affects current slot's communication)
        if config.BEAM_CONTROL_ENABLED:
            self._apply_beam_actions(actions)

        # 1. Process requests using current time slot state
        # 先初始化所有 UAV 的 working_cache，避免竞态条件
        # （协作 UAV 可能在请求处理时修改其他 UAV 的 _working_cache）
        for uav in self._uavs:
            uav.init_working_cache()
        for uav in self._uavs:
            uav.process_requests()

        for ue in self._ues:
            ue.update_service_coverage(self._time_step)

        for uav in self._uavs:
            uav.update_ema_and_cache()

        # 2. Execute UAV movement (updates _dist_moved)
        self._apply_actions_to_env(actions)

        # 3. Calculate energy (flight energy uses _dist_moved, comm energy uses this slot's time)
        for uav in self._uavs:
            uav.update_energy_consumption()

        rewards, metrics = self._get_rewards_and_metrics()

        if self._time_step % config.T_CACHE_UPDATE_INTERVAL == 0:
            for uav in self._uavs:
                uav.gdsf_cache_update()

        # 4. Prepare for next time slot
        for ue in self._ues:
            ue.update_position()

        for uav in self._uavs:
            uav.reset_for_next_step()

        self._prepare_for_next_step()
        next_obs: list[np.ndarray] = self._get_obs()
        return next_obs, rewards, metrics

    def _prepare_for_next_step(self) -> None:
        """Prepare environment state for the next time step.
        
        This includes:
        1. Generate UE requests
        2. Associate UEs to UAVs
        3. Set UAV requested files and neighbors
        4. Select collaborators
        5. Count requesting UAVs for bandwidth allocation
        6. Set communication rates
        7. Set frequency counts for caching policy
        """
        # 1. Generate requests for all UEs
        for ue in self._ues:
            ue.generate_request()
        
        # 2. Associate UEs to UAVs based on coverage
        self._associate_ues_to_uavs()
        
        # 3. Set requested files and neighbors for each UAV
        for uav in self._uavs:
            uav.set_current_requested_files()
            uav.set_neighbors(self._uavs)
        
        # 4. Select collaborator for each UAV
        for uav in self._uavs:
            uav.select_collaborator()
        
        # 5. Count how many UAVs selected each UAV as collaborator (for FDM bandwidth allocation)
        for uav in self._uavs:
            if uav.current_collaborator:
                uav.current_collaborator._num_requesting_uavs += 1
        
        # 6. Set communication rates (must be after _num_requesting_uavs is counted)
        for uav in self._uavs:
            uav._set_rates()
        
        # 7. Set frequency counts for GDSF caching policy
        for uav in self._uavs:
            uav.set_freq_counts()

    def _get_obs(self) -> list[np.ndarray]:
        """Construct local observation vector for each UAV agent.
        
        Observation structure per UAV:
        - Own state: position (2) + cache (NUM_FILES)
        - Neighbors: (position (2) + cache (NUM_FILES)) × MAX_UAV_NEIGHBORS
        - UEs: (position (2) + request_info (3)) × MAX_ASSOCIATED_UES
        """
        all_obs: list[np.ndarray] = []
        
        for uav in self._uavs:
            # Part 1: Own state (position and cache status)
            own_pos: np.ndarray = uav.pos[:2] / np.array([config.AREA_WIDTH, config.AREA_HEIGHT])
            own_cache: np.ndarray = uav.cache.astype(np.float32)
            own_state: np.ndarray = np.concatenate([own_pos, own_cache])

            # Part 2: Neighbors state (positions and cache status)
            neighbor_states: np.ndarray = np.zeros((config.MAX_UAV_NEIGHBORS, 2 + config.NUM_FILES))
            neighbors: list[UAV] = sorted(uav.neighbors, key=lambda n: float(np.linalg.norm(uav.pos - n.pos)))[: config.MAX_UAV_NEIGHBORS]
            for i, neighbor in enumerate(neighbors):
                relative_pos: np.ndarray = (neighbor.pos[:2] - uav.pos[:2]) / config.UAV_SENSING_RANGE
                neighbor_cache: np.ndarray = neighbor.cache.astype(np.float32)
                neighbor_states[i, :] = np.concatenate([relative_pos, neighbor_cache])

            # Part 3: State of associated UEs
            ue_states: np.ndarray = np.zeros((config.MAX_ASSOCIATED_UES, 2 + 3))
            ues: list[UE] = sorted(uav.current_covered_ues, key=lambda u: float(np.linalg.norm(uav.pos[:2] - u.pos[:2])))[: config.MAX_ASSOCIATED_UES]
            for i, ue in enumerate(ues):
                delta_pos: np.ndarray = (ue.pos[:2] - uav.pos[:2]) / config.AREA_WIDTH
                req_type, _, req_id = ue.current_request
                norm_id: float = float(req_id) / float(config.NUM_FILES)
                file_size: float = float(config.FILE_SIZES[req_id])
                max_file_size: float = float(np.max(config.FILE_SIZES))
                norm_size: float = file_size / max_file_size
                request_info: np.ndarray = np.array([req_type, norm_size, norm_id], dtype=np.float32)
                ue_states[i, :] = np.concatenate([delta_pos, request_info])

            # Part 4: Combine all parts into a single, flat observation vector
            obs: np.ndarray = np.concatenate([own_state, neighbor_states.flatten(), ue_states.flatten()])
            all_obs.append(obs)

        return all_obs

    def _apply_actions_to_env(self, actions: np.ndarray) -> None:
        """Calculates next positions and resolves potential collisions iteratively."""
        current_positions: np.ndarray = np.array([uav.pos[:2] for uav in self._uavs])
        max_dist: float = config.UAV_SPEED * config.TIME_SLOT_DURATION

        # Extract movement actions (first 2 dimensions only)
        movement_actions: np.ndarray = np.array(actions[:, :2], dtype=np.float32)

        # Interpret actions as a direct (x, y) vector
        delta_vec_raw: np.ndarray = movement_actions

        # Calculate the magnitude (distance) of this raw vector
        raw_magnitude: np.ndarray = np.linalg.norm(delta_vec_raw, axis=1, keepdims=True)

        # Clip the magnitude to be at most 1.0
        clipped_magnitude: np.ndarray = np.minimum(raw_magnitude, 1.0)
        distances: np.ndarray = clipped_magnitude * max_dist
        denom: np.ndarray = raw_magnitude + float(config.EPSILON)
        directions: np.ndarray = delta_vec_raw / denom
        delta_pos: np.ndarray = directions * distances

        proposed_positions: np.ndarray = current_positions + delta_pos

        min_boundary_gap: float = config.UAV_COVERAGE_RADIUS / 2.0
        for i, uav in enumerate(self._uavs):
            if not (min_boundary_gap <= proposed_positions[i, 0] <= config.AREA_WIDTH - min_boundary_gap and min_boundary_gap <= proposed_positions[i, 1] <= config.AREA_HEIGHT - min_boundary_gap):
                uav.boundary_violation = True
        next_positions: np.ndarray = np.clip(proposed_positions, [min_boundary_gap, min_boundary_gap], [config.AREA_WIDTH - min_boundary_gap, config.AREA_HEIGHT - min_boundary_gap])

        min_sep_sq: float = config.MIN_UAV_SEPARATION**2
        for _ in range(config.COLLISION_AVOIDANCE_ITERATIONS + 1):
            collision_detected_in_iter: bool = False
            for i in range(config.NUM_UAVS):
                for j in range(i + 1, config.NUM_UAVS):
                    pos_i: np.ndarray = next_positions[i]
                    pos_j: np.ndarray = next_positions[j]
                    dist_sq: float = np.sum((pos_i - pos_j) ** 2)
                    if dist_sq < min_sep_sq:
                        self._uavs[i].collision_violation = True
                        self._uavs[j].collision_violation = True
                        collision_detected_in_iter = True
                        
                        dist: float = np.sqrt(dist_sq)
                        if dist < config.EPSILON:
                            # If positions are identical, apply random direction
                            direction = np.random.randn(2)
                            direction /= (np.linalg.norm(direction) + config.EPSILON)
                            dist = config.EPSILON
                        else:
                            direction = (pos_i - pos_j) / dist
                            
                        overlap: float = config.MIN_UAV_SEPARATION - dist
                        next_positions[i] += direction * overlap * 0.5
                        next_positions[j] -= direction * overlap * 0.5
            if not collision_detected_in_iter:
                break

        final_positions: np.ndarray = np.clip(next_positions, [min_boundary_gap, min_boundary_gap], [config.AREA_WIDTH - min_boundary_gap, config.AREA_HEIGHT - min_boundary_gap])
        for i, uav in enumerate(self._uavs):
            uav.update_position(final_positions[i])

    def _apply_beam_actions(self, actions: np.ndarray) -> None:
        """Apply beam control actions from the agent.
        
        Actions format: [dx, dy, beam_theta, beam_phi] where beam_* are in [-1, 1]
        
        Two modes:
        - offset: beam angles are offsets from centroid direction
        - absolute: beam angles are absolute values
        """
        for i, uav in enumerate(self._uavs):
            if actions.shape[1] < 4:
                continue  # No beam control in action
            
            beam_action_theta = float(actions[i, 2])
            beam_action_phi = float(actions[i, 3])
            
            if config.BEAM_CONTROL_MODE == "offset":
                # Offset mode: [-1, 1] -> [-BEAM_OFFSET_RANGE, +BEAM_OFFSET_RANGE]
                delta_theta = beam_action_theta * config.BEAM_OFFSET_RANGE
                delta_phi = beam_action_phi * config.BEAM_OFFSET_RANGE
                uav.set_beam_offset(delta_theta, delta_phi)
            else:
                # Absolute mode: [-1, 1] -> [0, 90] for theta, [-180, 180] for phi
                theta = (beam_action_theta + 1.0) / 2.0 * 90.0  # [0, 90]
                phi = beam_action_phi * 180.0                    # [-180, 180]
                uav.set_beam_absolute(theta, phi)

    def _associate_ues_to_uavs(self) -> None:
        """Assigns each UE to at most one UAV, resolving overlaps by choosing the closest UAV."""
        for ue in self._ues:
            covering_uavs: list[tuple[UAV, float]] = []
            for uav in self._uavs:
                distance: float = float(np.linalg.norm(uav.pos[:2] - ue.pos[:2]))
                if distance <= config.UAV_COVERAGE_RADIUS:
                    covering_uavs.append((uav, distance))

            if not covering_uavs:
                continue
            best_uav, _ = min(covering_uavs, key=lambda x: x[1])
            best_uav.current_covered_ues.append(ue)
            ue.assigned = True

    def _get_rewards_and_metrics(self) -> tuple[list[float], tuple[float, float, float]]:
        """Returns the reward and other metrics."""
        total_latency: float = sum(ue.latency_current_request if ue.assigned else config.NON_SERVED_LATENCY_PENALTY for ue in self._ues)
        total_energy: float = sum(uav.energy for uav in self._uavs)
        sc_metrics: np.ndarray = np.array([ue.service_coverage for ue in self._ues])
        jfi: float = 0.0
        if sc_metrics.size > 0 and np.sum(sc_metrics**2) > 0:
            jfi = (np.sum(sc_metrics) ** 2) / (sc_metrics.size * np.sum(sc_metrics**2))

        r_fairness: float = config.ALPHA_3 * np.log(jfi + config.EPSILON)
        r_latency: float = config.ALPHA_1 * np.log(total_latency + config.EPSILON)
        r_energy: float = config.ALPHA_2 * np.log(total_energy + config.EPSILON)
        
        # 系统传输速率奖励（波束控制的直接反馈）
        total_rate: float = sum(uav.total_downlink_rate for uav in self._uavs)
        # 归一化：使用参考速率避免数值过大
        reference_rate: float = config.BANDWIDTH_EDGE * np.log2(1 + 100)  # 参考SNR=100时的速率
        normalized_rate: float = total_rate / (config.NUM_UES * reference_rate + config.EPSILON)
        r_rate: float = config.ALPHA_RATE * np.log(normalized_rate + config.EPSILON)
        
        reward: float = r_fairness + r_rate - r_latency - r_energy
        rewards: list[float] = [reward] * config.NUM_UAVS
        for uav in self._uavs:
            if uav.collision_violation:
                rewards[uav.id] -= config.COLLISION_PENALTY
            if uav.boundary_violation:
                rewards[uav.id] -= config.BOUNDARY_PENALTY
        rewards = [r * config.REWARD_SCALING_FACTOR for r in rewards]
        return rewards, (total_latency, total_energy, jfi)
