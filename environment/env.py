from environment.user_equipments import UE
from environment.uavs import UAV
import config
import numpy as np
# 基于强化学习框架，模拟多无人机（UAV）空中基站通信保障环境，管理 UAV、用户设备（UE）和宏基站（MBS）的状态、动作和奖励。


class RunningNormalizer:
    """使用指数移动平均的动态归一化器，用于平衡多目标奖励的量级。"""
    
    def __init__(self, momentum: float = 0.99) -> None:
        self.momentum = momentum
        self.mean = 0.0
        self.var = 1.0
        self.count = 0
    
    def normalize(self, x: float) -> float:
        """归一化输入值并更新统计量。"""
        self.count += 1
        if self.count == 1:
            self.mean = x
            self.var = 1.0
        else:
            delta = x - self.mean
            self.mean = self.momentum * self.mean + (1 - self.momentum) * x
            self.var = self.momentum * self.var + (1 - self.momentum) * delta ** 2
        
        std = np.sqrt(self.var)
        # 添加安全下限，防止方差过小时导致除零或梯度爆炸
        if std < 1e-6:
            std = 1e-6
        return (x - self.mean) / std


class Env:
    def __init__(self) -> None:
        self._mbs_pos: np.ndarray = config.MBS_POS
        UE.initialize_ue_class()
        self._ues: list[UE] = [UE(i) for i in range(config.NUM_UES)]
        self._uavs: list[UAV] = [UAV(i) for i in range(config.NUM_UAVS)]
        self._time_step: int = 0
        
        # 动态归一化器：跨 episode 积累统计量，平衡各奖励分量的量级
        self._latency_normalizer = RunningNormalizer()
        self._energy_normalizer = RunningNormalizer()
        self._jfi_normalizer = RunningNormalizer()
        self._rate_normalizer = RunningNormalizer()

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

    def step(self, actions: np.ndarray) -> tuple[list[np.ndarray], list[float], tuple[float, float, float, float]]:
        """Execute one time step of the simulation."""
        self._time_step += 1

        # 0. Apply beam control actions first (affects current slot's communication)
        if config.BEAM_CONTROL_ENABLED:
            self._apply_beam_actions(actions)

        # 1. Calculate co-channel interference AFTER beam actions are applied
        # 确保干扰计算使用的波束方向与信号计算一致
        self._calculate_ue_interference()

        # 2. Process requests using current time slot state
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
        - Own state: position (3) + cache (NUM_FILES)
        - Neighbors: (position (3) + cache (NUM_FILES)) × MAX_UAV_NEIGHBORS
        - UEs: (position (3) + request_info (3) [+ direction_angles (2)]) × MAX_ASSOCIATED_UES
        
        When BEAM_CONTROL_ENABLED, UE states include spherical direction angles (theta, phi)
        to help the agent learn the mapping from observation to beam direction actions.
        """
        all_obs: list[np.ndarray] = []
        
        # Normalization constants for 3D positions
        # 使用 UE 最大高度作为 z 归一化常量（因为 UE 可以比 UAV 更高）
        pos_norm = np.array([config.AREA_WIDTH, config.AREA_HEIGHT, config.UE_MAX_ALT])
        
        for uav in self._uavs:
            # Part 1: Own state (position and cache status)
            own_pos: np.ndarray = uav.pos / pos_norm
            own_cache: np.ndarray = uav.cache.astype(np.float32)
            own_state: np.ndarray = np.concatenate([own_pos, own_cache])

            # Part 2: Neighbors state (positions and cache status)
            # Compressed neighbor features: 3 (pos) + 2 (cache stats) = 5 dims
            neighbor_states: np.ndarray = np.zeros((config.MAX_UAV_NEIGHBORS, 3 + 2))
            neighbors: list[UAV] = sorted(uav.neighbors, key=lambda n: float(np.linalg.norm(uav.pos - n.pos)))[: config.MAX_UAV_NEIGHBORS]
            
            # Pre-calculate current requested files by served UEs for "Immediate Reward" feature
            my_requested_files = set()
            for ue in uav.current_covered_ues:
                req_type, _, req_id = ue.current_request
                if req_type == 1: # Only consider content requests
                    my_requested_files.add(req_id)

            for i, neighbor in enumerate(neighbors):
                # Relative 3D position normalized by sensing range
                relative_pos: np.ndarray = (neighbor.pos - uav.pos) / config.UAV_SENSING_RANGE
                
                # Feature 1: Immediate Help Capability (Immediate Reward)
                # Does neighbor have ANY file I currently need?
                # 1.0 if neighbor has at least one file I need, 0.0 otherwise
                immediate_help = 0.0
                for file_id in my_requested_files:
                    if neighbor.cache[file_id]:
                        immediate_help = 1.0
                        break
                
                # Feature 2: Cache Similarity (Long-term Potential)
                # Jaccard Similarity: Intersection / Union
                # We use 1 - Similarity to represent "Complementarity" or "Difference"
                # If we are very different (Similarity low), Complementarity is high (1.0)
                intersection = np.sum(np.logical_and(uav.cache, neighbor.cache))
                union = np.sum(np.logical_or(uav.cache, neighbor.cache))
                similarity = intersection / (union + config.EPSILON)
                complementarity = 1.0 - similarity
                
                neighbor_features = np.array([immediate_help, complementarity], dtype=np.float32)
                neighbor_states[i, :] = np.concatenate([relative_pos, neighbor_features])

            # Part 3: State of associated UEs
            ue_states: np.ndarray = np.zeros((config.MAX_ASSOCIATED_UES, config.UE_STATE_DIM))
            ues: list[UE] = sorted(uav.current_covered_ues, key=lambda u: float(np.linalg.norm(uav.pos - u.pos)))[: config.MAX_ASSOCIATED_UES]
            for i, ue in enumerate(ues):
                # Relative 3D position normalized by coverage radius (球形覆盖范围内)
                raw_delta: np.ndarray = ue.pos - uav.pos
                delta_pos: np.ndarray = raw_delta / config.UAV_COVERAGE_RADIUS
                req_type, _, req_id = ue.current_request
                norm_id: float = float(req_id) / float(config.NUM_FILES)
                file_size: float = float(config.FILE_SIZES[req_id])
                max_file_size: float = float(np.max(config.FILE_SIZES))
                norm_size: float = file_size / max_file_size
                request_info: np.ndarray = np.array([req_type, norm_size, norm_id], dtype=np.float32)
                
                if config.BEAM_CONTROL_ENABLED:
                    # Calculate spherical direction angles for beam control guidance
                    # These angles directly correspond to the action output for absolute beam control
                    # Action mapping: theta = (action + 1) / 2 * 180, so action = theta/90 - 1
                    # Observation should output the same normalized value as the desired action
                    dist: float = float(np.linalg.norm(raw_delta)) + config.EPSILON
                    # theta: 0° = zenith (z+), 180° = nadir (z-)
                    # arccos(z/dist) gives angle from z+ axis: z>0 → small angle, z<0 → large angle
                    theta_rad: float = np.arccos(np.clip(raw_delta[2] / dist, -1.0, 1.0))
                    phi_rad: float = np.arctan2(raw_delta[1], raw_delta[0])  # -pi to pi
                    # Normalize to match action space: action = theta/90 - 1, so theta_norm ∈ [-1, 1]
                    theta_norm: float = (theta_rad / np.pi) * 2.0 - 1.0  # [0, pi] -> [-1, 1]
                    phi_norm: float = phi_rad / np.pi  # [-pi, pi] -> [-1, 1]
                    direction_angles: np.ndarray = np.array([theta_norm, phi_norm], dtype=np.float32)
                    ue_states[i, :] = np.concatenate([delta_pos, request_info, direction_angles])
                else:
                    ue_states[i, :] = np.concatenate([delta_pos, request_info])

            # Part 4: Combine all parts into a single, flat observation vector
            obs: np.ndarray = np.concatenate([own_state, neighbor_states.flatten(), ue_states.flatten()])
            all_obs.append(obs)

        return all_obs

    def _apply_actions_to_env(self, actions: np.ndarray) -> None:
        """Calculates next 3D positions and resolves potential collisions iteratively."""
        current_positions: np.ndarray = np.array([uav.pos for uav in self._uavs])  # Full 3D positions
        max_dist: float = config.UAV_SPEED * config.TIME_SLOT_DURATION

        # Extract movement actions (first 3 dimensions: dx, dy, dz)
        movement_actions: np.ndarray = np.array(actions[:, :3], dtype=np.float32)

        # Interpret actions as a direct (x, y, z) vector
        delta_vec_raw: np.ndarray = movement_actions

        # Calculate the magnitude (distance) of this raw 3D vector
        raw_magnitude: np.ndarray = np.linalg.norm(delta_vec_raw, axis=1, keepdims=True)

        # Clip the magnitude to be at most 1.0
        clipped_magnitude: np.ndarray = np.minimum(raw_magnitude, 1.0)
        distances: np.ndarray = clipped_magnitude * max_dist
        denom: np.ndarray = raw_magnitude + float(config.EPSILON)
        directions: np.ndarray = delta_vec_raw / denom
        delta_pos: np.ndarray = directions * distances

        proposed_positions: np.ndarray = current_positions + delta_pos

        # Check boundary violations (3D)
        min_boundary_gap: float = config.UAV_COVERAGE_RADIUS / 2.0
        for i, uav in enumerate(self._uavs):
            in_xy_bounds = (min_boundary_gap <= proposed_positions[i, 0] <= config.AREA_WIDTH - min_boundary_gap and 
                           min_boundary_gap <= proposed_positions[i, 1] <= config.AREA_HEIGHT - min_boundary_gap)
            in_z_bounds = config.UAV_MIN_ALT <= proposed_positions[i, 2] <= config.UAV_MAX_ALT
            if not (in_xy_bounds and in_z_bounds):
                uav.boundary_violation = True
        
        # Clip to valid 3D boundaries
        next_positions: np.ndarray = np.clip(
            proposed_positions, 
            [min_boundary_gap, min_boundary_gap, config.UAV_MIN_ALT], 
            [config.AREA_WIDTH - min_boundary_gap, config.AREA_HEIGHT - min_boundary_gap, config.UAV_MAX_ALT]
        )

        # 3D collision detection and resolution
        min_sep_sq: float = config.MIN_UAV_SEPARATION**2
        for _ in range(config.COLLISION_AVOIDANCE_ITERATIONS + 1):
            collision_detected_in_iter: bool = False
            for i in range(config.NUM_UAVS):
                for j in range(i + 1, config.NUM_UAVS):
                    pos_i: np.ndarray = next_positions[i]
                    pos_j: np.ndarray = next_positions[j]
                    dist_sq: float = np.sum((pos_i - pos_j) ** 2)  # 3D distance squared
                    if dist_sq < min_sep_sq:
                        self._uavs[i].collision_violation = True
                        self._uavs[j].collision_violation = True
                        collision_detected_in_iter = True
                        
                        dist: float = np.sqrt(dist_sq)
                        if dist < config.EPSILON:
                            # If positions are identical, apply random 3D direction
                            direction = np.random.randn(3)
                            direction /= (np.linalg.norm(direction) + config.EPSILON)
                            dist = config.EPSILON
                        else:
                            direction = (pos_i - pos_j) / dist
                            
                        overlap: float = config.MIN_UAV_SEPARATION - dist
                        next_positions[i] += direction * overlap * 0.5
                        next_positions[j] -= direction * overlap * 0.5
            if not collision_detected_in_iter:
                break

        # Final clip to ensure within 3D boundaries after collision resolution
        final_positions: np.ndarray = np.clip(
            next_positions, 
            [min_boundary_gap, min_boundary_gap, config.UAV_MIN_ALT], 
            [config.AREA_WIDTH - min_boundary_gap, config.AREA_HEIGHT - min_boundary_gap, config.UAV_MAX_ALT]
        )
        for i, uav in enumerate(self._uavs):
            uav.update_position(final_positions[i])

    def _apply_beam_actions(self, actions: np.ndarray) -> None:
        """Apply beam control actions from the agent.
        
        Actions format: [dx, dy, dz, beam_theta, beam_phi] where beam_* are in [-1, 1]
        
        Two modes:
        - offset: beam angles are offsets from centroid direction
        - absolute: beam angles are absolute values
        """
        for i, uav in enumerate(self._uavs):
            if actions.shape[1] < 5:
                continue  # No beam control in action
            
            beam_action_theta = float(actions[i, 3])
            beam_action_phi = float(actions[i, 4])
            
            if config.BEAM_CONTROL_MODE == "offset":
                # Offset mode: [-1, 1] -> [-BEAM_OFFSET_RANGE, +BEAM_OFFSET_RANGE]
                delta_theta = beam_action_theta * config.BEAM_OFFSET_RANGE
                delta_phi = beam_action_phi * config.BEAM_OFFSET_RANGE
                uav.set_beam_offset(delta_theta, delta_phi)
            else:
                # Absolute mode: [-1, 1] -> [0, 180] for theta (full sphere), [-180, 180] for phi
                theta = (beam_action_theta + 1.0) / 2.0 * 180.0  # [0, 180]
                phi = beam_action_phi * 180.0                     # [-180, 180]
                uav.set_beam_absolute(theta, phi)

    def _associate_ues_to_uavs(self) -> None:
        """Assigns each UE to at most one UAV using 3D spherical coverage."""
        for ue in self._ues:
            covering_uavs: list[tuple[UAV, float]] = []
            for uav in self._uavs:
                # 使用 3D 距离判断球形覆盖范围
                distance_3d: float = float(np.linalg.norm(uav.pos - ue.pos))
                if distance_3d <= config.UAV_COVERAGE_RADIUS:
                    covering_uavs.append((uav, distance_3d))

            if not covering_uavs:
                continue
            best_uav, _ = min(covering_uavs, key=lambda x: x[1])
            best_uav.current_covered_ues.append(ue)
            ue.assigned = True

    def _calculate_ue_interference(self) -> None:
        """Calculate co-channel interference for each UE from non-serving UAVs.
        
        对于每个被服务的UE，计算来自所有其他UAV的同频干扰功率总和。
        干扰功率取决于：
        1. 干扰UAV到UE的距离和信道增益
        2. 干扰UAV的波束方向（3D beamforming）
        3. 干扰UAV是否有关联UE（无关联则不发射）
        """
        from environment import comm_model as comms
        
        # 预计算每个UAV的波束方向和关联UE数量
        uav_info: list[tuple[np.ndarray, tuple[float, float], int]] = []
        for uav in self._uavs:
            beam_dir = uav.get_final_beam_direction()
            num_ues = len(uav.current_covered_ues)
            uav_info.append((uav.pos, beam_dir, num_ues))
        
        # 为每个被服务的UE计算干扰
        for serving_uav_idx, uav in enumerate(self._uavs):
            for ue in uav.current_covered_ues:
                total_interference: float = 0.0
                
                # 累加来自所有其他UAV的干扰
                for interferer_idx, (interferer_pos, interferer_beam, interferer_num_ues) in enumerate(uav_info):
                    if interferer_idx == serving_uav_idx:
                        continue  # 跳过服务UAV本身
                    
                    # 计算该干扰UAV对此UE的干扰功率
                    interference = comms.calculate_interference_power(
                        interferer_pos, ue.pos, interferer_beam, interferer_num_ues
                    )
                    total_interference += interference
                
                ue.interference_power = total_interference

    def _get_rewards_and_metrics(self) -> tuple[list[float], tuple[float, float, float, float]]:
        """Returns the reward and other metrics (latency, energy, jfi, total_rate).
        
        使用动态归一化平衡各奖励分量：
        1. 先对原始指标取 log（压缩量级差异）
        2. 使用 RunningNormalizer 动态归一化到 ~N(0,1)
        3. 各分量乘以权重后相加
        """
        total_latency: float = sum(ue.latency_current_request if ue.assigned else config.NON_SERVED_LATENCY_PENALTY for ue in self._ues)
        total_energy: float = sum(uav.energy for uav in self._uavs)
        total_rate: float = sum(uav.total_downlink_rate for uav in self._uavs)
        
        sc_metrics: np.ndarray = np.array([ue.service_coverage for ue in self._ues])
        jfi: float = 0.0
        if sc_metrics.size > 0 and np.sum(sc_metrics**2) > 0:
            jfi = (np.sum(sc_metrics) ** 2) / (sc_metrics.size * np.sum(sc_metrics**2))

        # 先取 log 再动态归一化，使各分量量级一致
        log_latency = np.log(total_latency + config.EPSILON)
        log_energy = np.log(total_energy + config.EPSILON)
        log_jfi = np.log(jfi + config.EPSILON)
        log_rate = np.log(total_rate + config.EPSILON)
        
        r_latency: float = config.ALPHA_1 * self._latency_normalizer.normalize(log_latency)
        r_energy: float = config.ALPHA_2 * self._energy_normalizer.normalize(log_energy)
        r_fairness: float = config.ALPHA_3 * self._jfi_normalizer.normalize(log_jfi)
        r_rate: float = config.ALPHA_RATE * self._rate_normalizer.normalize(log_rate)
        
        # 奖励 = 正向指标 - 负向指标
        reward: float = r_fairness + r_rate - r_latency - r_energy
        rewards: list[float] = [reward] * config.NUM_UAVS
        for uav in self._uavs:
            if uav.collision_violation:
                rewards[uav.id] -= config.COLLISION_PENALTY
            if uav.boundary_violation:
                rewards[uav.id] -= config.BOUNDARY_PENALTY
        rewards = [r * config.REWARD_SCALING_FACTOR for r in rewards]
        return rewards, (total_latency, total_energy, jfi, total_rate)
