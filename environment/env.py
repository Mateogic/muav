from environment.user_equipments import UE
from environment.uavs import UAV
import config
import numpy as np
# 基于强化学习框架，模拟多无人机（UAV）空中基站通信保障环境，管理 UAV、用户设备（UE）和宏基站（MBS）的状态、动作和奖励。


class RunningNormalizer:
    """使用指数移动平均的动态归一化器，用于平衡多目标奖励的量级。
    
    关键设计：先用旧统计量归一化当前值，再更新统计量。
    这避免了当前样本影响自身归一化结果的问题。
    """
    
    def __init__(self, momentum: float = 0.96) -> None:
        self.momentum = momentum
        self.mean = 0.0
        self.var = 1.0
        self.count = 0
    
    def normalize(self, x: float) -> float:
        """先用旧统计量归一化，再更新统计量。"""
        self.count += 1
        
        if self.count == 1:
            # 第一个样本：初始化统计量，返回 0.0（无历史参考）
            self.mean = x
            self.var = 1.0
            return 0.0
        
        # 用旧统计量归一化当前值
        std = np.sqrt(self.var) + 1e-8
        normalized = (x - self.mean) / std
        
        # 更新统计量（Welford 在线算法的 EMA 变体）
        delta_old = x - self.mean
        self.mean = self.momentum * self.mean + (1 - self.momentum) * x
        delta_new = x - self.mean
        self.var = self.momentum * self.var + (1 - self.momentum) * delta_old * delta_new
        
        # 防止方差变为负数或过小
        self.var = max(self.var, 1e-6)
        
        return normalized


class Env:
    def __init__(self) -> None:
        self._mbs_pos: np.ndarray = config.MBS_POS
        UE.initialize_ue_class()
        self._ues: list[UE] = [UE(i) for i in range(config.NUM_UES)]
        self._uavs: list[UAV] = [UAV(i) for i in range(config.NUM_UAVS)]
        self._time_step: int = 0
        
        # 动态归一化器：跨 episode 积累统计量，平衡各奖励分量的量级
        # 注：JFI 使用固定映射，不需要动态归一化
        self._latency_normalizer = RunningNormalizer()
        self._energy_normalizer = RunningNormalizer()
        self._rate_normalizer = RunningNormalizer()

        # B1 丢包建模与超时诊断统计（在步长的不同阶段累计，供日志使用）
        self._pending_bits_dropped_last_boundary: float = 0.0
        self._backlog_bits_dropped_last_boundary: float = 0.0
        self._failed_requests_dropped_boundary: int = 0
        
        self._pending_bits_dropped_timeout: float = 0.0
        self._backlog_bits_dropped_timeout: float = 0.0
        self._failed_requests_dropped_timeout: int = 0

    @property
    def uavs(self) -> list[UAV]:
        return self._uavs

    @property
    def ues(self) -> list[UE]:
        return self._ues

    def reset(self) -> list[np.ndarray]:
        """Resets the environment to an initial state and returns the initial observations."""
        self._uavs = [UAV(i) for i in range(config.NUM_UAVS)]
        self._ues = [UE(i) for i in range(config.NUM_UES)]
        self._time_step = 0
        
        # 初始化物理拓扑 (coverage/associations)
        self._associate_ues_to_uavs()
        
        self._prepare_for_next_step()
        self._update_topology_and_collaborators()
        self._update_contention_and_rates()
        return self._get_obs()

    def step(self, actions: np.ndarray) -> tuple[list[np.ndarray], list[float], tuple[float, float, float, float, dict, int, int]]:
        """Execute one time step of the simulation."""
        self._time_step += 1

        # Reset UE per-step flags (delivered bits / completion markers)
        for ue in self._ues:
            ue.reset_step_flags()

        # 0. 立即执行移动并更新物理拓扑 (使 Action 直接影响产生的比特率和奖励)
        self._apply_actions_to_env(actions)
        for ue in self._ues:
            ue.update_position()
        
        # 应用波束控制动作 (针对新位置调整)
        if config.BEAM_CONTROL_ENABLED:
            self._apply_beam_actions(actions)

        # 刷新覆盖关系与 B1 强制丢包检测 (UE 移出/移入)
        self._associate_ues_to_uavs()
        self._check_and_drop_boundary_ues()

        # 阶段 A: 更新网络拓扑与协作者选择 (基于移动后的新位置)
        self._update_topology_and_collaborators()

        # 1. 在新位置下计算信道干扰 (用于 UE 下行速率)
        self._calculate_ue_interference()

        # 2. 处理请求 (新注入本时隙的任务量)
        for uav in self._uavs:
            uav.init_working_cache()
        for uav in self._uavs:
            uav.process_requests(self._time_step)

        # 阶段 B: 根据包含新任务的总负载，更新 MBS 和 U2U 的物理速率 (修正时序断层)
        self._update_contention_and_rates()

        for uav in self._uavs:
            uav.update_ema_and_cache(self._time_step)

        # 3. 计算能耗与传输比特 (使用包含新任务的真实速率)
        for uav in self._uavs:
            uav.update_energy_consumption(self._uavs, self._time_step)

        # 缓存落盘：确保本时隙完成传输的文件正式进入物理缓存
        for uav in self._uavs:
            uav.cache = uav._working_cache.copy()

        # Enforce UE-side timeout FIRST (to ensure failed requests don't count as success in fairness)
        self._handle_request_timeouts()

        # Update UE fairness metric (after transmissions and timeout check)
        for ue in self._ues:
            ue.update_service_coverage(self._time_step)

        rewards, metrics = self._get_rewards_and_metrics()

        if self._time_step % config.T_CACHE_UPDATE_INTERVAL == 0:
            for uav in self._uavs:
                uav.gdsf_cache_update()

        # 4. Prepare for next time slot
        # Count violations
        step_collisions = sum(1 for uav in self._uavs if uav.collision_violation)
        step_boundaries = sum(1 for uav in self._uavs if uav.boundary_violation)

        for uav in self._uavs:
            uav.reset_for_next_step()

        self._prepare_for_next_step()
        next_obs: list[np.ndarray] = self._get_obs()
        
        # 返回结构: obs, rewards, (latency, energy, jfi, rate, stats, collisions, boundaries)
        return next_obs, rewards, metrics + (step_collisions, step_boundaries)

    def _check_and_drop_boundary_ues(self) -> None:
        """B1 丢包建模：UE 离开覆盖范围即会话结束（在移动后立即生效）。"""
        # 重置本时隙所有丢包统计
        self._pending_bits_dropped_last_boundary = 0.0
        self._backlog_bits_dropped_last_boundary = 0.0
        self._failed_requests_dropped_boundary = 0
        
        self._pending_bits_dropped_timeout = 0.0
        self._backlog_bits_dropped_timeout = 0.0
        self._failed_requests_dropped_timeout = 0
        
        failed_ue_ids: set[int] = set()
        for uav in self._uavs:
            covered_ue_ids = {u.id for u in uav.current_covered_ues}
            
            # 搜集离开本 UAV 覆盖范围但仍有在途任务的 UE
            ues_leaving = set()
            for ue_id in uav._backlog_tx_ue_uav:
                if ue_id not in covered_ue_ids: ues_leaving.add(ue_id)
            for t in uav._mbs_rx_tasks:
                if t["target_type"] == "UE" and t["ue_id"] not in covered_ue_ids:
                    ues_leaving.add(t["ue_id"])
                elif t["target_type"] == "COL":
                    # 关键修复：协作任务中，检查 UE 是否离开“请求者” UAV 的覆盖范围，而非协作者
                    req_uav = self._uavs[t["target_id"]]
                    req_covered_ids = {u.id for u in req_uav.current_covered_ues}
                    if t["ue_id"] not in req_covered_ids: ues_leaving.add(t["ue_id"])
            for tasks in uav._u2u_rx_tasks.values():
                for t in tasks:
                    if t["target_type"] == "UE" and t["ue_id"] not in covered_ue_ids: ues_leaving.add(t["ue_id"])
            for wait_list in uav._u2u_wait_tasks.values():
                for _, ue_id in wait_list:
                    if ue_id not in covered_ue_ids: ues_leaving.add(ue_id)

            for uid in ues_leaving:
                failed_ue_ids.add(uid)

        # 统一清理全网流水线，确保每个失败的 UE 只处理一次（防止 Case 3 等导致重复计数）
        for ue_id in failed_ue_ids:
            self._drop_ue_pipeline_across_uavs(ue_id)

        # 针对失败的请求进行状态更新
        for ue_id in failed_ue_ids:
            if 0 <= ue_id < len(self._ues) and self._ues[ue_id].request_active:
                self._ues[ue_id].on_request_failed(self._time_step)

    def _prepare_for_next_step(self) -> None:
        """Prepare environment state for the next time step.
        
        This includes:
        1. Generate UE requests
        
        Note: Physical topology updates (RequestedFiles, Neighbors, Collaborators, Rates, FreqCounts) 
        are handled in _update_physical_network_and_rates() after movement in step().
        """
        # 1. Generate (or keep) requests for the upcoming slot
        for ue in self._ues:
            ue.generate_request(self._time_step + 1)

    def _update_topology_and_collaborators(self) -> None:
        """更新拓扑结构和协作者选择 (基于当前位置)。"""
        # 0. Set requested files for each UAV (Based on new associations)
        for uav in self._uavs:
            uav.set_current_requested_files()

        # 1. Update Neighbors (Topology) based on new positions
        for uav in self._uavs:
            uav.set_neighbors(self._uavs)

        # 2. Select Collaborators
        for uav in self._uavs:
            uav.select_collaborator()

    def _update_contention_and_rates(self) -> None:
        """统计链路竞争状况并设定最终物理速率 (基于当前位置和所有活跃任务)。"""
        # 3. Reset and Count Contention (U2U and MBS)
        # Clear previous contention counters
        for uav in self._uavs:
            uav._requesting_uav_ids.clear()
            uav._num_mbs_users = 1 # Default

        # Count U2U contention
        for uav in self._uavs:
            active_collab_ids = set()
            if uav.current_collaborator:
                active_collab_ids.add(uav.current_collaborator.id)
            
            # Add existing pipeline collaborators (backlog)
            active_collab_ids.update(uav._backlog_tx_uav_uav_as_requester.keys())
            active_collab_ids.update(uav._backlog_rx_uav_uav_as_requester.keys())
            # 新增本时隙刚产生的协作比特背景负荷
            active_collab_ids.update(uav._bits_tx_uav_uav_as_requester.keys())
            active_collab_ids.update(uav._bits_rx_uav_uav_as_requester.keys())
            
            for collab_id in active_collab_ids:
                if 0 <= collab_id < len(self._uavs):
                    self._uavs[collab_id]._requesting_uav_ids.add(uav.id)

        # Count MBS contention
        mbs_users = set()
        for uav in self._uavs:
            # 这里的指标包括了在 process_requests 期间注入的新任务
            if (uav._backlog_tx_uav_mbs > config.EPSILON or 
                uav._backlog_rx_uav_mbs > config.EPSILON or 
                uav._bits_tx_uav_mbs > config.EPSILON or 
                uav._bits_rx_uav_mbs > config.EPSILON or
                uav._mbs_rx_tasks):
                mbs_users.add(uav.id)
        
        num_mbs_users = max(1, len(mbs_users))
        for uav in self._uavs:
            uav._num_mbs_users = num_mbs_users
            
        # 4. Set Rates using new topology and contention stats
        for uav in self._uavs:
            uav._set_rates(self._uavs)

        # 5. Set frequency counts for GDSF caching policy (Must use new coverage/collaborators)
        for uav in self._uavs:
            uav.set_freq_counts()

    def _drop_ue_pipeline_across_uavs(self, ue_id: int, reason: str = "boundary") -> None:
        """从全网所有 UAV 的流水线中移除属于特定 UE 的在途传输任务。"""
        if reason == "boundary":
            self._failed_requests_dropped_boundary += 1
        else:
            self._failed_requests_dropped_timeout += 1

        for uav in self._uavs:
            # 1. 清理 MBS 接收队列
            orig_mbs = uav._mbs_rx_tasks
            uav._mbs_rx_tasks = [t for t in orig_mbs if t["ue_id"] != ue_id]
            m_rem = sum(t["bits"] for t in orig_mbs if t["ue_id"] == ue_id)
            uav._backlog_rx_uav_mbs = max(0.0, uav._backlog_rx_uav_mbs - m_rem)
            
            # 2. 清理 U2U 接收队列 (作为请求方)
            for nid in list(uav._u2u_rx_tasks.keys()):
                orig_u2u_rx = uav._u2u_rx_tasks[nid]
                uav._u2u_rx_tasks[nid] = [t for t in orig_u2u_rx if t["ue_id"] != ue_id]
                u_rem = sum(t["bits"] for t in orig_u2u_rx if t["ue_id"] == ue_id)
                uav._backlog_rx_uav_uav_as_requester[nid] = max(0.0, uav._backlog_rx_uav_uav_as_requester.get(nid, 0.0) - u_rem)
                m_rem += u_rem
                
            # 3. 清理 U2U 协作发送队列 (作为协作者)
            for rid in list(uav._u2u_tx_tasks_as_collab.keys()):
                orig_u2u_tx = uav._u2u_tx_tasks_as_collab[rid]
                uav._u2u_tx_tasks_as_collab[rid] = [t for t in orig_u2u_tx if t["ue_id"] != ue_id]
                t_rem = sum(t["bits"] for t in orig_u2u_tx if t["ue_id"] == ue_id)
                uav._backlog_tx_uav_uav_as_collaborator[rid] = max(0.0, uav._backlog_tx_uav_uav_as_collaborator.get(rid, 0.0) - t_rem)
                m_rem += t_rem

            # 统计总丢弃比特
            if reason == "boundary":
                self._pending_bits_dropped_last_boundary += m_rem
            else:
                self._pending_bits_dropped_timeout += m_rem

            # 4. 清理下行队列
            b_rem = 0.0
            if ue_id in uav._backlog_tx_ue_uav:
                b_rem += uav._backlog_tx_ue_uav[ue_id]
                del uav._backlog_tx_ue_uav[ue_id]
            
            if reason == "boundary":
                self._backlog_bits_dropped_last_boundary += b_rem
            else:
                self._backlog_bits_dropped_timeout += b_rem

            if ue_id in uav._dl_tasks:
                del uav._dl_tasks[ue_id]
            # (UE 上行积压通常不涉及流水线中转，简单清理即可)
            if ue_id in uav._backlog_rx_ue_uav:
                del uav._backlog_rx_ue_uav[ue_id]
            
            # 5. 清理等待协作队列 (Case 3)
            # 严格流水线逻辑：仅清理元数据，比特计费由协作者侧的队列负责，避免双计
            for nid in list(uav._u2u_wait_tasks.keys()):
                wait_list = uav._u2u_wait_tasks[nid]
                # 过滤并移除
                uav._u2u_wait_tasks[nid] = [wt for wt in wait_list if wt[1] != ue_id]

    def _handle_request_timeouts(self) -> None:
        """UE-side timeout: if a request waits too long, fail it and clear any in-flight pipeline."""
        for ue in self._ues:
            if not ue.request_active:
                continue
            age_slots = max(0, self._time_step - ue.request_start_step + 1)
            if age_slots >= config.UE_MAX_WAIT_TIME:
                ue.on_request_failed(self._time_step)
                self._drop_ue_pipeline_across_uavs(ue.id, reason="timeout")

    def _get_obs(self) -> list[np.ndarray]:
        """Construct local observation vector for each UAV agent.

        新观测结构（支持注意力机制）：
        - Own state: normalized position (3) + cache bitmap (NUM_FILES)
        - Neighbors (MAX_UAV_NEIGHBORS): features (25) + count (1)
        - Associated UEs (MAX_ASSOCIATED_UES): features (5) + count (1)

        关键改进：不再截断 UE 列表，包含所有关联的 UE，通过 count 字段生成 mask
        """
        all_obs: list[np.ndarray] = []

        # Normalization constants for 3D positions
        pos_norm = np.array([config.AREA_WIDTH, config.AREA_HEIGHT, config.UE_MAX_ALT])

        for uav in self._uavs:
            # Part 1: Own state (position and cache status)
            own_pos: np.ndarray = uav.pos / pos_norm
            own_cache: np.ndarray = uav.cache.astype(np.float32)
            own_state: np.ndarray = np.concatenate([own_pos, own_cache])

            # Part 2: Neighbors state
            neighbor_states: np.ndarray = np.zeros((config.MAX_UAV_NEIGHBORS, config.NEIGHBOR_STATE_DIM))
            neighbors: list[UAV] = sorted(uav.neighbors, key=lambda n: float(np.linalg.norm(uav.pos - n.pos)))[:config.MAX_UAV_NEIGHBORS]

            # Pre-calculate current requested files
            my_requested_files = set()
            for ue in uav.current_covered_ues:
                req_type, _, req_id = ue.current_request
                if req_type == 1:
                    my_requested_files.add(req_id)

            for i, neighbor in enumerate(neighbors):
                relative_pos: np.ndarray = (neighbor.pos - uav.pos) / config.UAV_SENSING_RANGE
                # 原始 cache bitmap（让注意力机制学习）
                neighbor_cache: np.ndarray = neighbor.cache.astype(np.float32)
                # 预处理特征（编码领域知识）
                immediate_help = 0.0
                for file_id in my_requested_files:
                    if neighbor.cache[file_id]:
                        immediate_help = 1.0
                        break
                intersection = np.sum(np.logical_and(uav.cache, neighbor.cache))
                union = np.sum(np.logical_or(uav.cache, neighbor.cache))
                similarity = intersection / (union + config.EPSILON)
                complementarity = 1.0 - similarity
                # 混合特征: pos(3) + cache(NUM_FILES) + immediate_help(1) + complementarity(1)
                neighbor_states[i, :] = np.concatenate([
                    relative_pos,
                    neighbor_cache,
                    np.array([immediate_help, complementarity], dtype=np.float32)
                ])

            # 邻居数量（用于生成 mask）
            neighbor_count = np.array([len(neighbors)], dtype=np.float32)

            # Part 3: State of ALL associated UEs (不再截断！)
            ue_states: np.ndarray = np.zeros((config.MAX_ASSOCIATED_UES, config.UE_STATE_DIM))
            # 按距离排序所有关联的 UE
            all_ues: list[UE] = sorted(uav.current_covered_ues, key=lambda u: float(np.linalg.norm(uav.pos - u.pos)))
            actual_ue_count = min(len(all_ues), config.MAX_ASSOCIATED_UES)

            for i in range(actual_ue_count):
                ue = all_ues[i]
                delta_pos: np.ndarray = (ue.pos - uav.pos) / config.UAV_COVERAGE_RADIUS
                _, _, req_id = ue.current_request
                norm_file_id: float = req_id / config.NUM_FILES
                cache_hit: float = 1.0 if uav.cache[req_id] else 0.0
                ue_states[i, :] = np.array([delta_pos[0], delta_pos[1], delta_pos[2], norm_file_id, cache_hit], dtype=np.float32)

            # UE 数量（用于生成 mask，仅注意力模式需要）
            ue_count = np.array([actual_ue_count], dtype=np.float32)

            # Part 4: Combine all parts
            # 注意力模式: [own_state, neighbor_states, neighbor_count, ue_states, ue_count]
            # MLP模式: [own_state, neighbor_states, ue_states]
            if config.USE_ATTENTION:
                obs: np.ndarray = np.concatenate([
                    own_state,
                    neighbor_states.flatten(),
                    neighbor_count,
                    ue_states.flatten(),
                    ue_count
                ])
            else:
                obs = np.concatenate([
                    own_state,
                    neighbor_states.flatten(),
                    ue_states.flatten()
                ])
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
        """Calculate downlink co-channel interference for UEs from non-serving UAVs."""
        from environment import comm_model as comms
        
        # 预计算每个 UAV 的位置和波束方向
        uav_info: list[tuple[np.ndarray, tuple[float, float]]] = []
        for uav in self._uavs:
            beam_dir = uav.get_final_beam_direction()
            uav_info.append((uav.pos, beam_dir))
        
        # 下行干扰 (UAV -> UE)
        for serving_uav_idx, uav in enumerate(self._uavs):
            for ue in uav.current_covered_ues:
                total_dl_interference: float = 0.0
                for interferer_idx, (interferer_pos, interferer_beam) in enumerate(uav_info):
                    if interferer_idx == serving_uav_idx:
                        continue
                    if len(self._uavs[interferer_idx].current_covered_ues) == 0:
                        continue
                    total_dl_interference += comms.calculate_interference_power(
                        interferer_pos, ue.pos, interferer_beam)
                ue.interference_power = total_dl_interference

    def _get_rewards_and_metrics(self) -> tuple[list[float], tuple[float, float, float, float, dict]]:
        """Returns the reward and other metrics (latency, energy, jfi, total_rate, normalizer_stats).
        
        使用动态归一化平衡各奖励分量：
        1. 先对原始指标取 log（压缩量级差异）
        2. 使用 RunningNormalizer 动态归一化到 ~N(0,1)
        3. 各分量乘以权重后相加
        """
        # Latency metric: completed request uses its completion latency; otherwise use waiting time.
        total_latency: float = 0.0
        for ue in self._ues:
            if ue.completed_this_step or ue.failed_this_step:
                total_latency += ue.latency_current_request
            elif ue.request_active:
                age_slots = max(0, self._time_step - ue.request_start_step + 1)
                total_latency += age_slots * config.TIME_SLOT_DURATION
            else:
                total_latency += 0.0
        total_energy: float = sum(uav.energy for uav in self._uavs)
        # 吞吐量指标：从物理层带宽总和改为本时隙各链路实际成功交付的总速率 (Throughput)
        total_rate: float = sum(uav.actual_throughput for uav in self._uavs)
        
        sc_metrics: np.ndarray = np.array([ue.service_coverage for ue in self._ues])
        jfi: float = 0.0
        if sc_metrics.size > 0 and np.sum(sc_metrics**2) > 0:
            jfi = (np.sum(sc_metrics) ** 2) / (sc_metrics.size * np.sum(sc_metrics**2))

        # 动态归一化：latency/energy/rate 取 log 压缩量级
        log_latency = np.log(total_latency + config.EPSILON)
        log_energy = np.log(total_energy + config.EPSILON)
        log_rate = np.log(total_rate + config.EPSILON)
        
        # 缓存归一化前的统计量（用于诊断）
        latency_mean_used = self._latency_normalizer.mean
        latency_var_used = self._latency_normalizer.var
        energy_mean_used = self._energy_normalizer.mean
        energy_var_used = self._energy_normalizer.var
        rate_mean_used = self._rate_normalizer.mean
        rate_var_used = self._rate_normalizer.var
        
        # 执行归一化（会更新统计量）
        r_latency: float = config.ALPHA_1 * self._latency_normalizer.normalize(log_latency)
        r_energy: float = config.ALPHA_2 * self._energy_normalizer.normalize(log_energy)
        # JFI 使用固定映射：以 config.JFI_CENTER 为中心，线性范围对应映射到 [-2, +2]
        r_fairness: float = config.ALPHA_3 * np.clip((jfi - config.JFI_CENTER) * config.JFI_SCALE, -2.0, 2.0)
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
        
        # Collect normalizer states and reward components for debugging
        completed_requests = sum(1 for ue in self._ues if ue.completed_this_step)
        failed_requests = sum(1 for ue in self._ues if ue.failed_this_step)
        normalizer_stats = {
            # 原始指标（step级别，支持平均）
            "total_latency": total_latency,
            "total_energy": total_energy,
            "total_rate": total_rate,
            "mbs_users_count": self._uavs[0]._num_mbs_users, # 统计 MBS 链路竞争激烈程度
            # B1 丢包建模诊断：上一个时隙边界因 UE 离开覆盖而丢弃的比特量
            "pending_bits_dropped_boundary": self._pending_bits_dropped_last_boundary,
            "backlog_bits_dropped_boundary": self._backlog_bits_dropped_last_boundary,
            "failed_requests_dropped_boundary": self._failed_requests_dropped_boundary,
            # 超时诊断：因等待时间过长而丢弃的量
            "pending_bits_dropped_timeout": self._pending_bits_dropped_timeout,
            "backlog_bits_dropped_timeout": self._backlog_bits_dropped_timeout,
            "failed_requests_dropped_timeout": self._failed_requests_dropped_timeout,
            # 任务层诊断：本时隙请求完成/失败数量
            "completed_requests": completed_requests,
            "failed_requests": failed_requests,
            # 归一化时使用的统计量（*_used 后缀）
            "latency_norm_mean_used": latency_mean_used,
            "latency_norm_var_used": latency_var_used,
            "energy_norm_mean_used": energy_mean_used,
            "energy_norm_var_used": energy_var_used,
            "rate_norm_mean_used": rate_mean_used,
            "rate_norm_var_used": rate_var_used,
            # 归一化后的当前统计量
            "latency_norm_mean": self._latency_normalizer.mean,
            "latency_norm_var": self._latency_normalizer.var,
            "energy_norm_mean": self._energy_normalizer.mean,
            "energy_norm_var": self._energy_normalizer.var,
            "rate_norm_mean": self._rate_normalizer.mean,
            "rate_norm_var": self._rate_normalizer.var,
            # Individual reward components (critical for multi-objective tuning)
            "r_latency": r_latency,
            "r_energy": r_energy,
            "r_fairness": r_fairness,
            "r_rate": r_rate,
            # Log-transformed values (before normalization, for normalizer diagnosis)
            "log_latency": log_latency,
            "log_energy": log_energy,
            "log_rate": log_rate,
        }
        
        return rewards, (total_latency, total_energy, jfi, total_rate, normalizer_stats)
