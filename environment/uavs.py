from __future__ import annotations
from environment.user_equipments import UE
from environment import comm_model as comms
import config
import numpy as np
# 模拟无人机（UAV）作为空中基站提供通信保障服务。它管理 UAV 的位置、缓存、通信、能量消耗和内容请求处理。


def _try_add_file_to_cache(uav: UAV, file_id: int) -> None:
    """Try to add a file to UAV cache if there's enough space."""
    if uav._working_cache[file_id]:
        return  # 文件已在缓存中，无需重复添加
    used_space: int = np.sum(uav._working_cache * config.FILE_SIZES)
    if used_space + config.FILE_SIZES[file_id] <= config.UAV_STORAGE_CAPACITY[uav.id]:
        uav._working_cache[file_id] = True


class UAV:
    def __init__(self, uav_id: int) -> None:
        self.id: int = uav_id
        self.pos: np.ndarray = np.array([
            np.random.uniform(0, config.AREA_WIDTH),
            np.random.uniform(0, config.AREA_HEIGHT),
            np.random.uniform(config.UAV_MIN_ALT, config.UAV_MAX_ALT)
        ])

        self._dist_moved: float = 0.0  # Distance moved in the current time slot
        self._current_covered_ues: list[UE] = []
        self._neighbors: list[UAV] = []
        self._current_collaborator: UAV | None = None
        self._energy_current_slot: float = 0.0  # Energy consumed for this time slot
        self.collision_violation: bool = False  # Track if UAV has violated minimum separation
        self.boundary_violation: bool = False  # Track if UAV has gone out of bounds

        # Cache and request tracking
        self._current_requested_files: np.ndarray = np.zeros(config.NUM_FILES, dtype=bool)
        self.cache: np.ndarray = np.zeros(config.NUM_FILES, dtype=bool)
        self._working_cache: np.ndarray = np.zeros(config.NUM_FILES, dtype=bool)
        self._freq_counts = np.zeros(config.NUM_FILES)  # For GDSF caching policy
        self._ema_scores = np.zeros(config.NUM_FILES)

        # Communication rates
        self._uav_uav_rate: float = 0.0
        self._uav_mbs_uplink_rate: float = 0.0   # UAV → MBS (发送请求)
        self._uav_mbs_downlink_rate: float = 0.0 # MBS → UAV (接收数据)
        
        # 被哪些 UAV 选为协作者（用于 UAV-UAV 带宽分配）
        self._requesting_uav_ids: set[int] = set()
        
        # 实时负载记录（由 Env 统一计算并注入）
        self._num_mbs_users: int = 1
        
        # Communication time/bits tracking for energy calculation (分为发送和接收)
        # 记录本时隙新产生的传输任务量（bits）
        # UE ↔ UAV 链路
        self._bits_tx_ue_uav: dict[int, float] = {}      # ue_id -> bits (UAV→UE 下行数据)
        self._bits_rx_ue_uav: dict[int, float] = {}      # ue_id -> bits (UE→UAV 上行请求)
        # UAV ↔ UAV 链路（作为请求方时）
        self._bits_tx_uav_uav_as_requester: dict[int, float] = {}  # collaborator_id -> bits
        self._bits_rx_uav_uav_as_requester: dict[int, float] = {}  # collaborator_id -> bits
        # UAV ↔ UAV 链路（作为协作者时）
        self._bits_tx_uav_uav_as_collaborator: dict[int, float] = {}  # requester_id -> bits
        self._bits_rx_uav_uav_as_collaborator: dict[int, float] = {}  # requester_id -> bits
        # UAV ↔ MBS 链路
        self._bits_tx_uav_mbs: float = 0.0     # bits (UAV→MBS 上行请求)
        self._bits_rx_uav_mbs: float = 0.0     # bits (MBS→UAV 下行数据)
        
        # 跨时隙积压比特（上一时隙未完成的传输）
        # 这些变量跨时隙保持，只在 episode reset 时清零
        self._backlog_tx_ue_uav: dict[int, float] = {}   # ue_id -> bits
        self._backlog_rx_ue_uav: dict[int, float] = {}   # ue_id -> bits
        self._backlog_tx_uav_uav_as_requester: dict[int, float] = {}  # collaborator_id -> bits
        self._backlog_rx_uav_uav_as_requester: dict[int, float] = {}  # collaborator_id -> bits
        self._backlog_tx_uav_uav_as_collaborator: dict[int, float] = {} # requester_id -> bits
        self._backlog_rx_uav_uav_as_collaborator: dict[int, float] = {} # requester_id -> bits
        self._backlog_tx_uav_mbs: float = 0.0  # bits
        self._backlog_rx_uav_mbs: float = 0.0  # bits
        
        # 3D Beamforming: 波束指向 (俯仰角, 方位角)
        self._beam_direction: tuple[float, float] = (0.0, 0.0)  # 基准方向（指向UE质心）
        self._beam_offset: tuple[float, float] = (0.0, 0.0)     # 智能体控制的偏移量
        
        # 延迟缓存机制
        self._pending_cache: list[tuple[int, int]] = []  # 已弃用，保留结构兼容
        self._pending_downlinks: list[tuple[int, int, float, int]] = []  # 已弃用，保留结构兼容

        # === 严格流水线任务队列 (Strict Pipelining) ===
        # 1. MBS -> UAV 接收队列: [(fid, remaining_bits, target_type, target_id), ...]
        #    target_type: "CACHE" (存入本地), "UE" (转发给UE), "COL" (转发给Requester UAV)
        self._mbs_rx_tasks: list[dict] = []
        
        # 2. Neighbor -> UAV 接收队列 (作为请求方): neighbor_id -> [(fid, bits, target_type, target_id), ...]
        self._u2u_rx_tasks: dict[int, list[dict]] = {}
        
        # 3. UAV -> Neighbor 发送队列 (作为协作者): requester_id -> [(fid, bits, ue_id), ...]
        self._u2u_tx_tasks_as_collab: dict[int, list[dict]] = {}

        # 4. UAV -> UE 下行队列 (最终分发): ue_id -> [(fid, bits), ...]
        self._dl_tasks: dict[int, list[tuple[int, float]]] = {}

        # 5. 等待队列 (Wait for Neighbor): neighbor_id -> [(fid, ue_id), ...]
        #    用于 Case 3 中请求方等待协作者从 MBS 取回数据。
        self._u2u_wait_tasks: dict[int, list[tuple[int, int]]] = {}

        # 传输速率记录（用于奖励计算和积压比特扣除）
        self._total_downlink_rate: float = 0.0      # 本时隙总下行链路带宽之和
        self._actual_processed_bits_dl: float = 0.0 # 本时隙实际成功处理的下行比特总量
        self._ue_rates_dl: dict[int, float] = {}  # ue_id -> dl_rate (本时隙实时值)
        self._ue_rates_ul: dict[int, float] = {}  # ue_id -> ul_rate
        self._neighbor_rates: dict[int, float] = {} # uav_id -> rate (本时隙针对活跃链路的计算值)

    def _get_pending_cache_space(self) -> float:
        """计算当前正在流水线中传输且目的地是本地缓存的文件空间总和。"""
        s = sum(config.FILE_SIZES[t["fid"]] for t in self._mbs_rx_tasks if t["target_type"] == "CACHE")
        for nid in self._u2u_rx_tasks:
            s += sum(config.FILE_SIZES[t["fid"]] for t in self._u2u_rx_tasks[nid] if t["target_type"] == "CACHE")
        return s

    def _is_ue_in_flight(self, ue_id: int) -> bool:
        """检查特定 UE 是否已有任务在流水线中。"""
        if ue_id in self._dl_tasks: return True
        if any(t["ue_id"] == ue_id for t in self._mbs_rx_tasks): return True
        for tasks in self._u2u_rx_tasks.values():
            if any(t["ue_id"] == ue_id for t in tasks): return True
        # 检查是否正在等待协作者回源 (Case 3)
        for tasks in self._u2u_wait_tasks.values():
            if any(tid == ue_id for _, tid in tasks): return True
        return False

    @property
    def energy(self) -> float:
        return self._energy_current_slot

    @property
    def current_covered_ues(self) -> list[UE]:
        return self._current_covered_ues

    @property
    def neighbors(self) -> list[UAV]:
        return self._neighbors

    @property
    def total_downlink_rate(self) -> float:
        return self._total_downlink_rate

    @property
    def actual_throughput(self) -> float:
        """Returns the actual bits processed in the current slot divided by slot duration."""
        return self._actual_processed_bits_dl / (config.TIME_SLOT_DURATION + config.EPSILON)

    @property
    def current_collaborator(self) -> UAV | None:
        return self._current_collaborator

    def reset_for_next_step(self) -> None:
        """Reset UAV state for a new time slot.
        
        注意：_backlog_* 变量不在这里重置，因为它们是跨时隙的。
        它们在 episode 开始时通过 UAV.__init__() 初始化为 0。
        """
        self._current_covered_ues = []
        self._neighbors = []
        self._current_collaborator = None
        self._current_requested_files = np.zeros(config.NUM_FILES, dtype=bool)
        self._freq_counts = np.zeros(config.NUM_FILES)
        self._energy_current_slot = 0.0
        # 本时隙新增的通信比特量（每时隙重置，分为发送和接收）
        self._bits_tx_ue_uav = {}
        self._bits_rx_ue_uav = {}
        # UAV-UAV 作为请求方（按协作者ID分组）
        self._bits_tx_uav_uav_as_requester = {}
        self._bits_rx_uav_uav_as_requester = {}
        # UAV-UAV 作为协作者（FDM并行，按请求方ID分组）
        self._bits_tx_uav_uav_as_collaborator = {}
        self._bits_rx_uav_uav_as_collaborator = {}
        self._bits_tx_uav_mbs = 0.0
        self._bits_rx_uav_mbs = 0.0
        # 注意：_backlog_bits_* 不重置，跨时隙保持
        self._requesting_uav_ids = set()  # 重置协作者请求集合
        self._beam_offset = (0.0, 0.0)  # 重置波束偏移
        self._total_downlink_rate = 0.0  # 重置传输速率记录
        self._actual_processed_bits_dl = 0.0
        self._ue_rates_dl = {}
        self._ue_rates_ul = {}
        self._neighbor_rates = {}
        self.collision_violation = False
        self.boundary_violation = False

    def update_position(self, next_pos: np.ndarray) -> None:
        """Update the UAV's position to the new 3D location chosen by the MARL agent."""
        # next_pos is now a full 3D position [x, y, z]
        self._dist_moved = float(np.linalg.norm(next_pos - self.pos))
        self.pos = next_pos

    def set_neighbors(self, all_uavs: list[UAV]) -> None:
        """Set neighboring UAVs within sensing range for this UAV."""
        self._neighbors = []
        for other_uav in all_uavs:
            if other_uav.id != self.id:
                distance = float(np.linalg.norm(self.pos - other_uav.pos))
                if distance <= config.UAV_SENSING_RANGE:
                    self._neighbors.append(other_uav)

    def set_current_requested_files(self) -> None:
        """Update the current requested files and beam direction based on covered UEs."""
        for ue in self._current_covered_ues:
            if ue.current_request:
                _, _, req_id = ue.current_request
                self._current_requested_files[req_id] = True
        
        # 更新波束基准方向（指向关联UE的质心）
        # 需要计算基准方向的情况：
        # 1. offset 模式（智能体基于此进行调整）
        # 2. 智能体控制被禁用（完全依赖规则，即指向质心）
        if config.BEAM_CONTROL_MODE == "offset" or not config.BEAM_CONTROL_ENABLED:
            ue_positions = [ue.pos for ue in self._current_covered_ues]
            self._beam_direction = comms.calculate_beam_direction(self.pos, ue_positions)

    def set_beam_offset(self, delta_theta: float, delta_phi: float) -> None:
        """Set beam offset from agent's action (offset mode)."""
        self._beam_offset = (delta_theta, delta_phi)

    def set_beam_absolute(self, theta: float, phi: float) -> None:
        """Set beam direction directly from agent's action (absolute mode)."""
        # 在absolute模式下，直接覆盖基准方向，偏移为0
        self._beam_direction = (theta, phi)
        self._beam_offset = (0.0, 0.0)

    def get_final_beam_direction(self) -> tuple[float, float]:
        """Get final beam direction combining base direction and offset.
        
        球坐标系：
        - theta: [0°, 180°]，0°=天顶，90°=水平，180°=天底
        - phi: [-180°, 180°]，方位角
        """
        base_theta, base_phi = self._beam_direction
        delta_theta, delta_phi = self._beam_offset
        
        # 计算最终角度（theta 范围扩展到 [0, 180]）
        final_theta = np.clip(base_theta + delta_theta, 0.0, 180.0)
        # 方位角周期性处理 [-180, 180]
        final_phi = ((base_phi + delta_phi + 180.0) % 360.0) - 180.0
        
        return (final_theta, final_phi)

    def select_collaborator(self) -> None:
        """Choose a single collaborating UAV from its list of neighbours.
        
        注意：_set_rates() 已移至 env.py 中统一调用，
        因为需要先统计所有 UAV 的 _requesting_uav_ids。
        """
        if not self._neighbors:
            return

        missing_requested_files: np.ndarray = self._current_requested_files & (~self.cache)
        has_u2u_backlog = (len(self._backlog_tx_uav_uav_as_requester) > 0 or 
                          len(self._backlog_rx_uav_uav_as_requester) > 0)
        if not np.any(missing_requested_files) and not has_u2u_backlog:
            return  # 如果没有任何确实的文件，且没有积压任务，不占用协作带宽

        best_collaborators: list[UAV] = []
        max_missing_overlap: int = -1

        # Find neighbors with maximum overlap
        for neighbor in self._neighbors:
            overlap: int = int(np.sum(missing_requested_files & neighbor.cache))
            if overlap > max_missing_overlap:
                max_missing_overlap = overlap
                best_collaborators = [neighbor]
            elif overlap == max_missing_overlap:
                best_collaborators.append(neighbor)

        # If only one best collaborator, select it
        if len(best_collaborators) == 1:
            self._current_collaborator = best_collaborators[0]
            return

        # If tie in overlap, select closest one(s)
        min_distance: float = float("inf")
        closest_collaborators: list[UAV] = []

        for collaborator in best_collaborators:
            distance: float = float(np.linalg.norm(self.pos - collaborator.pos))

            if distance < min_distance:
                min_distance = distance
                closest_collaborators = [collaborator]
            elif distance == min_distance:
                closest_collaborators.append(collaborator)

        # If still tied, select randomly
        if len(closest_collaborators) == 1:
            self._current_collaborator = closest_collaborators[0]
        else:
            self._current_collaborator = closest_collaborators[np.random.randint(0, len(closest_collaborators))]

    def set_freq_counts(self) -> None:
        """Set the request count for current slot based on cache availability."""
        for ue in self._current_covered_ues:
            _, _, req_id = ue.current_request
            self._freq_counts[req_id] += 1
            if not self.cache[req_id] and self._current_collaborator:
                self._current_collaborator._freq_counts[req_id] += 1

    def init_working_cache(self) -> None:
        """Initialize working cache from current cache state.
        
        必须在所有 UAV 的 process_requests() 之前统一调用，
        避免协作缓存更新时的竞态条件。
        """
        self._working_cache = self.cache.copy()

    def process_requests(self, current_time_step: int) -> None:
        """Process content requests from UEs covered by this UAV."""
        final_beam = self.get_final_beam_direction()
        
        # 性能优化：预计算空间占用，避免在 UE 循环中重复求和
        used_space = np.sum(self._working_cache * config.FILE_SIZES)
        pending_space = self._get_pending_cache_space()
        
        # 预计算协作者空间（如果有）
        collab_info = None
        collab_pending = {}
        if self._current_collaborator:
            collab = self._current_collaborator
            c_used = np.sum(collab._working_cache * config.FILE_SIZES)
            c_pending = collab._get_pending_cache_space()
            collab_info = (collab, c_used, c_pending)
            # 设置协作缓存标记字典 (fid -> dummy_step)
            collab_pending = {t["fid"]: current_time_step for t in collab._mbs_rx_tasks if t["target_type"] == "CACHE"}

        # 性能优化：历史待办状态
        historical_pending = {t["fid"]: current_time_step for t in self._mbs_rx_tasks if t["target_type"] == "CACHE"}
        
        # 本时隙内正在处理的请求文件（防止同一时隙多个 UE 请求同一文件导致外部带宽爆炸）
        fetching_this_slot: dict[int, int] = {} # fid -> dummy_step

        for ue in self._current_covered_ues:
            if not getattr(ue, "request_active", True):
                continue
            
            # 严格流水线检查：如果该 UE 已经有任务在在途传输中，不再重复注入
            if self._is_ue_in_flight(ue.id):
                continue

            channel_gain = comms.calculate_channel_gain(ue.pos, self.pos, final_beam)
            num_ues = len(self._current_covered_ues)
            # 下行速率：UAV → UE
            downlink_rate = comms.calculate_ue_uav_rate(channel_gain, num_ues, ue.interference_power)
            # 上行速率：UE → UAV (仅 SNR 受限)
            uplink_rate = comms.calculate_ue_uav_uplink_rate(channel_gain, num_ues)
            
            self._total_downlink_rate += downlink_rate
            
            # 记录当前时隙针对该 UE 的速率
            self._ue_rates_dl[ue.id] = downlink_rate
            self._ue_rates_ul[ue.id] = uplink_rate
            
            # 处理请求
            p_inc, c_p_inc = self._process_content_request(ue, downlink_rate, uplink_rate, 
                                                                 current_time_step, used_space, pending_space, 
                                                                 fetching_this_slot, collab_info,
                                                                 historical_pending, collab_pending)
            pending_space += p_inc
            if collab_info:
                # 更新协作者的 pending 空间，防止后续 UE 请求同一文件时重复累加
                collab_info = (collab_info[0], collab_info[1], collab_info[2] + c_p_inc)

    def _get_active_u2u_neighbor_ids(self) -> set[int]:
        """获取当前具有活跃 U2U 通信链路（包含积压和新任务）的邻居 ID 集合。"""
        active_ids = set()
        
        # 1. 角色：请求方 (Requester)
        # 包括历史积压和本时隙新增比特
        active_ids.update(self._backlog_tx_uav_uav_as_requester.keys())
        active_ids.update(self._backlog_rx_uav_uav_as_requester.keys())
        active_ids.update(self._bits_tx_uav_uav_as_requester.keys())
        active_ids.update(self._bits_rx_uav_uav_as_requester.keys())
        # 当前决策选中的协作者（可能尚未产生比特，但已分配带宽）
        if self._current_collaborator:
            active_ids.add(self._current_collaborator.id)
            
        # 2. 角色：协作者 (Collaborator)
        # 包括由 Env 注入的请求者 ID 集合
        active_ids.update(self._requesting_uav_ids)
        # 以及历史积压任务中记录的请求者
        active_ids.update(self._backlog_tx_uav_uav_as_collaborator.keys())
        active_ids.update(self._backlog_rx_uav_uav_as_collaborator.keys())
        active_ids.update(self._bits_tx_uav_uav_as_collaborator.keys())
        active_ids.update(self._bits_rx_uav_uav_as_collaborator.keys())
        
        return active_ids

    def _calculate_link_load(self) -> int:
        """计算本 UAV 当前参与的 FDM 并行链路总数。

        FDM 下，所有并行邻居共享总带宽。链路数即为去重后的活跃邻居数量。
        """
        active_neighbors = self._get_active_u2u_neighbor_ids()
        return max(1, len(active_neighbors))

    def _get_u2u_rate_with(self, other_uav: UAV) -> float:
        """计算本 UAV 与另一 UAV 之间的当前物理速率（考虑双端负载和干扰）。

        警告：此方法在 _set_rates 阶段调用，使用的是该时隙开始时的负载状态。
        确保在调用 _set_rates 之后、process_requests 之前，负载状态已固定。
        如果需要在 process_requests 期间使用速率，请使用缓存的 _neighbor_rates。
        """
        gain = comms.calculate_channel_gain(self.pos, other_uav.pos)
        load1 = self._calculate_link_load()
        load2 = other_uav._calculate_link_load()
        eff_load = max(load1, load2)
        eff_load = np.clip(eff_load, 1.0, float(config.NUM_UAVS))
        return comms.calculate_uav_uav_rate(gain, int(eff_load))

    def _set_rates(self, all_uavs: list[UAV]) -> None:
        """Update current communication rates for the next cycle.
        
        Since backlogs are stored in bits, we no longer need to rescale time-based backlogs.
        Rates updated here will be used in the next process_requests and energy_consumption calls.
        """
        # 1. UAV-MBS Rates
        mbs_channel_gain = comms.calculate_channel_gain(self.pos, config.MBS_POS)
        self._uav_mbs_uplink_rate = comms.calculate_uav_mbs_uplink_rate(mbs_channel_gain, self._num_mbs_users)
        self._uav_mbs_downlink_rate = comms.calculate_uav_mbs_downlink_rate(mbs_channel_gain, self._num_mbs_users)
        
        # 2. UAV-UAV Rate (针对当前决策选中的协作者，用于 process_requests 文献分发时延计算)
        if self._current_collaborator:
            self._uav_uav_rate = self._get_u2u_rate_with(self._current_collaborator)
        else:
            self._uav_uav_rate = 0.0
            
        # 3. Cache neighbor rates only for active U2U links (performance)
        active_ids = self._get_active_u2u_neighbor_ids()
        self._neighbor_rates = {uid: self._get_u2u_rate_with(all_uavs[uid]) for uid in active_ids if uid != self.id}

    def _add_mbs_rx_task(self, fid: int, bits: float, target_type: str, target_id: int, ue_id: int):
        """添加从 MBS 接收比特的任务，并计入实时负载。"""
        self._mbs_rx_tasks.append({
            "fid": fid, "bits": bits, "target_type": target_type, 
            "target_id": target_id, "ue_id": ue_id
        })
        self._bits_rx_uav_mbs += bits

    def _add_u2u_rx_task(self, neighbor_id: int, fid: int, bits: float, target_type: str, target_id: int, ue_id: int):
        """添加从邻居 UAV 接收比特的任务，并计入实时负载（请求方侧）。"""
        self._u2u_rx_tasks.setdefault(neighbor_id, []).append({
            "fid": fid, "bits": bits, "target_type": target_type, 
            "target_id": target_id, "ue_id": ue_id
        })
        self._bits_rx_uav_uav_as_requester[neighbor_id] = self._bits_rx_uav_uav_as_requester.get(neighbor_id, 0.0) + bits

    def _add_dl_task(self, ue_id: int, fid: int, bits: float):
        """直接添加下行到 UE 的任务（即比特已到达本 UAV）。"""
        self._dl_tasks.setdefault(ue_id, []).append((fid, bits))
        self._backlog_tx_ue_uav[ue_id] = self._backlog_tx_ue_uav.get(ue_id, 0.0) + bits

    def _process_content_request(self, ue: UE, ue_uav_downlink_rate: float, ue_uav_uplink_rate: float, current_time_step: int, 
                                 used_space: float, pending_space: float, fetching_this_slot: dict[int, int],
                                 collab_info: tuple | None, historical_pending: dict[int, int],
                                 collab_pending: dict[int, int]) -> tuple[float, float]:
        """Process a content request from a UE using strict pipelining.
        
        Returns:
            (pending_space_inc, collab_pending_space_inc)
        """
        _, _, req_id = ue.current_request
        file_size_bits = config.FILE_SIZES[req_id] * config.BITS_PER_BYTE
        request_size_bits = config.REQUEST_MSG_SIZE * config.BITS_PER_BYTE
        
        # 记录 UE->UAV 上行请求负载
        self._bits_rx_ue_uav[ue.id] = self._bits_rx_ue_uav.get(ue.id, 0.0) + request_size_bits
        
        def is_in_pipeline(fid: int, ue_id: int) -> bool:
            """检查该 UE 的该文件请求是否已在流水线中。"""
            # 1. 检查下行队列
            if any(f == fid for f, _ in self._dl_tasks.get(ue_id, [])): return True
            # 2. 检查 MBS 接收队列
            if any(t["fid"] == fid and t["ue_id"] == ue_id for t in self._mbs_rx_tasks): return True
            # 3. 检查 U2U 接收队列
            for tasks in self._u2u_rx_tasks.values():
                if any(t["fid"] == fid and t["ue_id"] == ue_id for t in tasks): return True
            return False

        def schedule_cache(target_uav: UAV, fid: int, u_space: float, p_space: float) -> float:
            """尝试预定缓存位。"""
            # 严格模式：只有当文件不在缓存且不在任何 pending/fetching 状态时才预定
            is_pending = (fid in historical_pending or fid in fetching_this_slot) if target_uav.id == self.id else (fid in collab_pending)
            if not target_uav.cache[fid] and not is_pending:
                if (u_space + p_space + config.FILE_SIZES[fid] <= config.UAV_STORAGE_CAPACITY[target_uav.id]):
                    # 注：此时不设置 arrival_step，缓存将在比特流到达时自动触发
                    if target_uav.id == self.id:
                        fetching_this_slot[fid] = current_time_step
                    else:
                        collab_pending[fid] = current_time_step
                    return config.FILE_SIZES[fid]
            return 0.0

        # 如果已在流水线中，不重复发起请求
        if is_in_pipeline(req_id, ue.id):
            return 0.0, 0.0

        p_inc, cp_inc = 0.0, 0.0

        if self.cache[req_id]:
            # Case 1: 本地缓存命中 -> 直接进入下行队列
            self._add_dl_task(ue.id, req_id, file_size_bits)
        elif collab_info:
            collab, c_used, c_pending_space = collab_info
            # 记录 U2U 请求负载 (双向基础)
            self._bits_tx_uav_uav_as_requester[collab.id] = self._bits_tx_uav_uav_as_requester.get(collab.id, 0.0) + request_size_bits
            collab._bits_rx_uav_uav_as_collaborator[self.id] = collab._bits_rx_uav_uav_as_collaborator.get(self.id, 0.0) + request_size_bits
            
            if collab.cache[req_id]:
                # Case 2: 协作者缓存命中 -> 协作者发，我收
                self._add_u2u_rx_task(collab.id, req_id, file_size_bits, "UE", ue.id, ue.id)
                collab._u2u_tx_tasks_as_collab.setdefault(self.id, []).append({
                    "fid": req_id, "bits": file_size_bits, "ue_id": ue.id
                })
                collab._bits_tx_uav_uav_as_collaborator[self.id] = collab._bits_tx_uav_uav_as_collaborator.get(self.id, 0.0) + file_size_bits
                p_inc += schedule_cache(self, req_id, used_space, pending_space)
            else:
                # Case 3: 协作者也未命中 -> 协作者从 MBS 取回源
                # 链路: MBS -> Collab (接收后) -> UAV -> UE
                # 1. 协作者发起 MBS 上行请求（注：U2U 请求比特已在进入 collab 分支时统一计费）
                collab._add_mbs_rx_task(req_id, file_size_bits, "COL", self.id, ue.id)
                collab._bits_tx_uav_mbs += request_size_bits
                self._u2u_wait_tasks.setdefault(collab.id, []).append((req_id, ue.id))
                
                # 预定空间
                p_inc += schedule_cache(self, req_id, used_space, pending_space)
                cp_inc += schedule_cache(collab, req_id, c_used, c_pending_space)
        else:
            # Case 4: 直连 MBS
            self._add_mbs_rx_task(req_id, file_size_bits, "UE", ue.id, ue.id)
            self._bits_tx_uav_mbs += request_size_bits
            p_inc += schedule_cache(self, req_id, used_space, pending_space)

        return p_inc, cp_inc

    def update_ema_and_cache(self, current_time_step: int) -> None:
        """Update EMA scores. Cache updates are now handled by strict pipelining in update_energy_consumption."""
        self._ema_scores = config.GDSF_SMOOTHING_FACTOR * self._freq_counts + (1 - config.GDSF_SMOOTHING_FACTOR) * self._ema_scores

    def gdsf_cache_update(self) -> None:
        """使用 GDSF 缓存策略在较长的时间尺度上更新缓存布局。
        
        修复逻辑：确保正在传输（In-flight）以及近期策略内仍需保留的文件受到保护，
        避免在高频更新时出现“刚下载完就被策略驱逐”的现象。
        """
        priority_scores = self._ema_scores / (config.FILE_SIZES + config.EPSILON)
        sorted_file_ids = np.argsort(-priority_scores)
        
        # 获取当前在途并计划存入缓存的文件 ID 集合
        in_flight_fids = set()
        for t in self._mbs_rx_tasks:
            if t["target_type"] == "CACHE": in_flight_fids.add(t["fid"])
        for tasks in self._u2u_rx_tasks.values():
            for t in tasks:
                if t["target_type"] == "CACHE": in_flight_fids.add(t["fid"])

        # 重新规划时必须保留已占用且正在传输的空间
        pending_space = self._get_pending_cache_space()
        used_space = 0.0
        new_working_cache = np.zeros(config.NUM_FILES, dtype=bool)

        # 1. 第一优先级：强制保留所有正在流水线中传输的文件，避免带宽浪费
        for fid in in_flight_fids:
            new_working_cache[fid] = True
            # used_space 不计入 fid 空间，因为它们已被 pending_space 覆盖

        # 2. 第二优先级：按得分填充剩余物理空间
        for file_id in sorted_file_ids:
            if file_id in in_flight_fids:
                continue
                
            file_size = config.FILE_SIZES[file_id]
            if used_space + pending_space + file_size <= config.UAV_STORAGE_CAPACITY[self.id]:
                new_working_cache[file_id] = True
                used_space += file_size
            else:
                break
        
        # 更新工作缓存（指导未来请求决策）
        self._working_cache = new_working_cache
        
        # --- 同步物理缓存 (落盘逻辑整合) ---
        # 1. 驱逐不在新规划中的现有物理文件
        for fid in range(config.NUM_FILES):
            if self.cache[fid] and not self._working_cache[fid]:
                self.cache[fid] = False
        
        # 2. 注入新规划中的预取任务 (非在途且未缓存的文件)
        for fid in range(config.NUM_FILES):
            if self._working_cache[fid] and not self.cache[fid] and fid not in in_flight_fids:
                # 注入 MBS RX 任务转向缓存
                file_size_bits = float(config.FILE_SIZES[fid] * config.BITS_PER_BYTE)
                self._add_mbs_rx_task(fid, file_size_bits, "CACHE", self.id, -1)

    def update_energy_consumption(self, all_uavs: list[UAV], current_time_step: int) -> None:
        """Update UAV energy consumption for the current time slot using strict bitstream pipelining.
        """
        # 1. 飞行能耗
        time_moving = self._dist_moved / (config.UAV_SPEED + config.EPSILON)
        time_moving = float(np.clip(time_moving, 0.0, config.TIME_SLOT_DURATION))
        time_hovering = float(np.clip(config.TIME_SLOT_DURATION - time_moving, 0.0, config.TIME_SLOT_DURATION))
        fly_energy = config.POWER_MOVE * time_moving + config.POWER_HOVER * time_hovering
        
        tau = config.TIME_SLOT_DURATION

        # --- (A) UE-UAV 链路 (OFDMA) ---
        for ue_id, bits in self._bits_rx_ue_uav.items():
            self._backlog_rx_ue_uav[ue_id] = self._backlog_rx_ue_uav.get(ue_id, 0.0) + bits
            
        # OFDMA 上行
        actual_rx_ue_uav = 0.0
        for uid in list(self._backlog_rx_ue_uav.keys()):
            rate = self._ue_rates_ul.get(uid, 0.0)
            if rate <= 0.0: continue
            proc = min(self._backlog_rx_ue_uav[uid], rate * tau)
            self._backlog_rx_ue_uav[uid] -= proc
            actual_rx_ue_uav = max(actual_rx_ue_uav, proc / rate)
            if self._backlog_rx_ue_uav[uid] < 1.0: del self._backlog_rx_ue_uav[uid]

        # OFDMA 下行 (分发给 UE)
        covered_ue_ids = {u.id for u in self._current_covered_ues}
        tx_times_ue = []
        ue_dl_bits = 0.0
        
        for ue_id in list(self._backlog_tx_ue_uav.keys()):
            rate = self._ue_rates_dl.get(ue_id, 0.0)
            if rate <= 0.0 or ue_id not in covered_ue_ids: continue
            
            backlog = self._backlog_tx_ue_uav[ue_id]
            proc = min(backlog, rate * tau)
            if proc <= 0.0: continue

            # 更新 UE 状态
            ue_obj = next((u for u in self._current_covered_ues if u.id == ue_id), None)
            if ue_obj: ue_obj.on_downlink_bits_delivered(proc)

            # 消费下行任务队列
            tasks = self._dl_tasks.get(ue_id, [])
            rem_to_consume = proc
            while rem_to_consume > 0 and tasks:
                fid, task_rem = tasks[0]
                used = min(task_rem, rem_to_consume)
                task_rem -= used
                rem_to_consume -= used
                if task_rem <= 1.0:
                    tasks.pop(0)
                    if ue_obj: ue_obj.on_request_completed(current_time_step)
                else:
                    tasks[0] = (fid, task_rem)
            
            ue_dl_bits += proc
            self._backlog_tx_ue_uav[ue_id] -= proc
            tx_times_ue.append(proc / rate)
            if self._backlog_tx_ue_uav[ue_id] < 1.0: del self._backlog_tx_ue_uav[ue_id]

        actual_tx_ue_uav = max(tx_times_ue, default=0.0)
        self._actual_processed_bits_dl = ue_dl_bits
        
        # --- (B) UAV-UAV 链路 (FDM) ---
        # 汇总新负荷
        for c_id, bits in self._bits_tx_uav_uav_as_requester.items():
            self._backlog_tx_uav_uav_as_requester[c_id] = self._backlog_tx_uav_uav_as_requester.get(c_id, 0.0) + bits
        for c_id, bits in self._bits_rx_uav_uav_as_requester.items():
            self._backlog_rx_uav_uav_as_requester[c_id] = self._backlog_rx_uav_uav_as_requester.get(c_id, 0.0) + bits
        for r_id, bits in self._bits_tx_uav_uav_as_collaborator.items():
            self._backlog_tx_uav_uav_as_collaborator[r_id] = self._backlog_tx_uav_uav_as_collaborator.get(r_id, 0.0) + bits
        for r_id, bits in self._bits_rx_uav_uav_as_collaborator.items():
            self._backlog_rx_uav_uav_as_collaborator[r_id] = self._backlog_rx_uav_uav_as_collaborator.get(r_id, 0.0) + bits

        #处理 U2U 发送 (作为协作者或请求者)
        actual_tx_uav_uav = 0.0
        # 1. 我作为请求方发送请求比特
        for target_id in list(self._backlog_tx_uav_uav_as_requester.keys()):
            rate = self._neighbor_rates.get(target_id, 0.0)
            if rate <= 0.0: continue
            proc = min(self._backlog_tx_uav_uav_as_requester[target_id], rate * tau)
            self._backlog_tx_uav_uav_as_requester[target_id] -= proc
            actual_tx_uav_uav = max(actual_tx_uav_uav, proc / rate)
            if self._backlog_tx_uav_uav_as_requester[target_id] < 1.0: del self._backlog_tx_uav_uav_as_requester[target_id]

        # 2. 我作为协作者发送数据比特给请求方
        for r_id in list(self._backlog_tx_uav_uav_as_collaborator.keys()):
            rate = self._neighbor_rates.get(r_id, 0.0)
            if rate <= 0.0: continue
            proc = min(self._backlog_tx_uav_uav_as_collaborator[r_id], rate * tau)
            
            # 流水线：从协作者发送队列消费
            tasks = self._u2u_tx_tasks_as_collab.get(r_id, [])
            rem = proc
            while rem > 0 and tasks:
                t = tasks[0]
                used = min(t["bits"], rem)
                t["bits"] -= used
                rem -= used
                if t["bits"] <= 1.0: tasks.pop(0)
                else: tasks[0] = t
            
            self._backlog_tx_uav_uav_as_collaborator[r_id] -= proc
            actual_tx_uav_uav = max(actual_tx_uav_uav, proc / rate)
            if self._backlog_tx_uav_uav_as_collaborator[r_id] < 1.0: del self._backlog_tx_uav_uav_as_collaborator[r_id]

        actual_rx_uav_uav = 0.0
        # 处理 U2U 接收 (分发到下一级)
        for target_id in list(self._backlog_rx_uav_uav_as_requester.keys()):
            rate = self._neighbor_rates.get(target_id, 0.0)
            if rate <= 0.0: continue
            proc = min(self._backlog_rx_uav_uav_as_requester[target_id], rate * tau)
            
            # 流水线：移动到下一级 backlog
            tasks = self._u2u_rx_tasks.get(target_id, [])
            rem = proc
            while rem > 0 and tasks:
                t = tasks[0]
                used = min(t["bits"], rem)
                t["bits"] -= used
                rem -= used
                if t["bits"] <= 1.0:
                    tasks.pop(0)
                    # 比特流到达 -> 注入下行 或 缓存
                    full_bits = config.FILE_SIZES[t["fid"]] * config.BITS_PER_BYTE
                    if t["target_type"] == "UE": self._add_dl_task(t["ue_id"], t["fid"], full_bits)
                    elif t["target_type"] == "CACHE": _try_add_file_to_cache(self, t["fid"])
                else: tasks[0] = t

            self._backlog_rx_uav_uav_as_requester[target_id] -= proc
            actual_rx_uav_uav = max(actual_rx_uav_uav, proc / rate)
            if self._backlog_rx_uav_uav_as_requester[target_id] < 1.0: del self._backlog_rx_uav_uav_as_requester[target_id]

        # 协作者侧的接收比特（简单消耗，不处理具体任务流，因为那是请求方的逻辑）
        for r_id in list(self._backlog_rx_uav_uav_as_collaborator.keys()):
            rate = self._neighbor_rates.get(r_id, 0.0)
            if rate <= 0.0: continue
            proc = min(self._backlog_rx_uav_uav_as_collaborator[r_id], rate * tau)
            self._backlog_rx_uav_uav_as_collaborator[r_id] -= proc
            actual_rx_uav_uav = max(actual_rx_uav_uav, proc / rate)
            if self._backlog_rx_uav_uav_as_collaborator[r_id] < 1.0: del self._backlog_rx_uav_uav_as_collaborator[r_id]
        
        # --- (C) UAV-MBS 链路 ---
        self._backlog_tx_uav_mbs += self._bits_tx_uav_mbs
        self._backlog_rx_uav_mbs += self._bits_rx_uav_mbs
        
        # MBS 上行
        actual_tx_uav_mbs = 0.0
        if self._uav_mbs_uplink_rate > 0:
            proc = min(self._backlog_tx_uav_mbs, self._uav_mbs_uplink_rate * tau)
            self._backlog_tx_uav_mbs -= proc
            actual_tx_uav_mbs = proc / self._uav_mbs_uplink_rate

        # MBS 下行 (全网流水线核心)
        actual_rx_uav_mbs = 0.0
        if self._uav_mbs_downlink_rate > 0:
            proc = min(self._backlog_rx_uav_mbs, self._uav_mbs_downlink_rate * tau)
            
            # 分发到任务
            rem = proc
            while rem > 0 and self._mbs_rx_tasks:
                t = self._mbs_rx_tasks[0]
                used = min(t["bits"], rem)
                t["bits"] -= used
                rem -= used
                if t["bits"] <= 1.0:
                    self._mbs_rx_tasks.pop(0)
                    # 到达目的地或中转站
                    full_bits = config.FILE_SIZES[t["fid"]] * config.BITS_PER_BYTE
                    if t["target_type"] == "UE": self._add_dl_task(t["ue_id"], t["fid"], full_bits)
                    elif t["target_type"] == "CACHE": _try_add_file_to_cache(self, t["fid"])
                    elif t["target_type"] == "COL": # 中转给请求者
                        # 此时比特已到达协作者（我），注入协作者到请求者的 U2U 链路
                        full_bits = config.FILE_SIZES[t["fid"]] * config.BITS_PER_BYTE
                        requester = all_uavs[t["target_id"]]
                        
                        # 协作者侧：增加 TX 积压
                        self._backlog_tx_uav_uav_as_collaborator[t["target_id"]] = self._backlog_tx_uav_uav_as_collaborator.get(t["target_id"], 0.0) + full_bits
                        self._u2u_tx_tasks_as_collab.setdefault(t["target_id"], []).append({
                            "fid": t["fid"], "bits": full_bits, "ue_id": t["ue_id"]
                        })
                        
                        # 请求者侧：增加 RX 任务及积压
                        requester._u2u_rx_tasks.setdefault(self.id, []).append({
                            "fid": t["fid"], "bits": full_bits, "target_type": "UE", 
                            "target_id": t["ue_id"], "ue_id": t["ue_id"]
                        })
                        requester._backlog_rx_uav_uav_as_requester[self.id] = requester._backlog_rx_uav_uav_as_requester.get(self.id, 0.0) + full_bits
                        
                        # 请求者侧：移除等待标记
                        if self.id in requester._u2u_wait_tasks:
                            requester._u2u_wait_tasks[self.id] = [wt for wt in requester._u2u_wait_tasks[self.id] if not (wt[0] == t["fid"] and wt[1] == t["ue_id"])]
                else: self._mbs_rx_tasks[0] = t

            self._backlog_rx_uav_mbs -= proc
            actual_rx_uav_mbs = proc / self._uav_mbs_downlink_rate

        # 3. 总能耗汇总
        total_tx_time = float(np.clip(actual_tx_ue_uav + actual_tx_uav_uav + actual_tx_uav_mbs, 0.0, 3*tau))
        total_rx_time = float(np.clip(actual_rx_ue_uav + actual_rx_uav_uav + actual_rx_uav_mbs, 0.0, 3*tau))
        comm_energy = (config.TRANSMIT_POWER * total_tx_time + config.RECEIVE_POWER * total_rx_time)

        self._energy_current_slot = fly_energy + comm_energy
