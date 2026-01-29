from collections import deque
import config
import numpy as np
# 模拟用户设备（UE）在空中基站通信保障场景中的行为。它管理 UE 的位置、通信需求生成和服务覆盖率。
# 代码基于随机游走模型（Random Waypoint Model）模拟 UE 移动，并使用 Zipf 分布生成内容请求。
class UE:
    content_ids: np.ndarray  # 内容文件ID数组，从0到NUM_FILES-1
    # 基于 Zipf 分布的概率数组，用于偏好热门文件
    content_zipf_probabilities: np.ndarray

    @classmethod
    # 初始化 Zipf 分布：计算内容的排名概率（1 / rank^beta），归一化后存储在类变量中。
    # 反映了真实网络流量：少数热门内容占多数请求。
    def initialize_ue_class(cls) -> None:
        # Content Zipf distribution
        cls.content_ids = np.arange(0, config.NUM_FILES)
        content_ranks: np.ndarray = np.arange(1, config.NUM_CONTENTS + 1)
        # config.ZIPF_BETA: Zipf 参数（通常 > 0），控制分布偏斜度（值越大，热门文件概率越高）。
        content_zipf_denom: float = np.sum(1 / content_ranks**config.ZIPF_BETA)
        cls.content_zipf_probabilities = (1 / content_ranks**config.ZIPF_BETA) / content_zipf_denom

    def __init__(self, ue_id: int) -> None:
        # self.id: 设置为传入的 ue_id。
        self.id: int = ue_id
        
        # 分层移动：根据 UE_GROUND_RATIO 确定 UE 是地面还是空中
        self._is_aerial: bool = ue_id >= int(config.NUM_UES * config.UE_GROUND_RATIO)
        
        # self.pos: 初始化为一个 NumPy 数组，表示 UE 的 3D 位置
        if self._is_aerial:
            # 空中 UE：在指定高度范围内随机初始化
            initial_z = np.random.uniform(config.UE_AERIAL_MIN_ALT, config.UE_AERIAL_MAX_ALT)
        else:
            # 地面 UE：z 固定为 0
            initial_z = 0.0
        self.pos: np.ndarray = np.array([
            np.random.uniform(0, config.AREA_WIDTH), 
            np.random.uniform(0, config.AREA_HEIGHT), 
            initial_z
        ])
        
        # Request lifecycle
        # current_request: (req_type=1, req_size, req_id)
        self.current_request: tuple[int, int, int] = (1, 0, 0)
        self.request_active: bool = False
        self.request_start_step: int = 0

        # Step-level flags (reset every environment step)
        self.received_bits_this_step: float = 0.0
        self.completed_this_step: bool = False
        self.failed_this_step: bool = False
        self.completed_requests_total: int = 0
        self.failed_requests_total: int = 0

        # Latency for current request (used for logging/metrics)
        self.latency_current_request: float = 0.0

        # Compatibility flag: whether UE is currently assigned to any UAV in this slot
        self.assigned: bool = False

        # Random Waypoint Model：随机游走模型变量（3D）
        self._waypoint: np.ndarray  # 目标位置坐标 [x, y, z]
        self._wait_time: int  # 在到达目标位置后等待的时间步数
        self._set_new_waypoint()  # Initialize first waypoint

        # Fairness Tracking：滑动窗口公平性跟踪
        self._service_history: deque[bool] = deque(maxlen=config.FAIRNESS_WINDOW_SIZE)
        self.service_coverage: float = 0.0  # 滑动窗口内的服务成功率
        
        # Co-channel Interference：同频干扰
        self.interference_power: float = 0.0  # 来自其他UAV的同频干扰功率总和

    def reset_step_flags(self) -> None:
        """Reset per-step flags before a new environment step is simulated."""
        self.received_bits_this_step = 0.0
        self.completed_this_step = False
        self.failed_this_step = False

    def on_downlink_bits_delivered(self, bits: float) -> None:
        """Called by the serving UAV when some downlink bits are actually delivered in this step."""
        if bits > 0.0:
            self.received_bits_this_step += bits

    def on_request_completed(self, current_time_step: int) -> None:
        """Mark current request as completed at the end of current step."""
        if not self.request_active:
            return
        self.completed_this_step = True
        self.request_active = False
        self.completed_requests_total += 1
        # Approximate end-to-end latency by integer slot time.
        self.latency_current_request = (current_time_step - self.request_start_step + 1) * config.TIME_SLOT_DURATION

    def on_request_failed(self, current_time_step: int) -> None:
        """Mark current request as failed (timeout/drop) at the end of current step."""
        if not self.request_active:
            return
        self.failed_this_step = True
        self.request_active = False
        self.failed_requests_total += 1
        # Penalize failed request with a large latency.
        self.latency_current_request = config.NON_SERVED_LATENCY_PENALTY

    def update_position(self) -> None:
        """Updates the UE's position for one time slot as per the 3D Random Waypoint model."""
        if self._wait_time > 0:
            self._wait_time -= 1
            return

        # 计算当前位置到目标的 3D 向量
        direction_vec: np.ndarray = self._waypoint - self.pos
        distance_to_waypoint: float = float(np.linalg.norm(direction_vec))

        if config.UE_MAX_DIST >= distance_to_waypoint:  # Reached the waypoint
            self.pos = self._waypoint.copy()
            self._set_new_waypoint()
        else:  # Move towards the waypoint
            move_vector = (direction_vec / distance_to_waypoint) * config.UE_MAX_DIST
            self.pos += move_vector

    def generate_request(self, next_time_step: int) -> None:
        """Generates a new content request for the upcoming time slot.

        Request persists across slots until completed/failed.
        """
        if self.request_active:
            return

        req_type: int = 1
        req_id: int = int(np.random.choice(UE.content_ids, p=UE.content_zipf_probabilities))
        req_size: int = 0

        self.current_request = (req_type, req_size, req_id)
        self.request_active = True
        self.request_start_step = next_time_step
        self.latency_current_request = 0.0


    def update_service_coverage(self, current_time_step_t: int) -> None:
        """使用滑动窗口更新服务覆盖率，判断本步是否成功获得服务。"""
        # 修正：如果本步判定为失败（如超时或丢包），即使收到了比特也不计为成功。
        # 配合 Env.step 中先执行超时判定，后执行此覆盖率更新。
        success: bool = (self.received_bits_this_step >= config.SUCCESS_BIT_THRESHOLD) and not self.failed_this_step
        self._service_history.append(success)
        # 滑动窗口内的成功率（窗口未满时使用实际长度）
        self.service_coverage = sum(self._service_history) / len(self._service_history)
    # 实现 3D 随机游走模型，为 UE 设置新的目标位置和等待时间。
    def _set_new_waypoint(self):
        """Set a new 3D destination and wait time as per the Random Waypoint model."""
        new_x = np.random.uniform(0, config.AREA_WIDTH)
        new_y = np.random.uniform(0, config.AREA_HEIGHT)
        
        if self._is_aerial:
            # 空中 UE：在指定高度范围内随机选择目标高度
            new_z = np.random.uniform(config.UE_AERIAL_MIN_ALT, config.UE_AERIAL_MAX_ALT)
        else:
            # 地面 UE：z 固定为 0
            new_z = 0.0
        
        self._waypoint = np.array([new_x, new_y, new_z])
        self._wait_time = np.random.randint(0, config.UE_MAX_WAIT_TIME + 1)
