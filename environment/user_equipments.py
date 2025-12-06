import config
import numpy as np
# 模拟用户设备（UE）在移动边缘计算环境中的行为。它管理 UE 的位置、请求生成和服务覆盖率。
# 代码基于随机游走模型（Random Waypoint Model）模拟 UE 移动，并使用 Zipf 分布生成请求。
class UE:
    service_ids: np.ndarray# 服务文件ID数组，从0到NUM_SERVICES-1
    content_ids: np.ndarray# 内容文件ID数组，从NUM_SERVICES到NUM_FILES-1
    # 基于 Zipf 分布的概率数组，用于偏好热门文件
    service_zipf_probabilities: np.ndarray
    content_zipf_probabilities: np.ndarray

    @classmethod
    # 初始化 Zipf 分布：计算服务和内容的排名概率（1 / rank^beta），归一化后存储在类变量中。
    # 反映了真实网络流量：少数热门内容占多数请求。
    def initialize_ue_class(cls) -> None:
        # Service Zipf distribution
        cls.service_ids = np.arange(0, config.NUM_SERVICES)
        service_ranks: np.ndarray = np.arange(1, config.NUM_SERVICES + 1)
        service_zipf_denom: float = np.sum(1 / service_ranks**config.ZIPF_BETA)
        cls.service_zipf_probabilities = (1 / service_ranks**config.ZIPF_BETA) / service_zipf_denom

        # Content Zipf distribution
        cls.content_ids = np.arange(config.NUM_SERVICES, config.NUM_FILES)
        # 一个 NumPy 数组，从 1 到 config.NUM_CONTENTS，表示内容的排名（1 为最热门）。
        content_ranks: np.ndarray = np.arange(1, config.NUM_CONTENTS + 1)
        # config.ZIPF_BETA: Zipf 参数（通常 > 0），控制分布偏斜度（值越大，热门文件概率越高）。
        # 1 / content_ranks**config.ZIPF_BETA: 计算每个排名的未归一化概率权重（Zipf 公式：概率 ∝ 1 / rank^beta）。
        # content_zipf_denom: 这些权重的总和（使用 np.sum 计算），用于归一化。
        content_zipf_denom: float = np.sum(1 / content_ranks**config.ZIPF_BETA)
        # cls.content_zipf_probabilities: 归一化后的概率数组（每个权重除以总和），确保总和为 1。用于 np.random.choice 选择内容 ID，模拟用户偏好热门文件。
        cls.content_zipf_probabilities = (1 / content_ranks**config.ZIPF_BETA) / content_zipf_denom

    def __init__(self, ue_id: int) -> None:
        # self.id: 设置为传入的 ue_id。
        self.id: int = ue_id
        # self.pos: 初始化为一个 NumPy 数组，表示 UE 的位置。x 和 y 坐标在区域宽度和高度范围内随机生成，z 坐标固定为 0（地面）。
        self.pos: np.ndarray = np.array([np.random.uniform(0, config.AREA_WIDTH), np.random.uniform(0, config.AREA_HEIGHT), 0.0])
        # self.current_request: 一个元组，表示当前请求，格式为 (请求类型, 请求大小, 请求 ID)。初始值为 (0, 0, 0)。请求类型包括服务请求（0）和内容请求（1）。
        self.current_request: tuple[int, int, int] = (0, 0, 0)  # Request : (req_type, req_size, req_id)
        # self.latency_current_request: 浮点数，表示当前请求的延迟，初始值为 0.0。
        self.latency_current_request: float = 0.0  # Latency for the current request
        # self.assigned: 布尔值，指示 UE 是否已分配给 UAV，初始值为 False。
        self.assigned: bool = False

        # Random Waypoint Model：随机游走模型变量
        self._waypoint: np.ndarray# 目标位置坐标 [x, y]
        self._wait_time: int# 在到达目标位置后等待的时间步数
        self._set_new_waypoint()  # Initialize first waypoint

        # Fairness Tracking：公平性跟踪
        self._successful_requests: int = 0# 成功处理的请求数量
        self.service_coverage: float = 0.0# 服务覆盖率（成功请求数 / 总请求数）

    def update_position(self) -> None:
        """Updates the UE's position for one time slot as per the Random Waypoint model."""
        if self._wait_time > 0:
            self._wait_time -= 1
            return

        direction_vec: np.ndarray = self._waypoint - self.pos[:2]
        distance_to_waypoint: float = float(np.linalg.norm(direction_vec))

        if config.UE_MAX_DIST >= distance_to_waypoint:  # Reached the waypoint
            self.pos[:2] = self._waypoint
            self._set_new_waypoint()
        else:  # Move towards the waypoint
            move_vector = (direction_vec / distance_to_waypoint) * config.UE_MAX_DIST
            self.pos[:2] += move_vector

    def generate_request(self) -> None:
        """Generates a new request tuple for the current time slot."""
        # Determine request type: 0=service, 1=content
        req_type: int = np.random.choice([0, 1])

        req_id: int = -1
        # Select file ID based on request type and corresponding Zipf probabilities
        if req_type == 0:  # Service request
            req_id = np.random.choice(UE.service_ids, p=UE.service_zipf_probabilities)
        else:  # Content request
            req_id = np.random.choice(UE.content_ids, p=UE.content_zipf_probabilities)

        # Determine input data size (L_m(t))
        req_size: int = 0
        if req_type == 0:
            req_size = np.random.randint(config.MIN_INPUT_SIZE, config.MAX_INPUT_SIZE)

        self.current_request = (req_type, req_size, req_id)
        self.latency_current_request = 0.0
        self.assigned = False

    def update_service_coverage(self, current_time_step_t: int) -> None:
        """Updates the fairness metric based on service outcome in the current slot."""
        if self.assigned and self.latency_current_request <= config.TIME_SLOT_DURATION:
            self._successful_requests += 1

        assert current_time_step_t > 0
        self.service_coverage = self._successful_requests / current_time_step_t
    # 实现随机游走模型，为 UE 设置新的目标位置、速度和等待时间。
    # 是否需要设置约束不要距离原本位置太远？
    def _set_new_waypoint(self):
        """Set a new destination, speed, and wait time as per the Random Waypoint model."""
        self._waypoint = np.array([np.random.uniform(0, config.AREA_WIDTH), np.random.uniform(0, config.AREA_HEIGHT)])
        self._wait_time = np.random.randint(0, config.UE_MAX_WAIT_TIME + 1)
