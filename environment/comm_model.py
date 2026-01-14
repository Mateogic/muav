import config
import numpy as np

"""
3D 球形覆盖信道模型 (3D Spherical Coverage Channel Model)
==========================================================
实现支持任意 3D 相对位置的信道模型，包括：
- UAV 在上方服务地面/低空 UE
- UAV 在下方服务高空 UE  
- UAV 与 UE 同高度的水平链路

模型特点：
1. LoS概率模型：基于仰角计算视距链路概率（适应高空 UE）
2. 3D 波束赋形：使用球坐标系，俯仰角范围 [0°, 180°]
   - 0° = 正上方（天顶）
   - 90° = 水平
   - 180° = 正下方（天底）
3. 球形覆盖区域：以 UAV 为中心的 3D 球形关联范围

位置坐标格式：[x, y, z] (meters)
"""


def _calculate_elevation_angle(pos1: np.ndarray, pos2: np.ndarray) -> float:
    """
    计算两点之间连线相对于水平面的仰角。
    
    Args:
        pos1, pos2: 两个 3D 位置 [x, y, z]
    
    Returns:
        仰角（度），范围 [0, 90]，始终为正值
    """
    horizontal_dist = np.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
    vertical_dist = abs(pos2[2] - pos1[2])
    
    if horizontal_dist < config.EPSILON:
        return 90.0  # 垂直方向
    
    # 仰角 = arctan(垂直距离 / 水平距离)
    elevation_rad = np.arctan(vertical_dist / horizontal_dist)
    return np.degrees(elevation_rad)


def _calculate_los_probability(pos1: np.ndarray, pos2: np.ndarray) -> float:
    """
    计算两点之间的视距(LoS)链路概率。
    
    对于空对空链路（双方都在高空），LoS概率接近1.0。
    对于有一方接近地面的链路，使用 ITU-R / 3GPP 模型。
    
    Args:
        pos1, pos2: 两个 3D 位置 [x, y, z]
    
    Returns:
        LoS概率，范围 [0, 1]
    """
    min_height = min(pos1[2], pos2[2])
    elevation_angle = _calculate_elevation_angle(pos1, pos2)
    
    # 空对空链路：双方都在空中UE最低高度以上时，遮挡概率很低
    if min_height >= config.UE_AERIAL_MIN_ALT:
        # 两端都在空中（≥50m），视为空对空链路
        # 仰角越大（更垂直），LoS 概率越高
        # P_LoS ∈ [0.8, 1.0]，比地对空链路更保守
        p_los = 0.8 + 0.2 * (elevation_angle / 90.0)
        return np.clip(p_los, 0.0, 1.0)
    
    # 有一端接近地面：使用原 LoS 模型
    a, b = config.LOS_PARAMS.get(config.ENVIRONMENT_TYPE, config.LOS_PARAMS['urban'])
    p_los = 1.0 / (1.0 + a * np.exp(-b * (elevation_angle - a)))
    return np.clip(p_los, 0.0, 1.0)

def _calculate_path_loss(distance: float) -> float:
    """计算简化路径损耗 (距离平方模型)。"""
    return distance ** 2


def _wrap_angle(angle: float) -> float:
    """将角度归一化到 [-180, 180] 范围。"""
    return ((angle + 180.0) % 360.0) - 180.0


def calculate_beam_direction(uav_pos: np.ndarray, ue_positions: list[np.ndarray]) -> tuple[float, float]:
    """
    计算波束指向角（指向关联UE的质心），使用球坐标系。
    
    球坐标系定义：
    - theta (俯仰角): 从 Z+ 轴（天顶）测量，[0°, 180°]
      - 0° = 正上方
      - 90° = 水平
      - 180° = 正下方
    - phi (方位角): 在 XY 平面上，[-180°, 180°]
    
    Args:
        uav_pos: UAV位置 [x, y, z]
        ue_positions: 关联UE位置列表
    
    Returns:
        (theta_0, phi_0): 波束指向的俯仰角和方位角（度）
    """
    if not ue_positions:
        return (90.0, 0.0)  # 默认水平方向
    
    centroid = np.mean(ue_positions, axis=0)
    dx = centroid[0] - uav_pos[0]
    dy = centroid[1] - uav_pos[1]
    dz = centroid[2] - uav_pos[2]  # 注意：这里是目标减UAV，可正可负
    
    horizontal_dist = np.sqrt(dx**2 + dy**2)
    distance_3d = np.sqrt(dx**2 + dy**2 + dz**2)
    
    if distance_3d < config.EPSILON:
        return (90.0, 0.0)  # 重合时默认水平
    
    # 球坐标系：theta 从 Z+ 轴测量
    # arccos(dz / r) 给出从天顶的角度
    theta_0 = np.degrees(np.arccos(np.clip(dz / distance_3d, -1.0, 1.0)))
    phi_0 = np.degrees(np.arctan2(dy, dx))
    
    return (theta_0, phi_0)


def _calculate_beam_gain(uav_pos: np.ndarray, target_pos: np.ndarray, 
                         beam_direction: tuple[float, float]) -> float:
    """
    计算3D波束赋形天线增益（3GPP TR 38.901模型），支持全向覆盖。
    
    使用球坐标系，theta 范围 [0°, 180°]：
    - 0° = 正上方（天顶）
    - 90° = 水平
    - 180° = 正下方（天底）
    
    通过计算波束方向与目标方向之间的真实角度偏差（大圆距离）来确定增益衰减，
    避免在极点附近因方位角差异导致的计算错误。
    """
    if not config.ENABLE_BEAMFORMING:
        return 1.0
    
    theta_0, phi_0 = beam_direction
    dx = target_pos[0] - uav_pos[0]
    dy = target_pos[1] - uav_pos[1]
    dz = target_pos[2] - uav_pos[2]
    
    distance_3d = np.sqrt(dx**2 + dy**2 + dz**2)
    
    if distance_3d < config.EPSILON:
        return 10.0 ** (config.G_MAX_DBI / 10.0)  # 重合时返回最大增益
    
    # 目标方向的单位向量
    target_vec = np.array([dx, dy, dz]) / distance_3d
    
    # 波束方向的单位向量（从球坐标转换）
    theta_0_rad = np.radians(theta_0)
    phi_0_rad = np.radians(phi_0)
    beam_vec = np.array([
        np.sin(theta_0_rad) * np.cos(phi_0_rad),
        np.sin(theta_0_rad) * np.sin(phi_0_rad),
        np.cos(theta_0_rad)
    ])
    
    # 计算真实角度偏差（两向量夹角）
    cos_angle = np.clip(np.dot(beam_vec, target_vec), -1.0, 1.0)
    angular_deviation = np.degrees(np.arccos(cos_angle))
    
    # 使用真实角度偏差计算增益衰减
    # G = G_max - min(12*(Δ/θ_3dB)², SLA)
    attenuation_db = min(12.0 * (angular_deviation / config.THETA_3DB)**2, config.SLA_DB)
    return 10.0 ** ((config.G_MAX_DBI - attenuation_db) / 10.0)


def calculate_channel_gain(pos1: np.ndarray, pos2: np.ndarray, 
                           beam_direction: tuple[float, float] | None = None) -> float:
    """
    计算两点之间的信道增益，支持任意 3D 相对位置。
    
    链路类型判断：
    - 有波束方向时：UAV-UE 链路，应用波束增益
    - 无波束方向时：UAV-UAV 或 UAV-MBS 链路，不应用波束增益
    
    Args:
        pos1, pos2: 位置 [x, y, z]
        beam_direction: UAV波束指向 (theta_0, phi_0)，仅用于 UAV-UE 链路
    
    Returns:
        信道增益（线性值）
    """
    distance = np.sqrt(np.sum((pos1 - pos2) ** 2))
    path_loss = _calculate_path_loss(distance)
    
    # 计算 LoS 概率
    p_los = _calculate_los_probability(pos1, pos2)
    
    # NLoS 额外损耗
    nlos_factor = 10.0 ** (config.NLOS_ADDITIONAL_LOSS_DB / 10.0)
    avg_path_loss = p_los * path_loss + (1.0 - p_los) * path_loss * nlos_factor
    
    # 波束增益（仅 UAV-UE 链路有波束方向）
    if beam_direction is not None:
        # 确定 UAV 位置（用于波束增益计算）
        # 通常 pos1 是 UE，pos2 是 UAV（调用时的惯例）
        # 但需要兼容两种调用顺序
        beam_gain = _calculate_beam_gain(pos2, pos1, beam_direction)
    else:
        beam_gain = 1.0
    
    return config.G_CONSTS_PRODUCT * beam_gain / (avg_path_loss + config.EPSILON)


def calculate_ue_uav_rate(channel_gain: float, num_associated_ues: int, interference_power: float = 0.0) -> float:
    """Calculates downlink data rate from UAV to UE with co-channel interference.
    
    下行链路：UAV → UE，使用 OFDMA 多址方式。
    总功率限制模型：UAV 的总发射功率固定，OFDMA 时每个 UE 分得 1/N 的带宽和功率。
    
    考虑同频干扰：其他UAV在相同频段发射的信号会对该UE造成干扰。
    
    OFDMA子载波级别SINR计算：
    - 信号功率：P_tx/N × G_signal（服务UAV分配给该UE的功率）
    - 噪声功率：σ²/N（子载波带宽上的热噪声）
    - 干扰功率：I_total/N（干扰功率也分散在整个频带，只有1/N落入该子载波）
    
    SINR = (P/N × G) / (σ²/N + I/N) = (P × G) / (σ² + I)
    
    Args:
        channel_gain: 服务UAV到UE的信道增益
        num_associated_ues: 服务UAV关联的UE数量（用于OFDMA功率/带宽分配）
        interference_power: 来自其他UAV的同频干扰功率总和（全频带）
    
    Returns:
        下行数据速率 (bits/s)
    """
    assert num_associated_ues != 0
    # OFDMA: 带宽平分给各 UE
    bandwidth_per_ue: float = config.BANDWIDTH_EDGE / num_associated_ues
    # SINR计算：由于噪声和干扰也按1/N缩放，最终等价于使用全功率除以全频带噪声+干扰
    # 这是OFDMA系统的标准特性：子载波SINR = 全频带SNR
    sinr: float = (config.TRANSMIT_POWER * channel_gain) / (config.AWGN + interference_power)
    return bandwidth_per_ue * np.log2(1 + sinr)


def calculate_ue_uav_uplink_rate(channel_gain: float, num_associated_ues: int) -> float:
    """Calculates uplink data rate from UE to UAV.
    
    上行链路：UE → UAV，使用 UE 发射功率（通常远小于 UAV）。
    """
    assert num_associated_ues != 0
    bandwidth_per_ue: float = config.BANDWIDTH_EDGE / num_associated_ues
    snr: float = (config.UE_TRANSMIT_POWER * channel_gain) / config.AWGN
    return bandwidth_per_ue * np.log2(1 + snr)


def calculate_uav_mbs_uplink_rate(channel_gain: float) -> float:
    """Calculates uplink data rate from UAV to MBS.
    
    上行链路：UAV → MBS，使用 UAV 发射功率。
    """
    snr: float = (config.TRANSMIT_POWER * channel_gain) / config.AWGN
    return config.BANDWIDTH_BACKHAUL * np.log2(1 + snr)


def calculate_uav_mbs_downlink_rate(channel_gain: float) -> float:
    """Calculates downlink data rate from MBS to UAV.
    
    下行链路：MBS → UAV，使用 MBS 发射功率（远大于 UAV）。
    """
    snr: float = (config.MBS_TRANSMIT_POWER * channel_gain) / config.AWGN
    return config.BANDWIDTH_BACKHAUL * np.log2(1 + snr)


def calculate_interference_power(interfering_uav_pos: np.ndarray, ue_pos: np.ndarray,
                                  interferer_beam_direction: tuple[float, float]) -> float:
    """计算单个干扰UAV对UE造成的全频带干扰功率（保守估计）。
    
    采用保守估计模型：假设干扰UAV的发射功率均匀分布在整个频带上，
    受干扰UE接收到干扰UAV的全部发射功率。这是同频复用场景下的标准做法。
    
    注：calculate_ue_uav_rate 中的 SINR 计算会隐式处理子载波分割，
    即 SINR = (P×G)/(σ²+I)，其中 I 为全频带干扰功率。
    
    Args:
        interfering_uav_pos: 干扰UAV的位置
        ue_pos: 受干扰UE的位置
        interferer_beam_direction: 干扰UAV的波束方向
    
    Returns:
        全频带干扰功率（线性值）
    """
    # 计算干扰链路的信道增益（考虑波束方向）
    interference_channel_gain = calculate_channel_gain(
        ue_pos, interfering_uav_pos, interferer_beam_direction
    )
    
    # 全频带干扰功率 = 干扰UAV总发射功率 × 信道增益
    interference_power = config.TRANSMIT_POWER * interference_channel_gain
    
    return interference_power


def calculate_uav_uav_rate(channel_gain: float, num_collaborating_uavs: int = 1) -> float:
    """Calculates data rate between two UAVs.
    
    采用频分复用(FDM)：当一个UAV被多个邻居选为协作者时，带宽和功率都需要平分。
    这符合总功率限制模型：UAV 的总发射功率固定，FDM 时每条链路分得 1/N 的功率。
    
    FDM 噪声缩放：由于每条链路使用独立的子频带 B/N，噪声功率也相应缩放为 σ²/N。
    因此 SNR = (P/N × G) / (σ²/N) = (P × G) / σ²，与协作者数量无关。
    
    Args:
        channel_gain: 信道增益
        num_collaborating_uavs: 需要服务的协作UAV数量（被多少个UAV选为协作者）
    """
    assert num_collaborating_uavs >= 1
    # FDM: 带宽和功率都平分给各链路，噪声也相应缩放
    bandwidth_per_link: float = config.BANDWIDTH_INTER / num_collaborating_uavs
    # FDM 子频带噪声功率 = 总噪声功率 / N（噪声功率与带宽成正比）
    # SNR = (P/N × G) / (σ²/N) = (P × G) / σ²
    snr: float = (config.TRANSMIT_POWER * channel_gain) / config.AWGN
    return bandwidth_per_link * np.log2(1 + snr)
