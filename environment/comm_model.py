import config
import numpy as np

"""
空对地信道模型 (Air-to-Ground Channel Model)
================================================
实现基于 ITU-R / 3GPP TR 36.777 的视距(LoS)/非视距(NLoS)概率信道模型，
支持 3D 波束赋形天线增益计算。

模型特点：
1. LoS概率模型：基于仰角计算视距链路概率
   P_LoS = 1 / (1 + a * exp(-b * (θ - a)))

2. 平均路径损耗：结合LoS和NLoS的加权平均
   PL_avg = P_LoS * PL_LoS + (1 - P_LoS) * PL_NLoS

3. 3D波束赋形增益 (3GPP TR 38.901):
   G(θ,φ) = G_max - min(12*((θ-θ0)/θ_3dB)^2 + 12*((φ-φ0)/φ_3dB)^2, SLA)

位置坐标格式：[x, y, z] (meters)
- UE: [x, y, 0.0] - 地面用户
- UAV: [x, y, UAV_ALTITUDE] - 空中基站
- MBS: [500.0, 500.0, 30.0] - 宏基站
"""


def _calculate_elevation_angle(pos_ground: np.ndarray, pos_aerial: np.ndarray) -> float:
    """
    计算从地面点到空中点的仰角（elevation angle）。
    
    Args:
        pos_ground: 地面位置 [x, y, z_ground]
        pos_aerial: 空中位置 [x, y, z_aerial]
    
    Returns:
        仰角（度），范围 [0, 90]
    """
    horizontal_dist = np.sqrt((pos_aerial[0] - pos_ground[0])**2 + (pos_aerial[1] - pos_ground[1])**2)
    vertical_dist = abs(pos_aerial[2] - pos_ground[2])
    
    if horizontal_dist < config.EPSILON:
        return 90.0  # 正上方
    
    # 仰角 = arctan(垂直距离 / 水平距离)
    elevation_rad = np.arctan(vertical_dist / horizontal_dist)
    return np.degrees(elevation_rad)


def _calculate_los_probability(elevation_angle: float) -> float:
    """
    计算视距(LoS)链路概率。
    基于 ITU-R / 3GPP 模型: P_LoS = 1 / (1 + a * exp(-b * (θ - a)))
    Args:
        elevation_angle: 仰角（度）
    
    Returns:
        LoS概率，范围 [0, 1]
    """
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
    计算波束指向角（指向关联UE的质心）。
    
    Args:
        uav_pos: UAV位置 [x, y, z]
        ue_positions: 关联UE位置列表
    
    Returns:
        (theta_0, phi_0): 波束指向的俯仰角和方位角（度）
    """
    if not ue_positions:
        return (0.0, 0.0)  # 默认垂直下倾
    
    centroid = np.mean(ue_positions, axis=0)
    dx, dy = centroid[0] - uav_pos[0], centroid[1] - uav_pos[1]
    dz = uav_pos[2] - centroid[2]
    
    horizontal_dist = np.sqrt(dx**2 + dy**2)
    theta_0 = np.degrees(np.arctan2(horizontal_dist, dz)) if dz > config.EPSILON else 90.0
    phi_0 = np.degrees(np.arctan2(dy, dx))
    
    return (theta_0, phi_0)


def _calculate_beam_gain(uav_pos: np.ndarray, target_pos: np.ndarray, 
                         beam_direction: tuple[float, float]) -> float:
    """
    计算3D波束赋形天线增益（3GPP TR 38.901模型）。
    G(θ,φ) = G_max - min(12*(Δθ/θ_3dB)² + 12*(Δφ/φ_3dB)², SLA)
    """
    if not config.ENABLE_BEAMFORMING:
        return 1.0
    
    theta_0, phi_0 = beam_direction
    dx, dy = target_pos[0] - uav_pos[0], target_pos[1] - uav_pos[1]
    dz = uav_pos[2] - target_pos[2]
    
    horizontal_dist = np.sqrt(dx**2 + dy**2)
    theta_target = np.degrees(np.arctan2(horizontal_dist, dz)) if dz > config.EPSILON else 90.0
    phi_target = np.degrees(np.arctan2(dy, dx))
    
    # 角度偏差
    theta_diff = theta_target - theta_0
    phi_diff = _wrap_angle(phi_target - phi_0)
    
    # 3GPP天线模型
    attenuation_db = min(
        12.0 * (theta_diff / config.THETA_3DB)**2 + 12.0 * (phi_diff / config.PHI_3DB)**2,
        config.SLA_DB
    )
    return 10.0 ** ((config.G_MAX_DBI - attenuation_db) / 10.0)


def calculate_channel_gain(pos1: np.ndarray, pos2: np.ndarray, 
                           beam_direction: tuple[float, float] | None = None) -> float:
    """
    计算两点之间的信道增益，考虑LoS/NLoS概率和3D波束赋形。
    
    Args:
        pos1, pos2: 位置 [x, y, z]
        beam_direction: UAV波束指向 (theta_0, phi_0)，仅用于空对地链路
    
    Returns:
        信道增益（线性值）
    """
    distance = np.sqrt(np.sum((pos1 - pos2) ** 2))
    is_air_to_ground = (pos1[2] < 1.0) or (pos2[2] < 1.0)
    
    if is_air_to_ground:
        pos_ground, pos_aerial = (pos1, pos2) if pos1[2] < pos2[2] else (pos2, pos1)
        
        elevation_angle = _calculate_elevation_angle(pos_ground, pos_aerial)
        p_los = _calculate_los_probability(elevation_angle)
        
        path_loss = _calculate_path_loss(distance)
        nlos_factor = 10.0 ** (config.NLOS_ADDITIONAL_LOSS_DB / 10.0)
        avg_path_loss = p_los * path_loss + (1.0 - p_los) * path_loss * nlos_factor
        
        beam_gain = _calculate_beam_gain(pos_aerial, pos_ground, beam_direction) if beam_direction else 1.0
    else:
        avg_path_loss = _calculate_path_loss(distance)
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
                                  interferer_beam_direction: tuple[float, float],
                                  interferer_num_ues: int) -> float:
    """计算单个干扰UAV对UE造成的干扰功率。
    
    干扰功率 = (干扰UAV的发射功率/其关联UE数) × 信道增益
    
    Args:
        interfering_uav_pos: 干扰UAV的位置
        ue_pos: 受干扰UE的位置
        interferer_beam_direction: 干扰UAV的波束方向
        interferer_num_ues: 干扰UAV关联的UE数量（用于确定其发射功率分配）
    
    Returns:
        干扰功率（线性值）
    """
    if interferer_num_ues == 0:
        return 0.0  # 干扰UAV没有关联UE时不发射
    
    # 计算干扰链路的信道增益（考虑波束方向）
    interference_channel_gain = calculate_channel_gain(
        ue_pos, interfering_uav_pos, interferer_beam_direction
    )
    
    # 干扰UAV的功率按OFDMA分配给其关联的UE
    # 对于受干扰UE，它接收到的是干扰UAV的全部发射功率（因为不在OFDMA子载波分配中）
    # 保守估计：使用干扰UAV的总发射功率
    interference_power = config.TRANSMIT_POWER * interference_channel_gain
    
    return interference_power


def calculate_uav_uav_rate(channel_gain: float, num_collaborating_uavs: int = 1) -> float:
    """Calculates data rate between two UAVs.
    
    采用频分复用(FDM)：当一个UAV被多个邻居选为协作者时，带宽和功率都需要平分。
    这符合总功率限制模型：UAV 的总发射功率固定，FDM 时每条链路分得 1/N 的功率。
    
    Args:
        channel_gain: 信道增益
        num_collaborating_uavs: 需要服务的协作UAV数量（被多少个UAV选为协作者）
    """
    assert num_collaborating_uavs >= 1
    # FDM: 带宽和功率都平分给各链路
    bandwidth_per_link: float = config.BANDWIDTH_INTER / num_collaborating_uavs
    power_per_link: float = config.TRANSMIT_POWER / num_collaborating_uavs
    snr: float = (power_per_link * channel_gain) / config.AWGN
    return bandwidth_per_link * np.log2(1 + snr)
