import config
import numpy as np
# 模拟无人机（UAV）、用户设备（UE）和宏基站（MBS）之间的无线通信
# 基于自由空间路径损耗模型计算两个位置之间的信道增益
"""
位置坐标 pos 是一个包含3个元素的 numpy 数组，格式为 [x, y, z]：

UE设备位置：在 user_equipments.py 中初始化为 [x, y, 0.0]，z坐标固定为0（地面） user_equipments.py:27
UAV位置：在 uavs.py 中初始化为 [x, y, UAV_ALTITUDE]，z坐标为100米（飞行高度） uavs.py:27
MBS位置：在 config.py 中定义为 [500.0, 500.0, 30.0]



是否足够符合真实环境？
不足之处：

路径损耗过于理想化：自由空间模型假设无障碍传播（直线视距），忽略阴影衰落（shadowing）、多径效应（multipath fading）和地形影响。在城市或室内环境中，信号会因建筑物、反射和散射而显著衰减。
忽略干扰和噪声：仅考虑 AWGN（加性高斯白噪声），未模拟同频干扰（co-channel interference）、邻频干扰或外部噪声源（如其他设备）。
静态信道假设：未考虑移动导致的多普勒效应（Doppler shift）、信道时变性或信道状态信息（CSI）的动态更新。
速率计算简化：香农公式适用于理想信道，但现实中需考虑调制编码方案（MCS）、误码率（BER）和自适应调制（如 OFDM）。
通信类型局限：仅覆盖 UE-UAV、UAV-MBS 和 UAV-UAV，未考虑地面网络（如 LTE/5G）或卫星链路。
能量和硬件忽略：未模拟发射机/接收机效率、功率放大器非线性或硬件约束。
符合程度：在高空 UAV 场景中（视距为主），模型大致合理，但对于地面密集 UE 或复杂环境（如城市），准确性低。适合概念验证，但不适用于精确性能评估。

可以考虑的改进
更复杂的路径损耗模型：采用 Okumura-Hata 或 ITU-R P.1411 模型，考虑频率、城市类型和高度。
阴影和快衰落：添加对数正态阴影衰落和 Rayleigh/Rician 快衰落模型。
干扰管理：模拟同频干扰（e.g., 其他 UAV/UE 的信号）和频率复用。
动态信道：引入多普勒效应和信道估计，考虑移动速度对速率的影响。
高级通信技术：支持 MIMO（多输入多输出）、波束赋形或毫米波通信。
能量效率：计算通信能量消耗（e.g., 基于功率放大器效率）。
协议栈模拟：添加 MAC 层调度、握手延迟或 QoS 保证。
环境因素：考虑天气（雨衰）、地形和网络拓扑变化。
验证与校准：通过实测数据或 NS-3 等仿真工具验证模型。
"""
def calculate_channel_gain(pos1: np.ndarray, pos2: np.ndarray) -> float:
    """Calculates channel gain based on the free-space path loss model."""
    distance_sq: float = np.sum((pos1 - pos2) ** 2)# 欧几里得距离平方
    # 天线增益常数乘积 / (距离平方 + 一个很小的数防止除零)
    return (config.G_CONSTS_PRODUCT) / (distance_sq + config.EPSILON)


def calculate_ue_uav_rate(channel_gain: float, num_associated_ues: int) -> float:
    """Calculates data rate between a UE and a UAV."""
    assert num_associated_ues != 0
    bandwidth_per_ue: float = config.BANDWIDTH_EDGE / num_associated_ues# 每个UE分配的带宽
    # 计算信噪比（发射功率/噪声功率）
    snr: float = (config.TRANSMIT_POWER * channel_gain) / config.AWGN
    # 香农公式计算数据速率
    return bandwidth_per_ue * np.log2(1 + snr)


def calculate_uav_mbs_rate(channel_gain: float) -> float:
    """Calculates data rate between a UAV and the MBS."""
    snr: float = (config.TRANSMIT_POWER * channel_gain) / config.AWGN
    return config.BANDWIDTH_BACKHAUL * np.log2(1 + snr)


def calculate_uav_uav_rate(channel_gain: float) -> float:
    """Calculates data rate between two UAVs."""
    snr: float = (config.TRANSMIT_POWER * channel_gain) / config.AWGN
    return config.BANDWIDTH_INTER * np.log2(1 + snr)
