"""
# SignalSimulate: 信号仿真模块, 提供准周期信号、冲击信号与调制信号等含噪仿真信号的生成方法

    - function:
        - `periodic`: 生成仿真含噪准周期信号
        - `impulse`: 生成仿真冲击序列和噪声冲击复合信号
        - `modulation`: 生成仿真频率-幅值调制信号
"""

__all__ = ["periodic", "impulse", "modulation"]

from collections.abc import Callable
from typing import Optional

import numpy as np

from ._core import Signal, t_Axis


# --------------------------------------------------------------------------------------------#
# --------------------------------------------------------------------------------#
# ------------------------------------------------------------------------#
# ----------------------------------------------------------------#
def periodic(fs: float, T: float, CosParams: tuple, noise: float = 0.0, seed: Optional[int] = None) -> Signal:
    """
    生成仿真含噪准周期信号

    Parameters
    ----------
    fs : float
        采样频率, 单位Hz
    T : float
        信号时长, 单位s
    CosParams : tuple
        余弦成分生成参数, 格式为((f, A, phi), ...), 分别为频率、幅值、初相位
    noise : float, default=0.0
        高斯白噪声标准差
    seed : int, optional
        随机数种子, 若不指定则不固定随机数种子

    Returns
    -------
    Signal
        仿真周期信号
    """
    sig = Signal(axis=t_Axis(T=T, fs=fs), label="仿真周期信号")
    time = sig.t_axis()
    for i, params in enumerate(CosParams):
        if len(params) != 3:
            raise ValueError(f"CosParams={params}: 余弦成分生成参数格式错误")
        f, A, phi = params
        sig += A * np.cos(2 * np.pi * f * time + phi)  # 生成任意频率、幅值、初相位的余弦信号
    rng = np.random.default_rng(seed)
    sig += rng.standard_normal(len(sig)) * noise
    return sig


def impulse(
    fs: float, T: float, ImpParams: tuple, noiseParams: Optional[tuple] = None, seed: Optional[int] = None
) -> Signal:
    """
    生成仿真冲击序列和噪声冲击复合信号

    Parameters
    ----------
    fs : float
        采样频率, 单位Hz
    T : float
        信号时长, 单位s
    ImpParams : tuple
        冲击序列生成参数, 格式为(fc, fe, alpha, A, tau).
        分别为中心频率、出现频率、滑移百分比(0-1)、冲击幅值(常数、数组或函数)和幅值衰减时间
    noiseParams : tuple, optional
        噪声冲击参数, 格式为(n, la), 分别为噪声冲击个数和幅值指数分布参数
    seed : int, optional
        随机数种子, 若不指定则不固定随机数种子

    Returns
    -------
    Signal
        仿真冲击信号
    """
    sig = Signal(axis=t_Axis(T=T, fs=fs), label="仿真冲击信号")
    N = len(sig)
    t = sig.t_axis()
    # 准备冲击序列参数
    if len(ImpParams) != 5:
        raise ValueError(f"ImpParams={ImpParams}: 冲击序列生成参数格式错误")
    fc, fe, alpha, A, tau = ImpParams
    idx_gap = fs / fe  # 平均冲击间隔
    idx_sliding = int(alpha * idx_gap)  # 冲击位置滑移范围
    imp_idx = np.arange(0, N, idx_gap)  # 冲击位置索引数组
    C = -np.log(0.05) / tau**2  # 衰减常数
    imp_wave = np.exp(-C * t[: int(tau * fs)] ** 2) * np.sin(2 * np.pi * fc * t[: int(tau * fs)])  # 单个冲击波形
    A_array = A(t) if isinstance(A, Callable) else A if isinstance(A, np.ndarray) else np.full(N, A)  # 冲击幅值数组
    # 生成冲击信号
    for idx in imp_idx:
        idx = int(idx)
        idx += np.random.randint(-idx_sliding, idx_sliding + 1)  # 冲击位置滑移
        idx1 = max(0, idx)  # 实际脉冲起始索引
        idx2 = min(N, idx + len(imp_wave))  # 实际脉冲结束索引
        if idx2 > idx1:
            sig._data[idx1:idx2] += imp_wave[idx1 - idx : idx2 - idx] * A_array[idx1]  # 单个冲击幅值不变
    # 加入噪声冲击
    if noiseParams is not None:
        rng = np.random.default_rng(seed)
        n, la = noiseParams
        noise_idx = rng.choice(N, size=n, replace=False)  # 随机选择噪声冲击位置索引
        noise_amplitudes = rng.exponential(scale=la, size=n)  # la越大, 噪声冲击幅值越大
        noise_imp_wave = np.bincount(noise_idx, weights=noise_amplitudes, minlength=N)
        sig._data += noise_imp_wave
    return sig


def modulation(
    fs: float, T: float, fc: float, AM: Callable, FM: Callable, noise: float = 0.0, seed: Optional[int] = None
) -> Signal:
    """
    生成仿真频率-幅值调制信号

    Parameters
    ----------
    fs : float
        采样频率, 单位Hz
    T : float
        信号时长, 单位s
    fc : float
        载波频率, 单位Hz
    AM : callable
        调幅函数, 接受时间轴数组作为输入, 返回调幅系数数组
    FM : callable
        调频函数, 接受时间轴数组作为输入, 返回调频偏移数组
    noise : float, default=0.0
        高斯白噪声标准差
    seed : int, optional
        随机数种子, 若不指定则不固定随机数种子

    Returns
    -------
    Signal
        仿真调制信号
    """
    sig = Signal(axis=t_Axis(T=T, fs=fs), label="仿真调制信号")
    t = sig.t_axis()
    # 生成调制相位
    phase = 2 * np.pi * fc * t + 2 * np.pi * np.cumsum(FM(t)) / fs
    # 生成调制信号
    sig += AM(t) * np.cos(phase)
    # 添加噪声
    rng = np.random.default_rng(seed)
    sig += rng.standard_normal(len(sig)) * noise
    return sig
