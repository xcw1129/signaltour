"""
# SignalSample: 信号采样模块, 提供重采样、边界延拓与滑窗分段等方法

    - function:
        - `resample`: 截取信号任意时间段并重采样, 支持多种采样方式以及上采样
        - `pad`: 对信号进行边界延拓, 并对沿拓部分进行数据填充
        - `slice`: 对信号进行滑窗跳步分段, 段间严格等长且等间隔, 首尾不沿拓
"""

__all__ = ["resample", "pad", "slice"]

from typing import Optional

import numpy as np

from ._core import Signal, t_Axis


# --------------------------------------------------------------------------------------------#
# --------------------------------------------------------------------------------#
# ------------------------------------------------------------------------#
# ----------------------------------------------------------------#
def resample(
    Sig: Signal,
    type: str = "spacing",
    dt: Optional[float] = None,
    t0: Optional[float] = None,
    T: Optional[float] = None,
) -> Signal:
    """
    截取信号任意时间段并重采样, 支持多种采样方式以及上采样

    Parameters
    ----------
    Sig : Signal
        待重采样信号
    type : str, default='spacing'
        重采样方法, 支持:
        - 'spacing'：时域等间隔抽取
        - 'fft'：频域裁剪/补零(支持上采样)
        - 'extreme'：时域极值抽取(用于绘图)
    dt : float, optional
        重采样间隔, 若不指定则与原信号一致
    t0 : float, optional
        重采样起点, 若不指定则与原信号起点一致
    T : float, optional
        重采样长度, 若不指定则采样至信号末尾

    Returns
    -------
    Signal
        重采样后信号
    """
    dt = Sig.t_axis.dt if dt is None else dt
    t0 = Sig.t_axis.t0 if t0 is None else t0
    # 获取重采样起点索引
    start_idx = int(np.ceil((t0 - Sig.t_axis.t0) / Sig.t_axis.dt - 1e-5))
    if start_idx < 0 or start_idx >= len(Sig):
        raise ValueError(f"t0={t0}, idx={start_idx}: 重采样起点超出信号时间轴范围")
    t0 = Sig.t_axis.t0 + start_idx * Sig.t_axis.dt  # 修正重采样起点, 避免浮点数误差
    # 截取重采样数据片段
    if T is None:
        data2rs = Sig._data[start_idx:].copy()
    else:
        N2rs = int(np.ceil(T / (Sig.t_axis.dt) - 1e-5))
        if start_idx + N2rs > len(Sig):
            raise ValueError(f"T={T}, idx={start_idx + N2rs}: 重采样终点超出信号时间轴范围")
        data2rs = Sig._data[start_idx : start_idx + N2rs].copy()
    # 计算重采样后信号点数
    N_in = len(data2rs)
    ratio2rs = Sig.t_axis.dt / dt
    N_out = int(round(N_in * ratio2rs))  # N_out = N_in * (dx_in / dx_out)
    if N_in == N_out:
        ratio2rs = 1.0
    # --------------------------------------------------------------------------------#
    # 下采样
    if ratio2rs < 1:
        if type == "fft":
            # 频域下采样：傅里叶变换后裁剪高频分量
            F_x = np.fft.fft(data2rs)
            idx1 = (N_out + 1) // 2
            idx2 = idx1 + (N_in - N_out)
            F_x_cut = np.zeros(N_out, dtype=complex)
            F_x_cut[:idx1] = F_x[:idx1]
            F_x_cut[idx1:] = F_x[idx2:]
            data2rs = np.ascontiguousarray(np.fft.ifft(F_x_cut).real)
            data2rs *= ratio2rs  # 幅值修正
        elif type == "extreme":
            # 极值法下采样
            M = max(1, min(N_out // 2, N_in))
            seg_len = N_in // M
            seg = data2rs[: seg_len * M].reshape(M, seg_len)
            idx_min = seg.argmin(axis=1)  # 各段极小值位置
            idx_max = seg.argmax(axis=1)  # 各段极大值位置
            rows = np.arange(M)
            is_min_first = idx_min < idx_max  # 判断极值出现先后, 以保持波形走向
            v_min = seg[rows, idx_min]
            v_max = seg[rows, idx_max]
            new_data = np.empty(2 * M, dtype=data2rs.dtype)
            new_data[0::2] = np.where(is_min_first, v_min, v_max)
            new_data[1::2] = np.where(is_min_first, v_max, v_min)
            data2rs = new_data
        elif type == "spacing":
            # 等间隔直接抽取
            idxs = np.linspace(0, N_in, N_out, dtype=int, endpoint=False)
            data2rs = data2rs[idxs]
        else:
            raise ValueError(f"type={type}: 不支持的重采样方法. 使用type=['spacing', 'fft', 'extreme']")
    # --------------------------------------------------------------------------------#
    # 上采样
    elif ratio2rs > 1:
        if type == "fft":
            # 频域上采样：傅里叶变换后补零扩展
            F_x = np.fft.fft(data2rs)
            F_x_pad = np.zeros(N_out, dtype=complex)
            idx1 = (N_in + 1) // 2
            idx2 = idx1 + (N_out - N_in)
            F_x_pad[:idx1] = F_x[:idx1]
            F_x_pad[idx2:] = F_x[idx1:]
            data2rs = np.ascontiguousarray(np.fft.ifft(F_x_pad).real)
            data2rs *= ratio2rs  # 幅值修正
        else:
            raise ValueError(f"dt={dt}, type={type}: 当前采样方法不支持上采样. 使用type='fft'")
    else:
        pass  # 采样频率相同, 不进行重采样
    return Signal(
        axis=t_Axis(len(data2rs), dt=dt, t0=t0),
        data=data2rs,
        name=Sig.name,
        unit=Sig.unit,
        label=Sig.label,
    )


def pad(Sig: Signal, pad_width: int, mode: str = "constant") -> Signal:
    """
    对信号进行边界延拓, 并对沿拓部分进行数据填充

    Parameters
    ----------
    Sig : Signal
        待延拓信号
    pad_width : int
        单边延拓长度
    mode : str, default='constant'
        沿拓数据填充方式, 可选:
        - 'constant'：常数填充, 填充值为0
        - 'edge'：沿拓数据为信号首尾值
        - 'linear_ramp'：沿拓数据为线性斜坡, 斜率由信号首尾值决定
        - 'maximum'：沿拓数据为信号最大值
        - 'mean'：沿拓数据为信号均值
        - 'median'：沿拓数据为信号中位数
        - 'minimum'：沿拓数据为信号最小值
        - 'reflect'：沿拓数据为信号首尾镜像
        - 'symmetric'：沿拓数据为信号首尾对称
        - 'wrap'：沿拓数据为信号首尾循环


    Returns
    -------
    Signal
        延拓后信号
    """
    extend_data = np.pad(Sig._data, pad_width, mode=mode)  # ty:ignore[no-matching-overload]
    t_axis_extend = Sig.t_axis.copy()
    t_axis_extend.N = len(extend_data)
    t_axis_extend.t0 = Sig.t_axis.t0 - pad_width * Sig.t_axis.dt
    return Signal(
        axis=t_axis_extend,
        data=extend_data,
        name=Sig.name,
        unit=Sig.unit,
        label=Sig.label,
    )


def slice(
    Sig: Signal,
    nseg: Optional[int] = None,
    tperseg: Optional[float] = None,
    overlap: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    对信号进行滑窗跳步分段, 段间严格等长且等间隔, 首尾不沿拓

    Parameters
    ----------
    Sig : Signal
        待分段信号
    nseg : int, optional
        分段数, 优先级高于 tperseg
    tperseg : float, optional
        每段时长, 单位为s, 若未指定 nseg 时生效
    overlap : float, default=0.5
        相邻分段的重叠比例, 输入范围: [0, 1).
        推荐不大于0.5, 以避免分段间相关性过高

    Returns
    -------
    seg_data_list : np.ndarray, shape=(nseg, nperseg)
        分段信号数据
    seg_time : np.ndarray
        分段中心时间轴
    """
    if nseg is not None and tperseg is None:
        nhop = int((len(Sig) - 1) / (nseg - 1))
        nperseg = int(nhop / (1 - overlap))
    elif nseg is None and tperseg is not None:
        nperseg = int(tperseg / Sig.t_axis.dt)
        nhop = int(nperseg * (1 - overlap))
        nseg = int((len(Sig) - 1) / nhop + 1)
    else:
        raise ValueError(f"nseg={nseg}, tperseg={tperseg}: 分段数和分段时长参数输入错误. 仅指定其中一个")
    seg_data_list = np.lib.stride_tricks.as_strided(
        Sig._data,  # 直接映射到原始数据, 省去拷贝开销
        shape=(nseg, nperseg),
        strides=(nhop * Sig._data.strides[0], Sig._data.strides[0]),
        writeable=False,  # 设置为只读, 避免修改原始数据
    )
    # 生成分段标志时刻列表
    seg_time = np.arange(seg_data_list.shape[0]) * (seg_data_list.shape[1] * Sig.t_axis.dt) + Sig.t_axis.t0
    return seg_data_list, seg_time
