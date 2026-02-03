"""
# SignalSampling

---

## 可用的接口

    - function:
        - `resample`: 对信号序列 Sig 进行任意时间段的重采样，支持下采样与上采样多种方式。
        - `pad`: 对信号对象进行边界延拓处理，支持镜像延拓和零填充方式
        - `slice`: 对信号进行滑窗跳步分段，首尾段自动延拓
"""

__all__ = ["resample", "pad", "slice"]

from .._Assist_Module.Dependencies import Optional, np
from .core import Signal, t_Axis


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
    对信号序列 Sig 进行任意时间段的重采样，支持下采样与上采样多种方式。

    Parameters
    ----------
    Sig : Signal
        输入信号对象。
    type : str, 默认: 'spacing'
        重采样方法，支持：
        - 'spacing'：等间隔直接抽取（时域抽取）
        - 'fft'：频域重采样（支持上采样与下采样）
        - 'extreme'：极值法（仅下采样）
    dt : float, 可选
        重采样后的采样间隔，若为 None 则与原信号一致。
    t0 : float, 可选
        重采样起始点，若为 None 则与原信号起点一致。
    T : float, 可选
        重采样区间长度，若为 None 则采样至信号末尾。

    Returns
    -------
    Signal
        重采样后的信号对象。

    Raises
    ------
    ValueError
        - 重采样起始点或长度超出原信号范围
        - 极值法采样点数计算错误
        - 不支持的重采样方法
    """
    if dt is None:
        dt = Sig.t_axis.dt
    if t0 is None:
        t0 = Sig.t_axis.t0
    # 获取重采样起始点的索引
    if not Sig.t_axis.t0 <= t0 < (Sig.t_axis.T + Sig.t_axis.t0):
        raise ValueError("重采样起始点不在序列轴范围内")
    else:
        start_idx = int(round((t0 - Sig.t_axis.t0) / Sig.t_axis.dt))

    # 获取重采样数据片段 data2rs
    if T is None:
        data2rs = Sig.data[start_idx:]
    elif T + t0 > Sig.t_axis.T + Sig.t_axis.t0:
        raise ValueError("重采样长度超出序列轴范围")
    else:
        N2rs = int(np.ceil(T / (Sig.t_axis.dt)))  # N = L / dx，向上取整
        data2rs = Sig.data[start_idx : start_idx + N2rs]
    # 获取重采样点数
    N_in = len(data2rs)
    ratio2rs = Sig.t_axis.dt / dt
    N_out = int(N_in * ratio2rs)  # N_out = N_in * (dx_in / dx_out)
    # --------------------------------------------------------------------------------#
    # 对信号片段进行重采样
    if ratio2rs < 1:  # 下采样
        if type == "fft":
            # 频域下采样：傅里叶变换后裁剪高频分量
            F_x = np.fft.fft(data2rs)
            keep = N_out // 2
            F_x_cut = np.zeros(N_out, dtype=complex)
            F_x_cut[:keep] = F_x[:keep]
            F_x_cut[-keep:] = F_x[-keep:]
            data2rs = np.fft.ifft(F_x_cut).real
            data2rs *= ratio2rs  # 幅值修正
        elif type == "extreme":
            # 极值法下采样：每段取极大/极小值
            idxs = np.linspace(0, N_in - 1, (N_out // 2) + 1, dtype=int)
            new_data = []
            for i in range((N_out // 2)):
                seg = data2rs[idxs[i] : idxs[i + 1]]
                new_data.append(np.min(seg))
                new_data.append(np.max(seg))
            # 保证采样点数为 N_out
            if N_out == len(new_data):
                pass
            elif N_out - len(new_data) == 1:
                new_data.append(data2rs[-1])
            else:
                raise ValueError("极值法采样点数计算错误")
            data2rs = np.array(new_data)
        elif type == "spacing":
            # 等间隔直接抽取
            idxs = np.linspace(0, N_in, N_out, dtype=int, endpoint=False)
            data2rs = data2rs[idxs]
        else:
            raise ValueError("下采样方法仅支持'fft', 'extreme', 'spacing'")
    elif ratio2rs > 1:  # 上采样
        if type != "fft":
            raise ValueError("仅支持fft方法进行上采样")
        # 频域上采样：傅里叶变换后补零扩展
        F_x = np.fft.fft(data2rs)
        F_x_pad = np.zeros(N_out, dtype=complex)
        F_x_pad[: N_in // 2] = F_x[: N_in // 2]
        F_x_pad[-N_in // 2 :] = F_x[-N_in // 2 :]
        data2rs = np.fft.ifft(F_x_pad).real
        data2rs *= ratio2rs  # 幅值修正
    else:
        pass  # 采样频率相同, 不进行重采样

    return Signal(
        axis=t_Axis(len(data2rs), dt=dt, t0=t0),
        data=data2rs,
        name=Sig.name,
        unit=Sig.unit,
        label=Sig.label,
    )


def pad(Sig: Signal, length: int, method: str = "mirror") -> Signal:
    """
    对信号对象进行边界延拓处理，支持镜像延拓和零填充方式

    Parameters
    ----------
    Sig : Signal
        输入信号对象
    length : int
        延拓长度，输入范围: >=1
    method : str, default: "mirror"
        延拓方式，支持:
        - "mirror": 镜像延拓（反射填充）
        - "zero": 零填充

    Returns
    -------
    Signal
        延拓后的信号对象

    Raises
    ------
    ValueError
        输入参数`method`不在指定范围内
    """
    data = Sig.data
    if method == "mirror":
        extend_data = np.pad(data, length, mode="reflect")
    elif method == "zero":
        extend_data = np.pad(data, length, mode="constant")
    else:
        raise ValueError(f"{method}: 无效的数据延拓方式")
    t_axis_extend = Sig.t_axis.copy()
    t_axis_extend.N = len(extend_data)
    t_axis_extend.t0 = Sig.t_axis.t0 - length * Sig.t_axis.dt
    Sig_extend = Signal(
        axis=t_axis_extend,
        data=extend_data,
        name=Sig.name,
        unit=Sig.unit,
        label=Sig.label,
    )
    return Sig_extend


def slice(
    Sig: Signal,
    segNum: Optional[int] = None,
    tperseg: Optional[float] = None,
    overlap: float = 0.5,
    pad_mode: str = "constant",
    projection: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    对信号进行滑窗跳步分段，首尾段自动延拓

    Parameters
    ----------
    Sig : Signal
        输入信号对象
    segNum : int, optional
        分段数，优先级高于 tperseg
    tperseg : float, optional
        每段时长[s]，若未指定 segNum 时生效
    overlap : float, default: 0.5
        相邻分段的重叠比例，输入范围: [0, 1)
        推荐不大于0.5, 以避免分段间相关性过高
    pad_mode : str, default: "constant"
        边界填充方式，参考 numpy.pad 的 mode 参数
    projection : bool, default: False
        是否使用内存映射分段机制，可节省内存开销但数据为只读

    Returns
    -------
    seg_data_list : np.ndarray
        分段后的信号数据，shape=(segNum, nperseg)
    seg_time : np.ndarray
        分段中心时间轴

    Notes
    -----
    - 当 projection=True 时,
    方法以分段中心点时刻作为分段整体时刻

    若希望分段时刻覆盖信号末尾时刻(t=(N-1)*dt), 则分段数 segNum 应满足(N-1)/(segNum-1)=nhop为整数

    - 当 projection=False 时,
    方法以分段起始点时刻作为分段整体时刻

    此时实际分段数segNum可能小于指定值, 以防止越界; 故推荐使用 tperseg 指定分段时长
    """
    # 计算分段关键参数: nhop, nperseg
    # N-nhop<(segNum-1)*nhop+1<=N
    # ⇒(N-1)/segNum<nhop<=(N-1)/(segNum-1)
    if segNum is not None:  # 同时指定segNum和tperseg时，以segNum为准
        nhop = int((len(Sig) - 1) / (segNum - 1))  # 向下取整
        nperseg = int(nhop / (1 - overlap))
    elif tperseg is not None:
        nperseg = int(tperseg / Sig.t_axis.dt)  # 向下取整
        nhop = int(nperseg * (1 - overlap))
    else:
        raise ValueError("请指定分段数: segNum 或 分段时长 tperseg ")
    # --------------------------------------------------------------------------------#
    if projection:
        # 以分段起始点时刻作为分段整体时刻
        n = int((len(Sig) - nperseg) / nhop + 1)  # 防止越界
        seg_data_list = np.lib.stride_tricks.as_strided(
            Sig._data,  # 直接映射到原始数据, 省去拷贝开销
            shape=(n, nperseg),
            strides=(nhop * Sig._data.strides[0], Sig._data.strides[0]),
            writeable=False,  # 设置为只读, 避免修改原始数据
        )
    else:
        # 以分段中心点时刻作为分段整体时刻
        n = int((len(Sig) - 1) / nhop + 1)
        # 以nperseg//2填充数据, 分段中心时刻自动保持为0, nhop*dt, 2*nhop*dt, ...
        # 若nperseg为偶数, 则分段中心点取右侧点, 实际分段中心时刻较理想值偏小dt/2
        pad_data = np.pad(Sig._data, nperseg // 2, mode=pad_mode)
        # 建立分段索引列表
        seg_idx_list = np.tile(np.arange(nperseg), (n, 1))
        seg_idx_list += (np.arange(n) * nhop)[:, None]
        # 提取分段信号
        seg_data_list = pad_data[seg_idx_list]
    # 计算分段整体时间轴
    seg_time = np.arange(n) * (nhop * Sig.t_axis.dt) + Sig.t_axis.t0

    return seg_data_list, seg_time
