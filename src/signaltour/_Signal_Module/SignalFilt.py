"""
# SignalFilt

---

## 可用的接口

    - function:
        - `filtFIR`: 基于有限冲击响应滤波器对信号进行各种类型滤波
        - `filtIIR`: 基于无限冲击响应滤波器对信号进行各种类型滤波
        - `filtMedian`: 基于中值滤波器对信号进行去噪处理
"""

__all__ = ["filtFIR", "filtIIR", "filtMedian"]

from .._Assist_Module.Dependencies import Tuple, signal
from .core import Signal


def filtFIR(
    Sig: Signal,
    cutoff: float | Tuple[float, float],
    order: int = 128,
    btype: str = "lowpass",
    window: str = "hamming",
    zero_phase: bool = False,
) -> Signal:
    """
    基于有限冲击响应滤波器对信号进行各种类型滤波

    Parameters
    ----------
    Sig : Signal
        待滤波信号
    cutoff : float or tuple[float, float]
        截止频率(Hz), band 类型需提供两个元素
    order : int, default: 128
        滤波器阶数
    btype : str, default: "lowpass"
        滤波器类型，支持: "lowpass", "highpass", "bandpass", "bandstop"
    window : str, default: "hamming"
        理想滤波器加窗类型
    zero_phase : bool, default: False
        是否进行零相移滤波

    Returns
    -------
    Signal
        滤波后的信号, 采样信息与输入信号一致

    See Also
    --------
    - scipy.signal.firwin: FIR滤波器系数生成
    - scipy.signal.lfilter: 一般单向滤波计算
    - scipy.signal.filtfilt: 零相位滤波计算
    """
    fs = Sig.t_axis.fs
    b = signal.firwin(
        numtaps=order + 1,
        cutoff=cutoff,
        window=window,
        pass_zero=btype,
        fs=fs,
    )
    filtered = signal.lfilter(b, [1.0], Sig._data) if zero_phase is False else signal.filtfilt(b, [1.0], Sig._data)
    return Sig.template(filtered)


def filtIIR(
    Sig: Signal,
    cutoff: float | Tuple[float, float],
    order: int = 4,
    btype: str = "lowpass",
    ftype: str = "butter",
    zero_phase: bool = False,
) -> Signal:
    """
    基于无限冲击响应滤波器对信号进行各种类型滤波

    滤波器具体实现通过级联二阶节sos形式, 以提高滤波器的数值稳定性

    Parameters
    ----------
    Sig : Signal
        待滤波信号
    cutoff : float or tuple[float, float]
        截止频率（Hz），band 类型需提供两个元素
    order : int, default: 4
        滤波器阶数
    btype : str, default: "lowpass"
        滤波器类型，支持: "lowpass", "highpass", "bandpass", "bandstop"
    ftype : str, default: "butter"
        滤波器子类型，支持: "butter", "cheby1", "ellip"
    zero_phase : bool, default: False
        是否进行零相位滤波

    Returns
    -------
    Signal
        滤波后的信号, 采样信息与输入信号一致

    See Also
    --------
    - scipy.signal.iirfilter: IIR滤波器系数生成
    - scipy.signal.sosfilt: 一般单向滤波计算
    - scipy.signal.sosfiltfilt: 零相位滤波计算

    Notes
    -----
    IIR滤波器相比FIR滤波器计算效率更高, 达到相同阻带衰减效果所需的阶数相比FIR滤波器低的多.
    但可能由于非线性相位延迟导致的波形失真和稳定性问题, 建议只在滤波精度要求不高、只关心频谱特性的场景下使用.
    """
    fs = Sig.t_axis.fs
    sos = signal.iirfilter(N=order, Wn=cutoff, btype=btype, ftype=ftype, fs=fs, output="sos")
    filtered = signal.sosfilt(sos, Sig._data) if zero_phase is False else signal.sosfiltfilt(sos, Sig._data)
    return Sig.template(filtered)


def filtMedian(
    Sig: Signal,
    size: int = 5,
) -> Signal:
    """
    基于中值滤波器对信号进行去噪处理

    Parameters
    ----------
    Sig : Signal
        待滤波信号
    size : int, default: 5
        滤波窗口长度, 必须为正奇数

    Returns
    -------
    Signal
        滤波后的信号, 采样信息与输入信号一致

    See Also
    --------
    - scipy.signal.medfilt: 一维中值滤波实现

    Notes
    -----
    中值滤波器对脉冲噪声和尖峰干扰有良好的抑制效果，常用于去除信号中的孤立异常点
    """
    filtered = signal.medfilt(Sig._data, kernel_size=size)
    return Sig.template(filtered)
