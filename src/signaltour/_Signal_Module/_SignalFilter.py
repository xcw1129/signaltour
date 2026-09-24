"""
# SignalFilter: 信号滤波模块, 提供FIR/IIR滤波器与中值滤波等去噪方法

    - function:
        - `filtFIR`: 基于有限冲击响应滤波器对信号进行卷积滑窗滤波
        - `filtIIR`: 基于无限冲击响应滤波器对信号进行反馈递归滤波
        - `filtMedian`: 基于中值滤波器对信号进行卷积滑窗滤波
"""

__all__ = ["filtFIR", "filtIIR", "filtMedian"]

from typing import Tuple

from scipy import signal

from ._core import Signal


def filtFIR(
    sig: Signal,
    cutoff: float | Tuple[float, float],
    order: int = 128,
    btype: str = "lowpass",
    window: str = "hamming",
    zero_phase: bool = False,
) -> Signal:
    """
    基于有限冲击响应滤波器对信号进行卷积滑窗滤波

    Parameters
    ----------
    sig : Signal
        待滤波信号
    cutoff : float or tuple[float, float]
        截止频率(Hz), band 类型需提供两个元素
    order : int, default=128
        滤波器阶数
    btype : str, default="lowpass"
        滤波器类型，支持: "lowpass", "highpass", "bandpass", "bandstop"
    window : str, default="hamming"
        理想滤波器加窗类型
    zero_phase : bool, default=False
        是否进行零相移滤波

    Returns
    -------
    Signal
        滤波后信号
    """
    fs = sig.t_axis.fs
    b = signal.firwin(
        numtaps=order,
        cutoff=cutoff,
        window=window,
        pass_zero=btype,
        fs=fs,
    )
    filtered = signal.lfilter(b, [1.0], sig._data) if zero_phase is False else signal.filtfilt(b, [1.0], sig._data)
    return sig.template(filtered)


def filtIIR(
    sig: Signal,
    cutoff: float | Tuple[float, float],
    order: int = 4,
    btype: str = "lowpass",
    ftype: str = "butter",
    zero_phase: bool = False,
) -> Signal:
    """
    基于无限冲击响应滤波器对信号进行反馈递归滤波

    滤波器具体实现通过级联二阶节sos形式, 以提高滤波器的数值稳定性

    Parameters
    ----------
    sig : Signal
        待滤波信号
    cutoff : float or tuple[float, float]
        截止频率(Hz)，band 类型需提供两个元素
    order : int, default=4
        滤波器阶数
    btype : str, default="lowpass"
        滤波器类型，支持: "lowpass", "highpass", "bandpass", "bandstop"
    ftype : str, default="butter"
        滤波器子类型，支持: "butter", "cheby1", "ellip"
    zero_phase : bool, default=False
        是否进行零相位滤波

    Returns
    -------
    Signal
        滤波后信号

    Notes
    -----
    IIR滤波器相比FIR滤波器计算效率更高, 达到相同阻带衰减效果所需的阶数相比FIR滤波器低的多.
    但可能由于非线性相位延迟导致的波形失真和稳定性问题, 建议只在滤波精度要求不高、只关心频谱特性的场景下使用.
    """
    fs = sig.t_axis.fs
    sos = signal.iirfilter(N=order, Wn=cutoff, btype=btype, ftype=ftype, fs=fs, output="sos")
    filtered = signal.sosfilt(sos, sig._data) if zero_phase is False else signal.sosfiltfilt(sos, sig._data)
    return sig.template(filtered)


def filtMedian(
    sig: Signal,
    size: int = 5,
) -> Signal:
    """
    基于中值滤波器对信号进行卷积滑窗滤波

    Parameters
    ----------
    sig : Signal
        待滤波信号
    size : int, default=5
        滤波窗口长度, 必须为正奇数

    Returns
    -------
    Signal
        滤波后信号, 采样信息与输入信号一致

    Notes
    -----
    中值滤波器对脉冲噪声和尖峰干扰有良好的抑制效果，常用于去除信号中的孤立异常点
    """
    filtered = signal.medfilt(sig._data, kernel_size=size)
    return sig.template(filtered)
